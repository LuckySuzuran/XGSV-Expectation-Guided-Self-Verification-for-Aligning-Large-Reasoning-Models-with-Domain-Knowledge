import torch
import pandas as pd
from torch_geometric.data import Data
from typing import Dict, Tuple
import numpy as np
from pcst_fast import pcst_fast
import torch_geometric.nn as gnn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time

def load_graph_data(
    entity_file: str,
    relation_file: str,
    edge_file: str,
    node_embedding_file: str,
    edge_embedding_file: str
) -> Tuple[Data, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int]]:
    entity_to_id = {}
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_to_id[parts[0]] = int(parts[1])
    
    relation_to_id = {}
    with open(relation_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation_to_id[parts[0]] = int(parts[1])
    
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    textual_nodes = pd.DataFrame({
        'entity_name': [id_to_entity[i] for i in range(len(entity_to_id))]
    })
    
    edges = []
    textual_edges = []
    relation_ids = []
    with open(edge_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                src, rel, dst = parts
                if src in entity_to_id and dst in entity_to_id and rel in relation_to_id:
                    edges.append({
                        'src': entity_to_id[src],
                        'dst': entity_to_id[dst],
                        'relation': rel,
                        'relation_id': relation_to_id[rel]
                    })
                    textual_edges.append({
                        'src': src,
                        'dst': dst,
                        'relation': rel,
                        'relation_id': relation_to_id[rel]
                    })
                    relation_ids.append(relation_to_id[rel])
    textual_edges = pd.DataFrame(textual_edges)
    
    node_embeddings = torch.load(node_embedding_file)
    edge_embeddings = torch.load(edge_embedding_file)
    
    edge_index = torch.tensor(
        [[e['src'] for e in edges], [e['dst'] for e in edges]], 
        dtype=torch.long
    )

    edge_attr = edge_embeddings[relation_ids]
    
    graph = Data(
        x=node_embeddings,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(entity_to_id)
    )
    
    return graph, textual_nodes, textual_edges, entity_to_id, relation_to_id

def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'relation', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        # cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]

    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'relation', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import os
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
def convert_data_to_bfloat16(data):
    if hasattr(data, 'x') and data.x is not None and torch.is_floating_point(data.x):
        data.x = data.x.to(dtype=torch.bfloat16)
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None and torch.is_floating_point(data.edge_attr):
        data.edge_attr = data.edge_attr.to(dtype=torch.bfloat16)
    
    return data

class GraphLLM(torch.nn.Module):
    def __init__(self, llm_model_path, if_llm_frozen, graph, textual_nodes, textual_edges, llm_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True)

        if if_llm_frozen == True:
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        self.model = model

        self.graph_encoder = gnn.GCNConv(
            in_channels=4096,
            out_channels=4096
        )
        if llm_name == "dpsk":
            self.projector = nn.Sequential(
                nn.Linear(4096, 2048),
                nn.Sigmoid(),
                nn.Linear(2048, 3584), #qwen3-8B 4096
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(4096, 2048),
                nn.Sigmoid(),
                nn.Linear(2048, 4096), #qwen3-8B 4096
            )            

        self.word_embedding = self.model.model.get_input_embeddings()
        
        self.graph = graph
        self.textual_nodes = textual_nodes
        self.textual_edges = textual_edges
        
    def forward(self, sample_ids, question_embeddings, total_text_ids, total_text_attn_masks, labels):
        question_embeddings = question_embeddings.to(torch.bfloat16)
    
        batch_size = sample_ids.shape[0]
        sub_graphs = []
        for b in range(batch_size):
            sub_graph,_ = retrieval_via_pcst(self.graph, question_embeddings[b].cpu().to(torch.float32), self.textual_nodes, self.textual_edges,
                                             topk=10, topk_e=6, cost_e=0.5)
            
            sub_graphs.append(convert_data_to_bfloat16(sub_graph).to("cuda"))
        gnn_output = []
        for data in sub_graphs:
            # out = self.graph_encoder(data.x, data.edge_index, data.edge_attr)
            out = self.graph_encoder(data.x, data.edge_index)
            out = nn.ReLU()(out)
            out = torch.mean(out, dim = 0)
            gnn_output.append(out)
        graph_embedding = torch.stack(gnn_output) # [batch_size, n_concept, hidden_size]
        graph_embedding = self.projector(graph_embedding)
        graph_embedding = nn.ReLU()(graph_embedding)

        input_embeds = self.word_embedding(total_text_ids)
        
        combined_embeddings = torch.cat([
            graph_embedding.unsqueeze(1),  # [batch_size, 1, hidden_size]
            input_embeds
        ], dim=1)

        combined_attention_mask = torch.cat([
            torch.ones(batch_size, 1, device=total_text_attn_masks.device),
            total_text_attn_masks
        ], dim=1)
        extended_labels = torch.full((batch_size, 1), fill_value=-100, device=labels.device, dtype=labels.dtype)
        combined_labels = torch.cat([extended_labels, labels], dim=1)

        model_outputs = self.model(
        inputs_embeds=combined_embeddings,
        attention_mask=combined_attention_mask,
        labels=combined_labels
        )
        
        return model_outputs.loss
    
    def get_graph_embeddings(self, question_embedding):
        with torch.no_grad():
            sub_graph,_ = retrieval_via_pcst(self.graph, question_embedding, self.textual_nodes, self.textual_edges,
                                             topk=10, topk_e=6, cost_e=0.5)
            data = sub_graph.to("cuda")
            # out = self.graph_encoder(data.x, data.edge_index, data.edge_attr)
            out = self.graph_encoder(data.x, data.edge_index)
            out = nn.ReLU()(out)
            out = torch.mean(out, dim = 0)
            graph_embedding = out
            # graph_embedding = torch.stack(gnn_output) # [batch_size, n_concept, hidden_size]
            graph_embedding = self.projector(graph_embedding)
            graph_embedding = nn.ReLU()(graph_embedding)
        return graph_embedding


def preproces_training_data(subgraph_desc, instruction, question, cot, answer, tokenizer, model, max_length = 4096):
    if model == "deepseek-r1-distill-qwen-7B":
        if len(subgraph_desc) > 1000:
            subgraph_desc = subgraph_desc[:1000]
        augmented_system_prompt = f"{instruction}. You can refer to the following knowledge: {subgraph_desc}."
        messages = [{"role":"system", "content":augmented_system_prompt}, {"role":"user","content":question}]
        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
        if answer !=None:
            if len(cot) >=4300:
                cot = cot[:4300]
            total_text = text + f"{cot}\n</think>\n\n{answer}<｜end▁of▁sentence｜>"
            tokenizer.padding_side = "right"
            tokenizer.pad_token = tokenizer.eos_token
            encoding = tokenizer(total_text, return_tensors = "pt", add_special_tokens=False,
                      padding="max_length", truncation=True, max_length=max_length)
            total_text_ids = encoding["input_ids"]
            total_text_attn_masks = encoding["attention_mask"]

            labels = total_text_ids.clone()
            think_ids = tokenizer.convert_tokens_to_ids("</think>")
            think_index = total_text_ids[0].tolist().index(think_ids)
            labels[:, :think_index] = -100
            pure_total_text_ids = tokenizer(total_text, add_special_tokens=False)["input_ids"]
            pure_total_text_ids_len = len(pure_total_text_ids)
            labels[:, pure_total_text_ids_len:] = -100

            return total_text_ids, total_text_attn_masks, labels
        else:
            encoding = tokenizer(text, return_tensors = "pt")
            text_ids = encoding["input_ids"]
            return text_ids

    elif model == "qwen3-8B":
        if len(subgraph_desc) > 2500:
            subgraph_desc = subgraph_desc[:2500]
        augmented_system_prompt = f"{instruction}. You can refer to the following knowledge: {subgraph_desc}."
        messages = [{"role":"system", "content":augmented_system_prompt}, {"role":"user","content":question}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True)
        if answer != None:            
            total_text = text + f"<think>\n{cot}\n</think>\n\n{answer}<|im_end|>"
            tokenizer.pad_token = tokenizer.eos_token
            encoding = tokenizer(total_text, return_tensors = "pt", add_special_tokens=False,
                      padding="max_length", truncation=True, max_length=max_length)
            total_text_ids = encoding["input_ids"]
            total_text_attn_masks = encoding["attention_mask"]

            labels = total_text_ids.clone()
            think_ids = tokenizer.convert_tokens_to_ids("</think>")
            think_index = total_text_ids[0].tolist().index(think_ids)
            labels[:, :think_index] = -100
            pure_total_text_ids = tokenizer(total_text, add_special_tokens=False)["input_ids"]
            pure_total_text_ids_len = len(pure_total_text_ids)
            labels[:, pure_total_text_ids_len:] = -100

            return total_text_ids, total_text_attn_masks, labels
        else:
            encoding = tokenizer(text, return_tensors = "pt")
            text_ids = encoding["input_ids"]
            return text_ids
from torch.utils.data import Dataset, DataLoader

class QADataset(Dataset):
    def __init__(self, graph, textual_nodes, textual_edges, question_embeddings,
                 instruction, questions, model_cots, answers,
                 reasoning_llm_path, model_name, max_length):
        self.graph = graph
        self.textual_nodes = textual_nodes
        self.textual_edges = textual_edges
        self.question_embeddings = question_embeddings
        
        self.instruction = instruction
        self.questions = questions
        self.model_cots = model_cots
        self.answers = answers

        self.tokenizer = AutoTokenizer.from_pretrained(reasoning_llm_path, trust_remote_code=True)
        self.max_length = max_length
        self.model_name = model_name
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        cot = self.model_cots[idx]
        answer = self.answers[idx]
        question_embedding = self.question_embeddings[idx]
        _, sub_graph_desc = retrieval_via_pcst(self.graph, question_embedding, self.textual_nodes,
                                                       self.textual_edges, topk=10, topk_e=6, cost_e=0.5)
        
        total_text_ids, total_text_attn_masks, labels = preproces_training_data(sub_graph_desc, self.instruction, question, cot, answer,
                                                                                self.tokenizer, self.model_name, self.max_length)
        return {
            'sample_index': idx,
            'question_embedding': question_embedding,
            'total_text_ids':total_text_ids,
            'total_text_attn_masks':total_text_attn_masks,
            'labels':labels
        }

def prepare_data(graph, textual_nodes, textual_edges, question_embeddings,
                 instruction, questions, model_cots, answers,
                 reasoning_llm_path, model_name, max_length, batch_size=4, num_workers=2):
    dataset = QADataset(graph, textual_nodes, textual_edges, question_embeddings,
                        instruction, questions, model_cots, answers,
                        reasoning_llm_path, model_name, max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: {
            'sample_index': torch.tensor([item['sample_index'] for item in batch]),
            'question_embedding': torch.cat([item['question_embedding'] for item in batch], dim=0),
            'total_text_ids': torch.cat([item['total_text_ids'] for item in batch], dim=0),
            'total_text_attn_masks': torch.cat([item['total_text_attn_masks'] for item in batch], dim=0),
            'labels': torch.cat([item['labels'] for item in batch], dim=0)
        }
    )
    return dataloader

def train_model(model,
                graph, textual_nodes, textual_edges, question_embeddings,
                instruction, train_questions, model_cots,train_answers,
                reasoning_llm_path, config):
    total_start_time = time.time()

    train_loader = prepare_data(graph, textual_nodes, textual_edges, question_embeddings,
                                instruction, train_questions, model_cots, train_answers,
                                reasoning_llm_path, config["llm_name"], config["max_length"],
                                batch_size=config['batch_size'])
    
    total_batches = len(train_loader)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    model = nn.DataParallel(model)
    model = model.to(config['device'])
    
    batch_losses = []
    epoch_losses = []
    
    for epoch in range(config['epochs']):
        print(f"\n[Epoch {epoch+1}/{config['epochs']}]")
        model.train()
        
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, 1):            
            sample_ids = batch["sample_index"].to(config["device"])
            total_text_ids = batch["total_text_ids"].to(config["device"])
            total_text_attn_masks = batch["total_text_attn_masks"].to(config["device"])
            labels = batch["labels"].to(config["device"])
            question_embeddings = batch["question_embedding"].to(config["device"])
            outputs = model(sample_ids, question_embeddings, total_text_ids, total_text_attn_masks, labels)
            
            if torch.cuda.device_count() > 1 and config['num_gpus'] > 1:
                loss = outputs.mean()
            else:
                loss = outputs
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                accumulated_time = time.time() - total_start_time
                print(f"  Batch {batch_idx}/{total_batches} | Loss: {batch_loss:.4f} | Accumulated Time: {accumulated_time:.2f}s")
        
        avg_epoch_loss = epoch_loss / total_batches
        epoch_losses.append(avg_epoch_loss)
        
#         epoch_time = time.time() - epoch_start_time
#         epoch_times.append(epoch_time)
        # print(f"  Epoch {epoch+1}/{config['epochs']} | Average Loss: {avg_epoch_loss:.4f} | Time: {e:.2f}s")
        print(f"  Epoch {epoch+1}/{config['epochs']} | Average Loss: {avg_epoch_loss:.4f}")
    
    total_train_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_train_time:.2f}s")
    
    return {
        'model': model,
        'batch_losses': batch_losses,
        'epoch_losses': epoch_losses,
        'total_train_time': total_train_time
    }

from peft import PeftModel
import json
import os
import json
import torch

def save_complete_model(model, save_dir, config, if_llm_frozen=False):
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    if if_llm_frozen:
        custom_layers = {
            'graph_encoder': model_module.graph_encoder.state_dict(),
            'projector': model_module.projector.state_dict(),
        }
        custom_layers_path = os.path.join(save_dir, "custom_layers.pt")
        torch.save(custom_layers, custom_layers_path)
    else:
        # lora model
        lora_save_path = os.path.join(save_dir, "lora_adapter")
        os.makedirs(lora_save_path, exist_ok=True)
        model_module.model.save_pretrained(lora_save_path)
        
        custom_layers = {
            'graph_encoder': model_module.graph_encoder.state_dict(),
            'projector': model_module.projector.state_dict()
        }
        custom_layers_path = os.path.join(save_dir, "custom_layers.pt")
        torch.save(custom_layers, custom_layers_path)
    
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    model_module.tokenizer.save_pretrained(tokenizer_path)
    
    save_config = config.copy()
    save_config['if_llm_frozen'] = if_llm_frozen
    with open(os.path.join(save_dir, "training_config.json"), 'w', encoding='utf-8') as f:
        json.dump(save_config, f, indent=2, ensure_ascii=False)

def load_prompt_tuning_model(llm_model_path, save_dir, llm_name, graph, textual_nodes, textual_edges):
    with open(os.path.join(save_dir, "training_config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_dir, "tokenizer"), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    from g_retriever import GraphLLM
    complete_model = GraphLLM(
        llm_model_path=llm_model_path,
        if_llm_frozen=True,
        graph=graph,
        textual_nodes=textual_nodes,
        textual_edges=textual_edges,
        llm_name=llm_name
    )
    
    custom_layers_path = os.path.join(save_dir, "custom_layers.pt")
    custom_layers = torch.load(custom_layers_path)
    
    complete_model.graph_encoder.load_state_dict(custom_layers['graph_encoder'])
    complete_model.projector.load_state_dict(custom_layers['projector'])
    
    complete_model.model = base_model
    complete_model.tokenizer = tokenizer
    
    return complete_model
    
def load_lora_model(llm_model_path, save_dir,llm_name, graph, textual_nodes, textual_edges):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_dir, "tokenizer"), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    lora_save_path = os.path.join(save_dir, "lora_adapter")
    model = PeftModel.from_pretrained(base_model, lora_save_path)
    
    from g_retriever import GraphLLM  # 替换为实际的模块名
    complete_model = GraphLLM(
        llm_model_path=llm_model_path,
        if_llm_frozen= False,
        graph=graph,
        textual_nodes=textual_nodes,
        textual_edges=textual_edges,
        llm_name=llm_name
    )
    
    custom_layers_path = os.path.join(save_dir, "custom_layers.pt")
    custom_layers = torch.load(custom_layers_path)
    complete_model.graph_encoder.load_state_dict(custom_layers['graph_encoder'])
    complete_model.projector.load_state_dict(custom_layers['projector'])
    
    complete_model.model = model
    complete_model.tokenizer = tokenizer
    
    return complete_model