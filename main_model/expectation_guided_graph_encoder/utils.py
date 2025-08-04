import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity
from torch import Tensor

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def find_similar_keywords(query_vector: torch.Tensor,
                          vector_library: torch.Tensor, 
                          top_k: int = 5):
    if query_vector.dim() == 1:
        query_vector = query_vector.unsqueeze(0)
    
    similarities = cosine_similarity(query_vector, vector_library, dim=1)
    
    top_indices = torch.topk(similarities, k=min(top_k, vector_library.shape[0])).indices
    return top_indices

def get_question_length(labels):
    batch_size, seq_len = labels.shape
    question_lens = []
    
    for b in range(batch_size):
        label = labels[b]
        
        answer_start = (label != -100).nonzero(as_tuple=True)[0]
        
        if len(answer_start) > 0:
            question_lens.append(answer_start[0].item())
        else:
            question_lens.append(seq_len)
    
    return question_lens

def preprocess_training_data(instruction, question, answer, tokenizer, model_name, max_length = 1024):
    if model_name == "qwen2.5-7B-Instruct":
        messages = [{"role":"system", "content":instruction}, {"role":"user","content":question}]
        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
        total_text = text + f"{answer}<|im_end|>"

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        encoding = tokenizer(total_text, return_tensors = "pt", add_special_tokens=False,
                  padding="max_length", truncation=True, max_length=max_length)
        total_text_ids = encoding["input_ids"]
        total_text_attn_masks = encoding["attention_mask"]
        
        labels = total_text_ids.clone()
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompt_ids_len = len(prompt_ids)
        labels[:, :prompt_ids_len] = -100
        
        pure_total_text_ids = tokenizer(total_text, add_special_tokens=False)["input_ids"]
        pure_total_text_ids_len = len(pure_total_text_ids)
        labels[:, pure_total_text_ids_len:] = -100

        return total_text_ids, total_text_attn_masks, labels
    elif model_name == "qwen3-8B":
        messages = [{"role":"system", "content":instruction}, {"role":"user","content":question}]
        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True, enable_thinking = False)
        total_text = text + f"{answer}<|im_end|>"

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        encoding = tokenizer(total_text, return_tensors = "pt", add_special_tokens=False,
                  padding="max_length", truncation=True, max_length=max_length)
        total_text_ids = encoding["input_ids"]
        total_text_attn_masks = encoding["attention_mask"]
        
        labels = total_text_ids.clone()
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompt_ids_len = len(prompt_ids)
        labels[:, :prompt_ids_len] = -100
        
        pure_total_text_ids = tokenizer(total_text, add_special_tokens=False)["input_ids"]
        pure_total_text_ids_len = len(pure_total_text_ids)
        labels[:, pure_total_text_ids_len:] = -100

        return total_text_ids, total_text_attn_masks, labels

def count_max_tokens(instructions, questions, answers, tokenizer, model_name):
    if model_name == "qwen2.5-7B-Instruct":
        num_tokens = []
        for idx in range(len(questions)):
            messages = [{"role":"system", "content":instructions[idx]}, {"role":"user","content":questions[idx]}]
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
            total_text = text + f"{answers[idx]}<|im_end|>"
            tokenizer.padding_side = "right"
            tokenizer.pad_token = tokenizer.eos_token
            encoding = tokenizer(total_text, add_special_tokens=False)
            total_text_ids = encoding["input_ids"]
            num_tokens.append(len(total_text_ids))
        return max(num_tokens)
    
    elif model_name == "qwen3-8B":
        num_tokens = []
        for idx in range(len(questions)):     
            messages = [{"role":"system", "content":instructions[idx]}, {"role":"user","content":questions[idx]}]
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True, enable_thinking = False)
            total_text = text + f"{answers[idx]}<|im_end|>"
            tokenizer.padding_side = "right"
            tokenizer.pad_token = tokenizer.eos_token
            encoding = tokenizer(total_text, add_special_tokens=False)
            total_text_ids = encoding["input_ids"]
            num_tokens.append(len(total_text_ids))
        return max(num_tokens)
    
class PreprocessedQADataset(Dataset):
    def __init__(self, instruction, questions, answers, tokenizer, llm_name, max_length):
        self.instruction = instruction
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.llm_name = llm_name

        self.preprocessed_data = self._preprocess_all_data()
        
    def _preprocess_all_data(self):
        preprocessed_data = []
        
        for i, (question, answer) in enumerate(zip(self.questions, self.answers)):
            total_text_ids, total_text_attn_masks, labels = preprocess_training_data(
                self.instruction, question, answer, self.tokenizer, self.llm_name, self.max_length
            )
            preprocessed_data.append({
                'question_index': i,
                'total_text_ids': total_text_ids,
                'total_text_attn_masks': total_text_attn_masks,
                'labels': labels
            })
            
        print(f"Dataset's num is {len(preprocessed_data)}")
        return preprocessed_data
    
    def __len__(self):
        return len(self.preprocessed_data)
    
    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

def prepare_data(instruction, questions, answers, tokenizer, model_name, max_length, batch_size=4, num_workers=2):
    dataset = PreprocessedQADataset(instruction, questions, answers, tokenizer, model_name, max_length)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: {
            'question_index': torch.tensor([item['question_index'] for item in batch]),
            'total_text_ids': torch.cat([item['total_text_ids'] for item in batch], dim=0),
            'total_text_attn_masks': torch.cat([item['total_text_attn_masks'] for item in batch], dim=0),
            'labels': torch.cat([item['labels'] for item in batch], dim=0)
        }
    )
    return dataloader

def create_optimizer(model, base_lr=1e-5, gnn_projector_lr=1e-4):
    # hierachical parameter groups for different learning rates
    param_groups = [
        {
            'params': list(model.gcn1.parameters()) + 
                      list(model.proj2llmspace.parameters()),
            'lr': gnn_projector_lr
        },
        {
            'params': list(model.query_proj.parameters()) +
                      list(model.key_proj.parameters()) +
                      list(model.value_proj.parameters()),
            'lr': base_lr
        },
        {
            'params': list(model.main_diag.parameters()) +
                      list(model.anti_diag.parameters()),
            'lr': base_lr
        },
    ]
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=0.01
    )
    
    return optimizer


def train_model(model, instruction, train_questions, train_answers, tokenizer, config=None):
    if config is None:
        config = {
            'batch_size': 4,
            'epochs': 3,
            'learning_rate': 5e-5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_gpus': 1,
        }
    total_start_time = time.time()

    train_loader = prepare_data(
        instruction,
        train_questions, 
        train_answers,
        tokenizer,
        config["llm_name"],
        config["max_length"],
        batch_size=config['batch_size']
    )
    
    total_batches = len(train_loader)
    
    # optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    optimizer = create_optimizer(model, base_lr=config["base_lr"], gnn_projector_lr=config["gnn_projector_lr"])
    
    if torch.cuda.device_count() > 1 and config['num_gpus'] > 1:
        model = nn.DataParallel(model, device_ids=list(range(config['num_gpus'])))
    model = model.to(config['device'])
    
    batch_losses = []
    
    epoch_losses = []
    
    for epoch in range(config['epochs']):
        # start_time = time.time()
        print(f"\n[Epoch {epoch+1}/{config['epochs']}]")
        model.train()
        
        epoch_loss = 0.0
        
        # 
        for batch_idx, batch in enumerate(train_loader, 1):
            # batch_start_time = time.time()
            
            question_ids = batch["question_index"].to(config["device"])
            total_text_ids = batch["total_text_ids"].to(config["device"])
            total_text_attn_masks = batch["total_text_attn_masks"].to(config["device"])
            labels = batch["labels"].to(config["device"])
            
            outputs = model(question_ids, total_text_ids, total_text_attn_masks, labels)
            
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
            
            # batch_time = time.time() - batch_start_time
            # batch_times.append(batch_time)
            
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                accumulated_time = time.time() - total_start_time
                print(f"  Batch {batch_idx}/{total_batches} | Loss: {batch_loss:.4f} | Accumulated Time: {accumulated_time:.2f}s")
                
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient norm: {total_norm:.4f}")
                if total_norm < 1e-6:
                    print("Warning: Gradient vanishing detected!")
        avg_epoch_loss = epoch_loss / total_batches
        epoch_losses.append(avg_epoch_loss)
        
        print(f"  Epoch {epoch+1}/{config['epochs']} | Average Loss: {avg_epoch_loss:.4f}")
    
    total_train_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_train_time:.2f}s")
    
    return {
        'model': model,
        'batch_losses': batch_losses,
        'epoch_losses': epoch_losses,
        'total_train_time': total_train_time
    }