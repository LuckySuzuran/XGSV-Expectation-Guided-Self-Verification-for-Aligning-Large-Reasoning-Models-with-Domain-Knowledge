import os
import torch
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch import Tensor
from torch.nn.functional import cosine_similarity
import pickle as pkl

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

# Model Saving and Loading Functions
def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    custom_state_dict = {}
    for name, param in model.state_dict().items():
        # Exclude all LLM-related parameters
        if not name.startswith('llm.'):
            custom_state_dict[name] = param
    
    custom_save_dir = os.path.join(os.path.dirname(save_path), "custom_params")
    torch.save(custom_state_dict, custom_save_dir)
    print(f"Model custom parameters saved to: {custom_save_dir}")

    # Save LoRA parameters separately if used
    if hasattr(model.llm, 'save_pretrained'):
        lora_save_dir = os.path.join(os.path.dirname(save_path), "lora_params")
        os.makedirs(lora_save_dir, exist_ok=True)
        model.llm.save_pretrained(lora_save_dir)
        print(f"LoRA parameters saved to: {lora_save_dir}")


def load_model(model, load_path):
    # Load custom parameters
    custom_load_dir = os.path.join(os.path.dirname(load_path), "custom_params")
    state_dict = torch.load(custom_load_dir, map_location=model.device)
    lora_load_dir = os.path.join(os.path.dirname(load_path), "lora_params")
    
    if hasattr(model.llm, 'from_pretrained') and os.path.exists(lora_load_dir):
        model.llm = model.llm.from_pretrained(model.llm.base_model, lora_load_dir)
        model.llm.to(model.device)
        print(f"LoRA parameters loaded from: {lora_load_dir}")
    
    # Load custom component parameters
    model.load_state_dict(state_dict, strict=False)
    print(f"Model custom parameters loaded from: {load_path}")
    
    return model


# Matrix Conversion Functions
def matrix_to_sparse(matrix: torch.Tensor) -> dict:
    """
    Convert dense matrix to sparse representation
    
    Args:
    matrix: Input dense matrix (torch.Tensor)
    
    Returns:
    Dictionary containing sparse representation with keys:
        - indices: Coordinates of non-zero elements (2 x nnz)
        - values: Values of non-zero elements (nnz)
        - size: Original matrix dimensions (tuple)
    """
    # Use PyTorch's sparse COO format
    sparse_matrix = matrix.to_sparse()
    
    # Extract key information for sparse representation
    return {
        'indices': sparse_matrix.indices(),
        'values': sparse_matrix.values(),
        'size': sparse_matrix.size()
    }


def sparse_to_matrix(sparse_repr: dict) -> torch.Tensor:
    """
    Reconstruct dense matrix from sparse representation
    
    Args:
    sparse_repr: Dictionary containing sparse representation with:
        - indices: Coordinates of non-zero elements (2 x nnz)
        - values: Values of non-zero elements (nnz)
        - size: Original matrix dimensions (tuple)
    
    Returns:
    Reconstructed dense matrix (torch.Tensor)
    """
    # Reconstruct sparse tensor using COO format
    sparse_matrix = torch.sparse_coo_tensor(
        indices=sparse_repr['indices'],
        values=sparse_repr['values'],
        size=sparse_repr['size']
    )
        
    # Convert back to dense matrix
    return sparse_matrix.to_dense()


# Data Preprocessing Functions
def preproces_training_data(instruction, question, cot, answer, tokenizer, model, max_length=4096):
    if model == "deepseek-r1-distill-qwen-7B":
        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if answer is not None:
            if len(cot) >= 4000:
                cot = cot[:4000]
                
            total_text = text + f"{cot}\n</think>\n\n{answer} <｜end▁of▁sentence｜>"
            tokenizer.padding_side = "right"
            tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token
            
            encoding = tokenizer(
                total_text, 
                return_tensors="pt", 
                add_special_tokens=False,
                padding="max_length", 
                truncation=True, 
                max_length=max_length
            )
            
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
            total_text = text + f"{cot}\n</think>"
            encoding = tokenizer(total_text, return_tensors="pt")
            text_ids = encoding["input_ids"]
            return text_ids

    elif model == "qwen3-8B":
        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        if answer is not None:            
            total_text = text + f"</think>\n{cot}\n</think>\n\n{answer} <|im_end|>"
            tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token
            
            encoding = tokenizer(
                total_text, 
                return_tensors="pt", 
                add_special_tokens=False,
                padding="max_length", 
                truncation=True, 
                max_length=max_length
            )
            
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
            total_text = text + f"</think>\n{cot}\n</think>"
            encoding = tokenizer(total_text, return_tensors="pt")
            text_ids = encoding["input_ids"]
            return text_ids


# Dataset and DataLoader Classes
class QADataset(Dataset):
    def __init__(self, instruction, questions, model_cots, answers, sparse_expected_relations, 
                 reasoning_llm_path, model_name, max_length):
        """
        Question answering data preparation class
        
        Args:
            questions: List of questions (List[str])
            model_cots: List of model-generated chain-of-thoughts (List[str])
            answers: Corresponding list of answers (List[str])
        """
        self.instruction = instruction
        self.questions = questions
        self.answers = answers
        self.model_cots = model_cots
        self.tokenizer = AutoTokenizer.from_pretrained(reasoning_llm_path, trust_remote_code=True)
        self.max_length = max_length
        self.model_name = model_name
        self.sparse_expected_relations = sparse_expected_relations

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        cot = self.model_cots[idx]
        answer = self.answers[idx]
        sparse_expected_relations = self.sparse_expected_relations[idx]
        expected_relations = sparse_to_matrix(sparse_expected_relations)
        
        total_text_ids, total_text_attn_masks, labels = preproces_training_data(
            self.instruction, question, cot, answer, self.tokenizer, self.model_name, self.max_length
        )
        
        return {
            'sample_index': idx,
            'total_text_ids': total_text_ids,
            'total_text_attn_masks': total_text_attn_masks,
            'labels': labels,
            'relation_matrix': expected_relations
        }


def prepare_data(instruction, questions, model_cots, answers, sparse_expected_relations, 
                reasoning_llm_path, model_name, max_length, batch_size=4, num_workers=2):
    """
    Prepare question answering data loader (supports multi-GPU training)
    
    Args:
        questions: List of questions (List[str])
        answers: Corresponding list of answers (List[str])
        llm_path: Path to pre-trained model
        batch_size: Batch size
        num_workers: Number of data loading threads, adjust based on CPU cores
        
    Returns:
        dataloader: Data loader
    """
    dataset = QADataset(
        instruction, questions, model_cots, answers, sparse_expected_relations, 
        reasoning_llm_path, model_name, max_length
    )
    
    # Use larger batch_size and set num_workers for multi-GPU training
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Accelerate data transfer to GPU
        collate_fn=lambda batch: {
            'sample_index': torch.tensor([item['sample_index'] for item in batch]),
            'total_text_ids': torch.cat([item['total_text_ids'] for item in batch], dim=0),
            'total_text_attn_masks': torch.cat([item['total_text_attn_masks'] for item in batch], dim=0),
            'labels': torch.cat([item['labels'] for item in batch], dim=0),
            'relation_matrix': torch.cat([item['relation_matrix'].unsqueeze(0) for item in batch], dim=0),
        }
    )
    
    return dataloader


# Training Utilities
def get_layered_lr_optimizer(model, lora_lr=3e-4, custom_layer_lr=1e-5, weight_decay=0.01):
    """
    Set different learning rates for different model layers
    - Higher learning rate for LoRA layers
    - Lower learning rate for other custom layers (GCN, projection layers, etc.)
    
    Args:
        model: reasoning_model instance
        lora_lr: Learning rate for LoRA parameters (higher)
        custom_layer_lr: Learning rate for other custom layers (lower)
        weight_decay: Weight decay coefficient
        
    Returns:
        Configured AdamW optimizer
    """
    # Define parameter groups
    param_groups = []
    
    # 1. Handle LoRA parameters (if enabled)
    if hasattr(model.llm, 'named_parameters') and hasattr(model.llm, 'peft_config'):
        lora_params = []
        for name, param in model.llm.named_parameters():
            if param.requires_grad:
                lora_params.append(param)
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': lora_lr,
                'weight_decay': weight_decay,
                'name': 'lora_parameters'
            })
            print(f"Set up LoRA parameter group, learning rate: {lora_lr}, parameter count: {sum(p.numel() for p in lora_params)}")
    
    # 2. Handle other custom trainable layers
    # These include GCN, projection layers, attention projections, etc., excluding LLM parts
    custom_params = []
    for name, param in model.named_parameters():
        # Exclude LLM-related parameters, only include custom layers
        if not name.startswith('llm.') and param.requires_grad:
            custom_params.append(param)
    
    if custom_params:
        param_groups.append({
            'params': custom_params,
            'lr': custom_layer_lr,
            'weight_decay': weight_decay,
            'name': 'custom_layers'
        })
        print(f"Set up custom layer parameter group, learning rate: {custom_layer_lr}, parameter count: {sum(p.numel() for p in custom_params)}")
    
    # 3. Check if there are trainable parameters
    if not param_groups:
        raise ValueError("Model has no trainable parameters, please check model configuration")
    
    # Create optimizer
    optimizer = AdamW(param_groups)
    return optimizer


def train_model(model, instruction, train_questions, model_cots, train_answers, 
               sparse_expected_relations, reasoning_llm_path, config):
    
    # Record total training time
    total_start_time = time.time()
    
    # Prepare data loader
    train_loader = prepare_data(
        instruction,
        train_questions,
        model_cots,
        train_answers,
        sparse_expected_relations,
        reasoning_llm_path,
        config["llm_name"],
        config["max_length"],
        batch_size=config['batch_size']
    )
    
    # Calculate total number of batches (for progress display)
    total_batches = len(train_loader)
    
    # Optimizer
    optimizer = get_layered_lr_optimizer(
        model, 
        lora_lr=config["lora_lr"], 
        custom_layer_lr=config["custom_lr"], 
        weight_decay=0.01
    )
    
    # Multi-GPU training setup
    model = nn.DataParallel(model)
    model = model.to(config['device'])
    
    # Record loss for each batch
    batch_losses = []
    epoch_losses = []
    
    for epoch in range(config['epochs']):
        print(f"\n[Epoch {epoch+1}/{config['epochs']}]")
        model.train()
        
        epoch_loss = 0.0
        
        # Add batch progress display
        for batch_idx, batch in enumerate(train_loader, 1):
            sample_ids = batch["sample_index"].to(config["device"])
            total_text_ids = batch["total_text_ids"].to(config["device"])
            total_text_attn_masks = batch["total_text_attn_masks"].to(config["device"])
            labels = batch["labels"].to(config["device"])
            expected_relation_matrix = batch["relation_matrix"].to(config["device"])
            
            # Forward pass
            outputs = model(sample_ids, total_text_ids, total_text_attn_masks, labels, expected_relation_matrix)
            
            # Calculate loss
            if torch.cuda.device_count() > 1 and config['num_gpus'] > 1:
                loss = outputs.mean()
            else:
                loss = outputs
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss for current batch
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            
            # Print training progress (every 10 batches or last batch)
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                accumulated_time = time.time() - total_start_time
                print(f"  Batch {batch_idx}/{total_batches} | Loss: {batch_loss:.4f} | Accumulated Time: {accumulated_time:.2f}s")
        
        # Calculate and record average loss for current epoch
        avg_epoch_loss = epoch_loss / total_batches
        epoch_losses.append(avg_epoch_loss)
        print(f"  Epoch {epoch+1}/{config['epochs']} | Average Loss: {avg_epoch_loss:.4f}")
    
    # Calculate total training time
    total_train_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_train_time:.2f}s")
    
    # Return model and recorded metrics
    return {
        'model': model,
        'batch_losses': batch_losses,
        'epoch_losses': epoch_losses,
        'total_train_time': total_train_time
    }
