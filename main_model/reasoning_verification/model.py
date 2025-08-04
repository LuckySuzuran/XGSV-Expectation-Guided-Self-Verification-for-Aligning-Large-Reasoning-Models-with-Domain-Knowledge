import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch_geometric.nn as gnn
from torch_geometric.utils import dense_to_sparse
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from peft import get_peft_model, LoraConfig, TaskType
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

class Model(nn.Module):
    def __init__(self, 
                 reasoning_llm_path: str, 
                 cot_embeddings: Tensor, 
                 cot_masks: Tensor, 
                 question_embeddings: Tensor, 
                 key_concept_feature_matrix: Tensor, 
                 prompt_embedding_size: int, 
                 attn_proj_hidden_size: int,
                 top_k: int, 
                 device: str, 
                 entity_similarity_loss_weight: float = 0.3, 
                 if_lora: bool = False):
        super().__init__()
        
        # Concept (Entity) feature dimensions
        self.concept_feature_dim = key_concept_feature_matrix.shape[1]
        self.n_concepts = key_concept_feature_matrix.shape[0]
        
        # Registered buffers for fixed embeddings
        self.register_buffer('cot_embeddings', cot_embeddings)
        self.register_buffer('cot_masks', cot_masks)
        self.register_buffer('key_concept',
                             torch.tensor(key_concept_feature_matrix, dtype=torch.bfloat16, device=device))
        self.register_buffer('question_last_token_embeddings', question_embeddings)
        
        # Device and configuration parameters
        self.device = device
        self.top_k = top_k
        self.entity_similarity_loss_weight = entity_similarity_loss_weight
        
        # Attention projection layers
        self.query_proj = nn.Sequential(
            nn.Linear(self.concept_feature_dim, attn_proj_hidden_size),
            nn.LayerNorm(attn_proj_hidden_size),
            nn.GELU(),
            nn.Linear(attn_proj_hidden_size, attn_proj_hidden_size, bias=False)
        ).to(self.device)
        
        self.key_proj = nn.Sequential(
            nn.Linear(prompt_embedding_size, attn_proj_hidden_size),
            nn.LayerNorm(attn_proj_hidden_size),
            nn.GELU(),
            nn.Linear(attn_proj_hidden_size, attn_proj_hidden_size, bias=False)
        ).to(self.device)
        
        self.value_proj = nn.Sequential(
            nn.Linear(prompt_embedding_size, attn_proj_hidden_size),
            nn.LayerNorm(attn_proj_hidden_size),
            nn.GELU(),
            nn.Linear(attn_proj_hidden_size, attn_proj_hidden_size, bias=False)
        ).to(self.device)
        
        # Fusion attention projections
        self.query_proj_fusion = nn.Sequential(
            nn.Linear(self.concept_feature_dim, attn_proj_hidden_size),
            nn.LayerNorm(attn_proj_hidden_size),
            nn.GELU(),
            nn.Linear(attn_proj_hidden_size, attn_proj_hidden_size, bias=False)
        ).to(self.device)
        
        self.key_proj_fusion = nn.Sequential(
            nn.Linear(prompt_embedding_size, attn_proj_hidden_size),
            nn.LayerNorm(attn_proj_hidden_size),
            nn.GELU(),
            nn.Linear(attn_proj_hidden_size, attn_proj_hidden_size, bias=False)
        ).to(self.device)
        
        self.value_proj_fusion = nn.Sequential(
            nn.Linear(prompt_embedding_size, attn_proj_hidden_size),
            nn.LayerNorm(attn_proj_hidden_size),
            nn.GELU(),
            nn.Linear(attn_proj_hidden_size, attn_proj_hidden_size, bias=False)
        ).to(self.device)
        
        # Relation matrix components
        self.main_diag = nn.Sequential(
            nn.Linear(prompt_embedding_size, prompt_embedding_size),
            nn.LayerNorm(prompt_embedding_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(prompt_embedding_size, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.anti_diag = nn.Sequential(
            nn.Linear(prompt_embedding_size * 2, prompt_embedding_size),
            nn.LayerNorm(prompt_embedding_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(prompt_embedding_size, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Confusion alpha for fusion attention
        self.confusion_alpha = nn.Sequential(
            nn.Linear(prompt_embedding_size, prompt_embedding_size),
            nn.LayerNorm(prompt_embedding_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(prompt_embedding_size, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Language model initialization
        self.tokenizer = AutoTokenizer.from_pretrained(reasoning_llm_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            reasoning_llm_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True)
        
        # Apply LoRA if enabled
        if if_lora:
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        else:
            # Freeze LLM parameters if not using LoRA
            for name, param in model.named_parameters():
                param.requires_grad = False
                
        self.llm = model
            
        # Graph neural network components
        self.gcn1 = gnn.GCNConv(
            in_channels=self.concept_feature_dim,
            out_channels=self.concept_feature_dim,
        ).to(self.device)
        
        self.gcn_recon_1 = gnn.GCNConv(
            in_channels=self.concept_feature_dim,
            out_channels=self.concept_feature_dim,
        ).to(self.device)
        
        # Projection to LLM space
        self.proj2llmspace = nn.Linear(
            in_features=self.concept_feature_dim,
            out_features=self.llm.config.hidden_size
        ).to(self.device)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        
        # Dropout layers
        self.dropout = nn.Dropout(0.1)
        self.attn_dropout = nn.Dropout(0.1)
        self.graph_dropout = nn.Dropout(0.1)
        
        # Text reconstruction
        self.recon_text_from_relation =  nn.Sequential(
            nn.Linear(self.concept_feature_dim, prompt_embedding_size),
            nn.GELU()
        ).to(self.device)

    def forward(self, 
                sample_index: Tensor, 
                total_text_ids: Tensor, 
                total_text_attn_masks: Tensor, 
                labels: Tensor, 
                expected_matrices: Tensor) -> Tensor:
        # Get embeddings for current batch
        cot_emb = self.cot_embeddings[sample_index]
        q_last_token_emb = self.question_last_token_embeddings[sample_index]
        batch_size, seq_len, _ = cot_emb.shape

        # Select top-k entities for each sample
        top_k = min(self.top_k, n_concepts)
        attn_masks = self.cot_masks[sample_index]
        last_token_embeddings = last_token_pool(cot_emb, attn_masks)
        
        top_indices, top_values = self._select_top_k_entities(
            last_token_embeddings, main_diag_values, top_k
        )

        # Compute attention for entity conditioned representations
        query = self.query_proj(self.key_concept)
        key = self.key_proj(cot_emb)
        value = self.value_proj(cot_emb)
        
        attn_scores = torch.matmul(
            query.unsqueeze(0),
            key.transpose(1, 2)
        )
        attn_weights = self.softmax(attn_scores / (self.key_proj[-1].out_features ** 0.5))
        weight_diff_loss = self.compute_similarity_loss(attn_weights)

        # Compute weighted embeddings for concepts
        weighted_emb = torch.matmul(attn_weights, value)
        weighted_emb = self.gelu(weighted_emb)

        # Build concept relationship graph
        batch_size, n_concepts, hidden_size = weighted_emb.shape
        main_diag_values = self.main_diag(weighted_emb).squeeze(-1)

        # Initialize and populate relation matrices
        relation_matrices = self._build_relation_matrices(
            weighted_emb, top_indices, top_values, n_concepts
        )

        # Fusion with expected matrices
        fusion_matrices = self._fuse_matrices(
            weighted_emb, cot_emb, q_last_token_emb, relation_matrices, expected_matrices, n_concepts
        )

        # Process through GCN
        graph_embedding = self._process_graphs(fusion_matrices, batch_size)

        # Combine with LLM
        input_embeds = self.llm.model.get_input_embeddings()(total_text_ids)
        combined_embeddings = torch.cat([
            graph_embedding.unsqueeze(1),
            input_embeds
        ], dim=1)

        # Prepare attention masks and labels
        combined_attention_mask = torch.cat([
            torch.ones(batch_size, 1, device=total_text_attn_masks.device),
            total_text_attn_masks
        ], dim=1)

        extended_labels = torch.full(
            (batch_size, 1), 
            fill_value=-100, 
            device=labels.device, 
            dtype=labels.dtype
        )
        combined_labels = torch.cat([extended_labels, labels], dim=1)

        # Get LLM outputs
        model_outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )
        main_loss = model_outputs.loss
        
        # Combine losses
        overall_loss = (main_loss + self.entity_similarity_loss_weight * weight_diff_loss)
        
        return overall_loss


    def get_graph_embedding(self, 
                           cot_embeddings: Tensor, 
                           attention_masks: Tensor, 
                           expected_matrices: Tensor, 
                           question_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """
        Generate graph embeddings for inference
        """
        with torch.no_grad():
            cot_emb = cot_embeddings.to(self.device)
            batch_size, seq_len, _ = cot_emb.shape
            # topk retrieval
            top_k = min(self.top_k, n_concepts)
            
            attn_masks = attention_masks.to(self.device)
            last_token_embeddings = last_token_pool(cot_emb, attn_masks)
            
            top_indices, top_values = self._select_top_k_entities(
                last_token_embeddings, main_diag_values, top_k
            )

            # Compute concept attention
            query = self.query_proj(self.key_concept)
            key = self.key_proj(cot_emb)
            value = self.value_proj(cot_emb)
            
            attn_scores = torch.matmul(
                query.unsqueeze(0),
                key.transpose(1, 2)
            )
            attn_weights = self.softmax(attn_scores / (self.key_proj[-1].out_features ** 0.5))
            weighted_emb = torch.matmul(attn_weights, value)
            weighted_emb = self.gelu(weighted_emb)

            # Build relation matrices
            batch_size, n_concepts, hidden_size = weighted_emb.shape
            main_diag_values = self.main_diag(weighted_emb).squeeze(-1)


            relation_matrices = self._build_relation_matrices(
                weighted_emb, top_indices, top_values, n_concepts
            )

            # Fusion with expected matrices
            fusion_matrices = self._fuse_matrices(
                weighted_emb, cot_emb, question_embeddings, 
                relation_matrices, expected_matrices, n_concepts
            )

            # Process through GCN
            gnn_output = []
            for b in range(batch_size):
                gnn_output.append(self._process_single_graph(fusion_matrices[b]))
            
            gnn_output = torch.stack(gnn_output)
            graph_embedding = self.proj2llmspace(gnn_output)
            graph_embedding = self.gelu(graph_embedding)
            
            return graph_embedding.detach().cpu(), torch.tensor(0.0)  # Placeholder alpha


    def answer_generation(self, 
                         cot_embeddings: Tensor, 
                         attention_masks: Tensor, 
                         test_total_ids: Tensor, 
                         expected_matrices: Tensor, 
                         question_embeddings: Tensor, 
                         generation_config: GenerationConfig) -> str:
        with torch.no_grad():
            cot_emb = cot_embeddings.to(self.device)
            batch_size, seq_len, _ = cot_emb.shape

            # Compute concept attention
            query = self.query_proj(self.key_concept)
            key = self.key_proj(cot_emb)
            value = self.value_proj(cot_emb)
            
            attn_scores = torch.matmul(
                query.unsqueeze(0),
                key.transpose(1, 2)
            )
            attn_weights = self.softmax(attn_scores / (self.key_proj[-1].out_features ** 0.5))
            weighted_emb = torch.matmul(attn_weights, value)
            weighted_emb = self.gelu(weighted_emb)

            # Build relation matrices
            batch_size, n_concepts, hidden_size = weighted_emb.shape
            main_diag_values = self.main_diag(weighted_emb).squeeze(-1)
            top_k = min(self.top_k, n_concepts)
            
            attn_masks = attention_masks.to(self.device)
            last_token_embeddings = last_token_pool(cot_emb, attn_masks)
            
            top_indices, top_values = self._select_top_k_entities(
                last_token_embeddings, main_diag_values, top_k
            )

            relation_matrices = self._build_relation_matrices(
                weighted_emb, top_indices, top_values, n_concepts
            )

            # Fusion with expected matrices
            fusion_matrices = self._fuse_matrices(
                weighted_emb, cot_emb, question_embeddings, 
                relation_matrices, expected_matrices, n_concepts
            )

            # Process through GCN
            gnn_output = []
            for b in range(batch_size):
                gnn_output.append(self._process_single_graph(fusion_matrices[b]))
            
            gnn_output = torch.stack(gnn_output)
            graph_embedding = self.proj2llmspace(gnn_output)
            graph_embedding = self.gelu(graph_embedding)
            
            # Combine with input embeddings
            input_embeds = self.llm.model.get_input_embeddings()(test_total_ids)
            combined_embeddings = torch.cat([
                graph_embedding.unsqueeze(1),
                input_embeds
            ], dim=1)

            # Generate output
            outputs = self.llm.generate(
                inputs_embeds=combined_embeddings,
                generation_config=generation_config
            )
            output_text = self.tokenizer.decode(outputs.detach().cpu()[0].tolist())
            
            return output_text

    def get_cot_graph(self, 
                     cot_embeddings: Tensor, 
                     attention_masks: Tensor, 
                     question_embeddings: Tensor, 
                     expected_matrices: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            cot_emb = cot_embeddings.to(self.device)
            batch_size, seq_len, _ = cot_emb.shape
            top_k = min(self.top_k, n_concepts)
            
            attn_masks = attention_masks.to(self.device)
            last_token_embeddings = last_token_pool(cot_emb, attn_masks)
            
            top_indices, top_values = self._select_top_k_entities(
                last_token_embeddings, main_diag_values, top_k
            )

            # Compute concept attention
            query = self.query_proj(self.key_concept)
            key = self.key_proj(cot_emb)
            value = self.value_proj(cot_emb)
            
            attn_scores = torch.matmul(
                query.unsqueeze(0),
                key.transpose(1, 2)
            )
            attn_weights = self.softmax(attn_scores / (self.key_proj[-1].out_features ** 0.5))
            weighted_emb = torch.matmul(attn_weights, value)
            weighted_emb = self.gelu(weighted_emb)

            # Build relation matrices
            batch_size, n_concepts, hidden_size = weighted_emb.shape
            main_diag_values = self.main_diag(weighted_emb).squeeze(-1)

            relation_matrices = self._build_relation_matrices(
                weighted_emb, top_indices, top_values, n_concepts
            )

            # Calculate fusion alpha
            query_fusion = self.query_proj_fusion(question_embeddings.to(self.device))
            key_fusion = self.key_proj_fusion(cot_emb)
            value_fusion = self.value_proj_fusion(cot_emb)
            
            attn_scores_fusion = torch.matmul(
                query_fusion.unsqueeze(0),
                key_fusion.transpose(1, 2)
            )
            attn_weights_fusion = self.softmax(attn_scores_fusion / (self.key_proj_fusion[-1].out_features ** 0.5))
            weighted_emb_fusion = torch.matmul(attn_weights_fusion, value_fusion)
            weighted_emb_fusion = self.gelu(weighted_emb_fusion)      
            alpha = self.confusion_alpha(weighted_emb_fusion)
            
            return relation_matrices, alpha


    def compute_similarity_loss(self, attn_weights: Tensor, high: float = 0.7) -> Tensor:
        """
        Compute loss to encourage diverse attention weights
        """
        batch_size, n_concepts, seq_len = attn_weights.shape
        attn_weights = F.normalize(attn_weights, p=2, dim=2)

        # Calculate similarity matrix
        dot_product = torch.matmul(attn_weights, attn_weights.transpose(1, 2))
        norm = torch.norm(attn_weights, dim=2, keepdim=True)
        norm_product = torch.matmul(norm, norm.transpose(1, 2))
        concept_sim = dot_product / (norm_product + 1e-8)

        # Mask diagonal elements
        mask = 1 - torch.eye(n_concepts, device=attn_weights.device).unsqueeze(0)
        non_diag_sim = concept_sim * mask

        # Penalty for high similarity
        loss_high = F.relu(non_diag_sim - high).mean()
        return loss_high

    def _select_top_k_entities(self, 
                              last_token_embeddings: Tensor, 
                              main_diag_values: Tensor, 
                              top_k: int) -> tuple[Tensor, Tensor]:
        """Select top-k entities based on similarity to last token"""
        top_indices = []
        top_values = []
        for b in range(last_token_embeddings.shape[0]):
            last_token_embedding = last_token_embeddings[b]
            indices = find_similar_keywords(last_token_embedding, self.key_concept, top_k=top_k)
            values = main_diag_values[b][indices]
            top_indices.append(indices)
            top_values.append(values)
        return torch.stack(top_indices, dim=0), torch.stack(top_values, dim=0)


    def _build_relation_matrices(self, 
                                weighted_emb: Tensor, 
                                top_indices: Tensor, 
                                top_values: Tensor, 
                                n_concepts: int) -> Tensor:
        batch_size = weighted_emb.shape[0]
        top_k = top_indices.shape[1]
        
        # Initialize relation matrices
        relation_matrices = torch.zeros(
            batch_size, n_concepts, n_concepts,
            device=weighted_emb.device, dtype=torch.bfloat16
        )

        # Fill main diagonal values
        batch_idx = torch.arange(batch_size, device=weighted_emb.device).view(-1, 1).expand(-1, top_k)
        row_idx = top_indices
        col_idx = top_indices
        
        indices = torch.stack([
            batch_idx.reshape(-1),
            row_idx.reshape(-1),
            col_idx.reshape(-1)
        ], dim=1)
        
        values = top_values.reshape(-1)
        relation_matrices = relation_matrices.reshape(-1)
        flat_indices = indices[:, 0] * (n_concepts * n_concepts) + indices[:, 1] * n_concepts + indices[:, 2]
        relation_matrices.scatter_(0, flat_indices, values)
        relation_matrices = relation_matrices.reshape(batch_size, n_concepts, n_concepts)

        # Fill pairwise relations (upper and lower triangles)
        i_indices = torch.arange(top_k, device=weighted_emb.device).view(-1, 1)
        j_indices = torch.arange(top_k, device=weighted_emb.device).view(1, -1)
        mask = j_indices > i_indices
        valid_pairs = torch.stack(torch.where(mask), dim=1)
        num_pairs = valid_pairs.shape[0]

        if num_pairs > 0:
            batch_idx = torch.arange(batch_size, device=weighted_emb.device).view(-1, 1, 1).expand(-1, num_pairs, 1)
            pair_idx = valid_pairs.unsqueeze(0).expand(batch_size, -1, -1)
            
            batch_idx = batch_idx.reshape(-1)
            i_pos = pair_idx[:, :, 0].reshape(-1)
            j_pos = pair_idx[:, :, 1].reshape(-1)
            
            idx_i = top_indices[batch_idx, i_pos]
            idx_j = top_indices[batch_idx, j_pos]
            
            emb_i = weighted_emb[batch_idx, idx_i]
            emb_j = weighted_emb[batch_idx, idx_j]
            pair_embeddings = torch.cat([emb_i, emb_j], dim=1)
            
            rel_values = self.anti_diag(pair_embeddings).squeeze(-1)
            
            upper_indices = torch.stack([batch_idx, idx_i, idx_j], dim=1)
            lower_indices = torch.stack([batch_idx, idx_j, idx_i], dim=1)
            
            relation_matrices = relation_matrices.reshape(-1)
            flat_upper_indices = upper_indices[:, 0] * (n_concepts * n_concepts) + upper_indices[:, 1] * n_concepts + upper_indices[:, 2]
            flat_lower_indices = lower_indices[:, 0] * (n_concepts * n_concepts) + lower_indices[:, 1] * n_concepts + lower_indices[:, 2]
            
            relation_matrices.scatter_(0, flat_upper_indices, rel_values)
            relation_matrices.scatter_(0, flat_lower_indices, rel_values)
            relation_matrices = relation_matrices.reshape(batch_size, n_concepts, n_concepts)

        return relation_matrices

    def _fuse_matrices(self, 
                      weighted_emb: Tensor, 
                      cot_emb: Tensor, 
                      question_emb: Tensor, 
                      relation_matrices: Tensor, 
                      expected_matrices: Tensor, 
                      n_concepts: int) -> Tensor:
        query_fusion = self.query_proj_fusion(question_emb)
        key_fusion = self.key_proj_fusion(cot_emb)
        value_fusion = self.value_proj_fusion(cot_emb)
        
        attn_scores_fusion = torch.matmul(
            query_fusion.unsqueeze(0),
            key_fusion.transpose(1, 2)
        )
        attn_weights_fusion = self.softmax(attn_scores_fusion / (self.key_proj_fusion[-1].out_features ** 0.5))
        weighted_emb_fusion = torch.matmul(attn_weights_fusion, value_fusion)
        weighted_emb_fusion = self.gelu(weighted_emb_fusion)      
        
        alpha = self.confusion_alpha(weighted_emb_fusion)
        alpha_matrix = alpha.expand(-1, n_concepts, n_concepts)
        
        return alpha_matrix * relation_matrices + (1 - alpha_matrix) * expected_matrices


    def _process_graphs(self, fusion_matrices: Tensor, batch_size: int) -> Tensor:
        gnn_output = []
        for b in range(batch_size):
            gnn_output.append(self._process_single_graph(fusion_matrices[b]))
        graph_embedding = torch.stack(gnn_output)
        graph_embedding = self.proj2llmspace(graph_embedding)
        graph_embedding = self.gelu(graph_embedding)
        graph_embedding = self.graph_dropout(graph_embedding)
        return graph_embedding


    def _process_single_graph(self, fusion_matrix: Tensor) -> Tensor:
        # Get subgraph indices from non-zero diagonal elements
        diag_elements = torch.diag(fusion_matrix)
        nonzero_indices = torch.nonzero(diag_elements, as_tuple=True)[0]
        subgraph_indices = nonzero_indices
        
        if subgraph_indices.numel() == 0:
            return torch.zeros(self.concept_feature_dim, device=self.device)
            
        # Extract subgraph
        subgraph_relations = fusion_matrix[subgraph_indices][:, subgraph_indices]
        subgraph_concepts = self.key_concept[subgraph_indices]
        
        # Convert to sparse format
        edge_index, edge_weight = dense_to_sparse(subgraph_relations)
        
        # GCN processing
        out = self.gcn1(subgraph_concepts, edge_index, edge_weight)
        out = self.gelu(out)
        out = self.dropout(out)
        
        return torch.mean(out, dim=0)
    