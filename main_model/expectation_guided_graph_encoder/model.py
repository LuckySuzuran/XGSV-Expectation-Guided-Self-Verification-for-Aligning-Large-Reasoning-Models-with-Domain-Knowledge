import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch_geometric.nn as gnn
from torch_geometric.utils import dense_to_sparse
from torch.nn.functional import cosine_similarity, normalize
import torch.nn.functional as F
from utils import *

class RelationActivation(nn.Module):
    """Predictor for graph relations"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class ProjectionHead(nn.Module):
    """Projection head for attention mechanism"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        return self.layers(x)

class GraphEncoder(nn.Module):    
    def __init__(self, llm_path, key_concept_feature_matrix, prompt_embedding_size, attn_proj_hidden_size,
                 question_embeddings, question_masks, answer_embeddings, device='cuda',
                 top_k=50, attn_sim_loss_weight=0.1):
        super().__init__()
        self.device = device
        self.top_k = top_k
        
        # Register buffers for embeddings and masks
        self.register_buffer('question_embeddings', question_embeddings)
        self.register_buffer('question_masks', question_masks)
        self.register_buffer('answer_embeddings', answer_embeddings)
        self.register_buffer('key_concept', 
                          torch.tensor(key_concept_feature_matrix, dtype=torch.bfloat16, device=self.device))

        # Concept feature dimensions
        self.concept_feature_dim = key_concept_feature_matrix.shape[1]
        self.n_concepts = key_concept_feature_matrix.shape[0]
        
        # Attention projection heads
        self.query_proj = ProjectionHead(
            self.concept_feature_dim, 
            attn_proj_hidden_size, 
            attn_proj_hidden_size
        ).to(self.device)
        
        self.key_proj = ProjectionHead(
            prompt_embedding_size, 
            attn_proj_hidden_size, 
            attn_proj_hidden_size
        ).to(self.device)
        
        self.value_proj = ProjectionHead(
            prompt_embedding_size, 
            attn_proj_hidden_size, 
            attn_proj_hidden_size
        ).to(self.device)
        
        # Graph activation components
        self.main_diag = RelationActivation(
            prompt_embedding_size, 
            prompt_embedding_size
        ).to(self.device)

        self.anti_diag = RelationActivation(
            prompt_embedding_size * 2, 
            prompt_embedding_size
        ).to(self.device)

        # Language model setup
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        
        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # GNN setup
        self.gcn1 = gnn.GCNConv(
            in_channels=self.concept_feature_dim,
            out_channels=self.concept_feature_dim
        ).to(self.device)

        # Projector to LLM space
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
        
        # Loss weights
        self.attn_sim_loss_weight = attn_sim_loss_weight

    def forward(self, question_index, total_text_ids, total_text_attn_masks, labels):        
        last_hidden_states = self.question_embeddings[question_index] #Get text deep representations
        batch_size, seq_len, _ = last_hidden_states.shape

        #Step 1: Top-k Retrieval
        top_k = min(self.top_k, n_concepts)  # Ensure k doesn't exceed total entities
        attn_masks = self.question_masks[question_index]

        last_token_embeddings = last_token_pool(last_hidden_states, attn_masks)
        top_indices, top_values = self._get_top_concepts(
            last_token_embeddings, main_diag_values, top_k
        )

        #Step 2: Attnention mechanism for conditioned entity representation
        query = self.query_proj(self.key_concept)
        key = self.key_proj(last_hidden_states)
        value = self.value_proj(last_hidden_states)
        
        attn_scores = torch.matmul(
            query.unsqueeze(0),  # [1, n_concepts, attn_proj_hidden_size]
            key.transpose(1, 2)  # [batch_size, attn_proj_hidden_size, seq_len]
        )  
        attn_weights = self.softmax(attn_scores / (self.key_proj.layers[-1].out_features ** 0.5))

        weighted_emb = torch.matmul(
            attn_weights,  # [batch_size, n_concepts, seq_len]
            value  # [batch_size, seq_len, llm_hidden_size]
        )
        weighted_emb = self.gelu(weighted_emb) # [batch_size, n_concepts, llm_hidden_size]

        #Step 3: Graph Activation
        batch_size, n_concepts, hidden_size = weighted_emb.shape
        main_diag_values = self.main_diag(weighted_emb).squeeze(-1)  # [batch, n_concepts]
        
        batch_idx = torch.arange(batch_size, device=attn_weights.device).unsqueeze(1)
        top_attn_weights = attn_weights[batch_idx, top_indices]

        weight_diff_loss = self.compute_similarity_loss(top_attn_weights)

        relation_matrices = self._build_relation_matrices(
            weighted_emb, top_indices, top_values, batch_size, n_concepts
        )

        #Last Step: Graph Prompt Tuning Task on Downstream LLMs
        graph_embedding = self._process_graphs(
            relation_matrices, top_indices, batch_size
        )

        main_loss = self._tune_llm(
            graph_embedding, total_text_ids, total_text_attn_masks, labels, batch_size
        )

        return main_loss + self.attn_sim_loss_weight * weight_diff_loss


    def _get_top_concepts(self, last_token_embeddings, main_diag_values, top_k):
        """Get top k concepts based on cosine similarity"""
        batch_size = last_token_embeddings.shape[0]
        top_indices = []
        top_values = []
        
        for b in range(batch_size):
            last_token_embedding = last_token_embeddings[b]
            indices = find_similar_keywords(last_token_embedding, self.key_concept, top_k=top_k)
            values = main_diag_values[b][indices]
            top_indices.append(indices)
            top_values.append(values)
            
        return torch.stack(top_indices, dim=0), torch.stack(top_values, dim=0)


    def _build_relation_matrices(self, weighted_emb, top_indices, top_values, batch_size, n_concepts):
        """Build relation matrices for concepts"""
        # Initialize relation matrices with zeros
        relation_matrices = torch.zeros(
            batch_size, n_concepts, n_concepts,
            device=weighted_emb.device, dtype=torch.bfloat16
        )

        # Set main diagonal values
        relation_matrices = self._set_diagonal_values(
            relation_matrices, top_indices, top_values, batch_size, n_concepts
        )

        # Set off-diagonal values for top-k entity pairs
        relation_matrices = self._set_off_diagonal_values(
            relation_matrices, weighted_emb, top_indices, batch_size, n_concepts
        )

        return relation_matrices


    def _set_diagonal_values(self, relation_matrices, top_indices, top_values, batch_size, n_concepts):
        """Set diagonal values in relation matrices"""
        top_k = top_indices.shape[1]
        
        # Create indices for batch, rows, and columns
        batch_idx = torch.arange(batch_size, device=relation_matrices.device).view(-1, 1).expand(-1, top_k)
        row_idx = top_indices
        col_idx = top_indices

        # Combine into 2D index tensor [batch_size*top_k, 3]
        indices = torch.stack([
            batch_idx.reshape(-1),
            row_idx.reshape(-1),
            col_idx.reshape(-1)
        ], dim=1)

        # Prepare values to fill
        values = top_values.reshape(-1)

        # Fill diagonal values using scatter
        flat_relation = relation_matrices.reshape(-1)
        flat_indices = indices[:, 0] * (n_concepts * n_concepts) + indices[:, 1] * n_concepts + indices[:, 2]
        flat_relation.scatter_(0, flat_indices, values)
        
        return flat_relation.reshape(batch_size, n_concepts, n_concepts)


    def _set_off_diagonal_values(self, relation_matrices, weighted_emb, top_indices, batch_size, n_concepts):
        """Set off-diagonal values in relation matrices"""
        top_k = top_indices.shape[1]
        hidden_size = weighted_emb.shape[2]

        # Generate indices for upper triangle pairs (i < j)
        i_indices = torch.arange(top_k, device=weighted_emb.device).view(-1, 1)
        j_indices = torch.arange(top_k, device=weighted_emb.device).view(1, -1)
        mask = j_indices > i_indices  # Upper triangle mask
        valid_pairs = torch.stack(torch.where(mask), dim=1)  # [num_pairs, 2]
        num_pairs = valid_pairs.shape[0]

        if num_pairs == 0:
            return relation_matrices

        # Expand to each batch
        batch_idx = torch.arange(batch_size, device=weighted_emb.device).view(-1, 1, 1).expand(-1, num_pairs, 1)
        pair_idx = valid_pairs.unsqueeze(0).expand(batch_size, -1, -1)

        # Flatten indices for batch processing
        batch_idx = batch_idx.reshape(-1)
        i_pos = pair_idx[:, :, 0].reshape(-1)
        j_pos = pair_idx[:, :, 1].reshape(-1)

        # Get actual concept indices
        idx_i = top_indices[batch_idx, i_pos]
        idx_j = top_indices[batch_idx, j_pos]

        # Extract and concatenate embeddings for pairs
        emb_i = weighted_emb[batch_idx, idx_i]
        emb_j = weighted_emb[batch_idx, idx_j]
        pair_embeddings = torch.cat([emb_i, emb_j], dim=1)

        # Calculate relationship values
        rel_values = self.anti_diag(pair_embeddings).squeeze(-1)

        # Fill upper and lower triangles
        flat_relation = relation_matrices.reshape(-1)
        
        # Upper triangle indices
        upper_indices = torch.stack([batch_idx, idx_i, idx_j], dim=1)
        flat_upper = upper_indices[:, 0] * (n_concepts * n_concepts) + upper_indices[:, 1] * n_concepts + upper_indices[:, 2]
        
        # Lower triangle indices (symmetric)
        lower_indices = torch.stack([batch_idx, idx_j, idx_i], dim=1)
        flat_lower = lower_indices[:, 0] * (n_concepts * n_concepts) + lower_indices[:, 1] * n_concepts + lower_indices[:, 2]

        # Fill values
        flat_relation.scatter_(0, flat_upper, rel_values)
        flat_relation.scatter_(0, flat_lower, rel_values)
        
        return flat_relation.reshape(batch_size, n_concepts, n_concepts)


    def _process_graphs(self, relation_matrices, top_indices, batch_size):
        """Process graphs using GNN"""
        gnn_output = []
        
        for b in range(batch_size):
            # Extract subgraph for top-k concepts
            subgraph_indices = top_indices[b]
            subgraph_relations = relation_matrices[b, subgraph_indices][:, subgraph_indices]
            subgraph_concepts = self.key_concept[subgraph_indices]

            # Convert to sparse format
            edge_index, edge_weight = dense_to_sparse(subgraph_relations)

            # GCN processing
            out = self.gcn1(subgraph_concepts, edge_index, edge_weight)
            out = self.gelu(out)
            out = self.dropout(out)
            
            # Average pooling
            out = torch.mean(out, dim=0)
            gnn_output.append(out)
            
        # Project to LLM space
        graph_embedding = torch.stack(gnn_output)
        graph_embedding = self.proj2llmspace(graph_embedding)
        graph_embedding = self.gelu(graph_embedding)
        graph_embedding = self.graph_dropout(graph_embedding)
        
        return graph_embedding


    def _tune_llm(self, graph_embedding, total_text_ids, total_text_attn_masks, labels, batch_size):
        """Tune LLM with graph embeddings as prompts"""
        # Get input embeddings and combine with graph embedding
        input_embeds = self.llm.get_input_embeddings()(total_text_ids)
        seq_len = labels.shape[1]

        # Combine embeddings
        combined_embeddings = torch.cat([
            graph_embedding.unsqueeze(1),  # [batch_size, 1, hidden_size]
            input_embeds
        ], dim=1)

        # Combine attention masks
        combined_attention_mask = torch.cat([
            torch.ones(batch_size, 1, device=total_text_attn_masks.device),
            total_text_attn_masks
        ], dim=1)

        # Extend labels to match combined input length
        extended_labels = torch.full(
            (batch_size, 1), 
            fill_value=-100, 
            device=labels.device, 
            dtype=labels.dtype
        )
        combined_labels = torch.cat([extended_labels, labels], dim=1)

        # Call LLM with combined inputs
        model_outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )
        
        return model_outputs.loss


    def compute_similarity_loss(self, attn_weights, high=0.7):
        """Compute similarity loss for attention weights"""
        batch_size, n_concepts, seq_len = attn_weights.shape
        attn_weights = normalize(attn_weights, p=2, dim=2)

        # Calculate concept similarities
        dot_product = torch.matmul(attn_weights, attn_weights.transpose(1, 2))
        norm = torch.norm(attn_weights, dim=2, keepdim=True)
        norm_product = torch.matmul(norm, norm.transpose(1, 2))
        concept_sim = dot_product / (norm_product + 1e-8)

        # Mask out diagonal elements
        mask = 1 - torch.eye(n_concepts, device=attn_weights.device).unsqueeze(0)
        non_diag_sim = concept_sim * mask

        # Calculate loss for similarities that are too high
        loss_high = F.relu(non_diag_sim - high).mean()
        return loss_high


    def get_relation_matrices(self, question_embeddings, question_masks):
        """Get relation matrices using model parameters"""
        with torch.no_grad():
            last_hidden_states = question_embeddings
            batch_size, seq_len, _ = last_hidden_states.shape
            
            # Calculate attention projections
            query = self.query_proj(self.key_concept)
            key = self.key_proj(last_hidden_states)
            value = self.value_proj(last_hidden_states)
            
            # Calculate attention weights
            attn_scores = torch.matmul(
                query.unsqueeze(0),
                key.transpose(1, 2)
            )  
            attn_weights = self.softmax(attn_scores / (self.key_proj.layers[-1].out_features ** 0.5))

            # Calculate weighted embeddings
            weighted_emb = torch.matmul(attn_weights, value)
            weighted_emb = self.gelu(weighted_emb)

            # Build concept relationship graph
            batch_size, n_concepts, hidden_size = weighted_emb.shape
            main_diag_values = self.main_diag(weighted_emb).squeeze(-1)

            # Get top concepts
            top_k = min(self.top_k, n_concepts)
            attn_masks = question_masks
            last_token_embeddings = last_token_pool(last_hidden_states, attn_masks)
            top_indices, top_values = self._get_top_concepts(
                last_token_embeddings, main_diag_values, top_k
            )

            # Build relation matrices
            relation_matrices = self._build_relation_matrices(
                weighted_emb, top_indices, top_values, batch_size, n_concepts
            )

        return relation_matrices
