import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class KGATLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, n_relations, aggregator='bi-interaction'):
        super(KGATLayer, self).__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations
        self.aggregator = aggregator
        
        # Transformation matrices
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.W_r = nn.Parameter(torch.Tensor(n_relations, in_dim, out_dim))
        
        # Attention parameters
        if aggregator == 'bi-interaction':
            self.a = nn.Parameter(torch.Tensor(out_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        
        self.init_parameters()
        
    def init_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_r)
        if self.aggregator == 'bi-interaction':
            nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, edge_index, edge_type=None):
        return self.propagate(edge_index, x=x, edge_type=edge_type)
    
    def message(self, x_i, x_j, edge_type, index, ptr, size_i):
        if edge_type is not None:
            # Knowledge graph edges with relations
            w_r = self.W_r[edge_type]
            x_j_transformed = torch.bmm(x_j.unsqueeze(1), w_r).squeeze(1)
        else:
            # User-item edges
            x_j_transformed = torch.matmul(x_j, self.W)
        
        # Calculate attention scores
        if self.aggregator == 'bi-interaction':
            x_i_transformed = torch.matmul(x_i, self.W)
            attention_logits = (x_i_transformed * x_j_transformed).sum(dim=1, keepdim=True)
            attention_logits = self.leaky_relu(attention_logits)
            
            # Normalize attention scores
            attention_scores = F.softmax(attention_logits, dim=0)
            
            return attention_scores * x_j_transformed
        else:
            return x_j_transformed
    
    def update(self, aggr_out, x):
        return aggr_out


class KGAT(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, embed_dim, layer_dims, 
                 aggregator='bi-interaction', reg_weight=1e-5):
        super(KGAT, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        self.aggregator = aggregator
        self.reg_weight = reg_weight
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.entity_embedding = nn.Embedding(n_entities, embed_dim)
        self.relation_embedding = nn.Embedding(n_relations, embed_dim)
        
        # KGAT layers
        self.layers = nn.ModuleList()
        in_dim = embed_dim
        for out_dim in layer_dims:
            self.layers.append(KGATLayer(in_dim, out_dim, n_relations, aggregator))
            in_dim = out_dim
        
        self.init_embeddings()
        
    def init_embeddings(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
    def forward(self, user_indices, item_indices, edge_index_ui, edge_index_kg, edge_type_kg):
        # Get embeddings
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.entity_embedding(item_indices)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_embeds, item_embeds], dim=0)
        
        # Apply KGAT layers
        for layer in self.layers:
            # User-item graph propagation
            x_ui = layer(x, edge_index_ui)
            
            # Knowledge graph propagation (only for items/entities)
            if edge_index_kg is not None:
                x_kg = layer(item_embeds, edge_index_kg, edge_type_kg)
                # Update item embeddings with KG information
                x[self.n_users:] += x_kg
            
            x = F.relu(x_ui)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Split back to user and item embeddings
        user_embeds_final = x[:self.n_users]
        item_embeds_final = x[self.n_users:]
        
        return user_embeds_final, item_embeds_final
    
    def predict(self, users, items):
        """Predict scores for user-item pairs"""
        scores = (users * items).sum(dim=1)
        return scores
    
    def calculate_loss(self, users, pos_items, neg_items):
        """BPR loss calculation"""
        pos_scores = self.predict(users, pos_items)
        neg_scores = self.predict(users, neg_items)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            users.norm(2).pow(2) + 
            pos_items.norm(2).pow(2) + 
            neg_items.norm(2).pow(2)
        ) / users.size(0)
        
        return bpr_loss + reg_loss
    
    def get_user_embedding(self, user_id):
        """Get embedding for a specific user"""
        return self.user_embedding(torch.tensor(user_id))
    
    def get_item_embedding(self, item_id):
        """Get embedding for a specific item"""
        return self.entity_embedding(torch.tensor(item_id))
    
    def get_relation_embedding(self, relation_id):
        """Get embedding for a specific relation"""
        return self.relation_embedding(torch.tensor(relation_id))