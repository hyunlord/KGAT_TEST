import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sp


class KGATConvImproved(MessagePassing):
    """Improved KGAT Convolution layer with multiple aggregation types"""
    
    def __init__(self, in_dim, out_dim, n_relations, aggregator='bi-interaction', 
                 dropout=0.1, use_attention=True):
        # Set aggregation based on aggregator type
        if aggregator == 'gcn':
            super(KGATConvImproved, self).__init__(aggr='add')
        elif aggregator == 'graphsage':
            super(KGATConvImproved, self).__init__(aggr='mean')
        else:  # bi-interaction
            super(KGATConvImproved, self).__init__(aggr='add')
            
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations
        self.aggregator = aggregator
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Transformation matrices
        if aggregator == 'gcn':
            self.W = nn.Linear(in_dim, out_dim, bias=False)
            self.W_r = nn.ModuleList([
                nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
            ])
        elif aggregator == 'graphsage':
            self.W = nn.Linear(in_dim * 2, out_dim, bias=True)
            self.W_r = nn.ModuleList([
                nn.Linear(in_dim * 2, out_dim, bias=True) for _ in range(n_relations)
            ])
        else:  # bi-interaction
            self.W = nn.Linear(in_dim, out_dim, bias=False)
            self.W_r = nn.ModuleList([
                nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
            ])
        
        # Attention parameters (for bi-interaction)
        if aggregator == 'bi-interaction' and use_attention:
            self.a = nn.Parameter(torch.Tensor(1, out_dim))
            nn.init.xavier_uniform_(self.a)
            
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize all linear layers
        if hasattr(self.W, 'reset_parameters'):
            self.W.reset_parameters()
        for w_r in self.W_r:
            if hasattr(w_r, 'reset_parameters'):
                w_r.reset_parameters()
    
    def forward(self, x, edge_index, edge_type=None, edge_norm=None):
        # Add self-loops for GCN
        if self.aggregator == 'gcn':
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            
        return self.propagate(edge_index, x=x, edge_type=edge_type, 
                            edge_norm=edge_norm, size=(x.size(0), x.size(0)))
    
    def message(self, x_i, x_j, edge_type, edge_norm, index):
        # Transform neighbor embeddings based on aggregator type
        if self.aggregator == 'gcn':
            # GCN-style transformation
            if edge_type is not None:
                # For knowledge graph edges
                x_j_list = []
                for idx, rel in enumerate(edge_type):
                    x_j_list.append(self.W_r[rel](x_j[idx].unsqueeze(0)))
                x_j = torch.cat(x_j_list, dim=0)
            else:
                # For user-item edges
                x_j = self.W(x_j)
                
            # Apply edge normalization if provided
            if edge_norm is not None:
                x_j = edge_norm.view(-1, 1) * x_j
                
        elif self.aggregator == 'graphsage':
            # GraphSAGE-style: concatenate self and neighbor embeddings
            x_concat = torch.cat([x_i, x_j], dim=1)
            if edge_type is not None:
                x_j_list = []
                for idx, rel in enumerate(edge_type):
                    x_j_list.append(self.W_r[rel](x_concat[idx].unsqueeze(0)))
                x_j = torch.cat(x_j_list, dim=0)
            else:
                x_j = self.W(x_concat)
                
        else:  # bi-interaction
            # Transform embeddings
            if edge_type is not None:
                x_j_list = []
                for idx, rel in enumerate(edge_type):
                    x_j_list.append(self.W_r[rel](x_j[idx].unsqueeze(0)))
                x_j = torch.cat(x_j_list, dim=0)
            else:
                x_j = self.W(x_j)
            
            # Calculate attention if enabled
            if self.use_attention:
                x_i = self.W(x_i)
                
                # Element-wise multiplication for attention
                attention_input = x_i * x_j
                attention_scores = torch.sum(attention_input * self.a, dim=1, keepdim=True)
                attention_scores = self.leaky_relu(attention_scores)
                
                # Apply attention weights
                x_j = attention_scores * x_j
        
        return x_j
    
    def update(self, aggr_out):
        return self.dropout_layer(aggr_out)


class KGATImproved(pl.LightningModule):
    """Improved KGAT model with adjacency matrix support and multiple aggregators"""
    
    def __init__(self, config):
        super(KGATImproved, self).__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.n_users = config.n_users
        self.n_entities = config.n_entities
        self.n_relations = config.n_relations
        self.embed_dim = config.embed_dim
        self.layer_dims = config.layer_dims
        self.dropout = config.dropout
        self.reg_weight = config.reg_weight
        self.lr = config.lr
        self.aggregator = config.aggregator
        self.use_pretrain = getattr(config, 'use_pretrain', False)
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embed_dim)
        # relation_embedding은 KGATConvImproved 내부에서 처리되므로 여기서는 제거
        
        # KGAT layers
        self.convs = nn.ModuleList()
        in_dim = self.embed_dim
        for out_dim in self.layer_dims:
            self.convs.append(
                KGATConvImproved(in_dim, out_dim, self.n_relations, 
                               self.aggregator, self.dropout)
            )
            in_dim = out_dim
            
        # Final dimension: initial embedding + all layer outputs
        self.final_dim = self.embed_dim + sum(self.layer_dims)
        
        # Optional: transformation for final embeddings
        self.transform = nn.Linear(self.final_dim, self.embed_dim, bias=False)
        
        # Edge normalization cache
        self.edge_norm_ui = None
        self.edge_norm_kg = None
        
        # Initialize embeddings
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.transform.weight)
        
    def compute_edge_norm(self, edge_index, num_nodes):
        """Compute edge normalization for GCN-style aggregation"""
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Compute normalization: D^{-1/2} A D^{-1/2}
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return norm
        
    def forward(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        # Get all embeddings
        user_embeds = self.user_embedding.weight
        entity_embeds = self.entity_embedding.weight
        
        # Initial embedding: concatenate users and entities
        x = torch.cat([user_embeds, entity_embeds], dim=0)
        
        # Compute edge normalizations if using GCN
        if self.aggregator == 'gcn':
            if self.edge_norm_ui is None:
                self.edge_norm_ui = self.compute_edge_norm(
                    edge_index_ui, self.n_users + self.n_entities
                )
            edge_norm_ui = self.edge_norm_ui
            
            if edge_index_kg is not None and self.edge_norm_kg is None:
                self.edge_norm_kg = self.compute_edge_norm(
                    edge_index_kg, self.n_entities
                )
            edge_norm_kg = self.edge_norm_kg if edge_index_kg is not None else None
        else:
            edge_norm_ui = None
            edge_norm_kg = None
        
        # Store outputs from each layer (including initial embedding)
        all_embeddings = [x]
        
        # Apply KGAT convolutions
        for conv in self.convs:
            # User-item graph convolution
            x_ui = conv(x, edge_index_ui, edge_norm=edge_norm_ui)
            
            # Knowledge graph convolution (entities only)
            if edge_index_kg is not None and edge_type_kg is not None:
                # Extract entity embeddings
                entity_x = x[self.n_users:]
                x_kg = conv(entity_x, edge_index_kg, edge_type_kg, edge_norm_kg)
                
                # Update x with KG information
                x = torch.cat([x_ui[:self.n_users], x_ui[self.n_users:] + x_kg], dim=0)
            else:
                x = x_ui
                
            # Apply activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # L2 normalization (crucial for KGAT)
            x = F.normalize(x, p=2, dim=1)
            
            # Store current layer output
            all_embeddings.append(x)
        
        # Concatenate all layer embeddings (KGAT's key feature)
        final_embeds = torch.cat(all_embeddings, dim=1)
        
        # Optional: transform back to original dimension
        final_embeds = self.transform(final_embeds)
        
        # Split back to users and entities
        user_final = final_embeds[:self.n_users]
        entity_final = final_embeds[self.n_users:]
        
        return user_final, entity_final
    
    def bpr_loss(self, users, pos_items, neg_items):
        """Bayesian Personalized Ranking loss with L2 regularization"""
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        
        # BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        
        # L2 regularization on embeddings
        reg_loss = self.reg_weight * (
            users.norm(2).pow(2) + 
            pos_items.norm(2).pow(2) + 
            neg_items.norm(2).pow(2)
        ) / users.size(0)
        
        return bpr_loss + reg_loss
    
    def training_step(self, batch, batch_idx):
        # Move all tensors to current device
        edge_index_ui = batch['edge_index_ui'].to(self.device)
        edge_index_kg = batch.get('edge_index_kg', None)
        edge_type_kg = batch.get('edge_type_kg', None)
        
        if edge_index_kg is not None:
            edge_index_kg = edge_index_kg.to(self.device)
        if edge_type_kg is not None:
            edge_type_kg = edge_type_kg.to(self.device)
        
        # Sample positive and negative items
        user_ids = batch['user_ids']
        pos_item_ids = batch['pos_item_ids']
        neg_item_ids = batch['neg_item_ids']
        
        # Forward pass
        user_embeds, entity_embeds = self(edge_index_ui, edge_index_kg, edge_type_kg)
        
        # Get specific embeddings
        users = user_embeds[user_ids]
        pos_items = entity_embeds[pos_item_ids]
        neg_items = entity_embeds[neg_item_ids]
        
        # Calculate loss
        loss = self.bpr_loss(users, pos_items, neg_items)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, 'test')
    
    def _evaluation_step(self, batch, batch_idx, stage):
        edge_index_ui = batch['edge_index_ui']
        edge_index_kg = batch.get('edge_index_kg', None)
        edge_type_kg = batch.get('edge_type_kg', None)
        
        # Forward pass
        user_embeds, entity_embeds = self(edge_index_ui, edge_index_kg, edge_type_kg)
        
        # Evaluate batch users
        metrics = defaultdict(list)
        
        for idx, user_id in enumerate(batch['eval_user_ids']):
            user_embed = user_embeds[user_id]
            
            # Calculate scores with all items
            scores = torch.matmul(user_embed, entity_embeds.t())
            
            # Exclude training items
            train_items = batch['train_items'][idx]
            scores[train_items] = -float('inf')
            
            # Ground truth
            test_items = batch['test_items'][idx]
            
            # Calculate metrics for different K values
            for k in [10, 20, 50]:
                # Handle case where number of items < k
                k_actual = min(k, scores.size(0))
                _, top_indices = torch.topk(scores, k_actual)
                recommended = set(top_indices.cpu().numpy())
                
                # Handle different formats of test_items
                if torch.is_tensor(test_items):
                    ground_truth = set(test_items.cpu().numpy())
                else:
                    ground_truth = set(test_items)
                
                # Recall@K
                recall = len(recommended & ground_truth) / len(ground_truth) if ground_truth else 0
                metrics[f'recall@{k}'].append(recall)
                
                # Precision@K
                precision = len(recommended & ground_truth) / k_actual
                metrics[f'precision@{k}'].append(precision)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(top_indices.cpu().numpy(), ground_truth, k_actual)
                metrics[f'ndcg@{k}'].append(ndcg)
        
        # Average metrics
        avg_metrics = {f'{stage}_{key}': np.mean(values) 
                      for key, values in metrics.items()}
        
        self.log_dict(avg_metrics, prog_bar=True, sync_dist=True)
        
        return avg_metrics
    
    def _calculate_ndcg(self, ranked_list, ground_truth, k):
        """Calculate NDCG@k"""
        dcg = 0.0
        for i, item in enumerate(ranked_list[:k]):
            if item in ground_truth:
                dcg += 1 / np.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def configure_optimizers(self):
        # Scale learning rate for DDP
        effective_lr = self.lr
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'world_size') and self.trainer.world_size > 1:
            # Linear scaling rule for distributed training
            effective_lr = self.lr * self.trainer.world_size
            print(f"Scaling learning rate from {self.lr} to {effective_lr} for {self.trainer.world_size} GPUs")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=effective_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_recall@20',
                'frequency': 1,
                'interval': 'epoch',
                'strict': False  # Don't error if metric is not available yet
            }
        }
    
    def get_user_embeddings(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        """Get all user embeddings"""
        self.eval()
        with torch.no_grad():
            user_embeds, _ = self(edge_index_ui, edge_index_kg, edge_type_kg)
        return user_embeds
    
    def get_item_embeddings(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        """Get all item embeddings"""
        self.eval()
        with torch.no_grad():
            _, entity_embeds = self(edge_index_ui, edge_index_kg, edge_type_kg)
        return entity_embeds
    
    def recommend_items(self, user_id, edge_index_ui, edge_index_kg=None, 
                       edge_type_kg=None, exclude_items=None, k=10):
        """Recommend top-k items for a user"""
        self.eval()
        with torch.no_grad():
            user_embeds, entity_embeds = self(edge_index_ui, edge_index_kg, edge_type_kg)
            
            user_embed = user_embeds[user_id]
            scores = torch.matmul(user_embed, entity_embeds.t())
            
            # Exclude items if specified
            if exclude_items is not None:
                scores[exclude_items] = -float('inf')
            
            # Get top-k items
            _, top_indices = torch.topk(scores, k)
            
            return top_indices.cpu().numpy()