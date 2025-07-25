import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class KGATConv(MessagePassing):
    def __init__(self, in_dim, out_dim, n_relations, aggregator='bi-interaction', dropout=0.1):
        super(KGATConv, self).__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations
        self.aggregator = aggregator
        self.dropout = dropout
        
        # Transformation matrices
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
        ])
        
        # Attention parameters
        if aggregator == 'bi-interaction':
            self.attention = nn.Linear(out_dim, 1)
            
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_type=None):
        return self.propagate(edge_index, x=x, edge_type=edge_type)
    
    def message(self, x_i, x_j, edge_type, index):
        # Transform neighbor embeddings
        if edge_type is not None:
            # For KG edges with relations
            x_j_list = []
            for idx, rel in enumerate(edge_type):
                x_j_list.append(self.W_r[rel](x_j[idx].unsqueeze(0)))
            x_j = torch.cat(x_j_list, dim=0)
        else:
            # For user-item edges
            x_j = self.W(x_j)
        
        # Calculate attention if using bi-interaction
        if self.aggregator == 'bi-interaction':
            x_i = self.W(x_i)
            
            # Element-wise multiplication for attention
            attention_input = x_i * x_j
            attention_scores = self.attention(attention_input)
            attention_scores = self.leaky_relu(attention_scores)
            
            # Return weighted messages
            return attention_scores * x_j
        else:
            return x_j
    
    def update(self, aggr_out):
        return self.dropout_layer(aggr_out)


class KGATLightning(pl.LightningModule):
    def __init__(self, config):
        super(KGATLightning, self).__init__()
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
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embed_dim)
        
        # KGAT layers
        self.convs = nn.ModuleList()
        in_dim = self.embed_dim
        for out_dim in self.layer_dims:
            self.convs.append(
                KGATConv(in_dim, out_dim, self.n_relations, 
                        self.aggregator, self.dropout)
            )
            in_dim = out_dim
            
        # Final dimension
        self.final_dim = self.layer_dims[-1] if self.layer_dims else self.embed_dim
        
        # Initialize embeddings
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
    def forward(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        # Get all embeddings
        user_embeds = self.user_embedding.weight
        entity_embeds = self.entity_embedding.weight
        
        # Initial embeddings: concatenate users and entities
        x = torch.cat([user_embeds, entity_embeds], dim=0)
        
        # Store embeddings at each layer for skip connections
        layer_embeds = [x]
        
        # Apply KGAT convolutions
        for conv in self.convs:
            # User-item graph convolution
            x_ui = conv(x, edge_index_ui)
            
            # Knowledge graph convolution (only for entities)
            if edge_index_kg is not None and edge_type_kg is not None:
                # Extract entity embeddings
                entity_x = x[self.n_users:]
                x_kg = conv(entity_x, edge_index_kg, edge_type_kg)
                
                # Update entity part of x
                x = torch.cat([x_ui[:self.n_users], x_ui[self.n_users:] + x_kg], dim=0)
            else:
                x = x_ui
                
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_embeds.append(x)
        
        # Aggregate embeddings from all layers
        final_embeds = torch.stack(layer_embeds, dim=1).mean(dim=1)
        
        # Split back to users and entities
        user_final = final_embeds[:self.n_users]
        entity_final = final_embeds[self.n_users:]
        
        return user_final, entity_final
    
    def bpr_loss(self, users, pos_items, neg_items):
        """Bayesian Personalized Ranking loss"""
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            users.norm(2).pow(2) + 
            pos_items.norm(2).pow(2) + 
            neg_items.norm(2).pow(2)
        ) / users.size(0)
        
        return bpr_loss + reg_loss
    
    def training_step(self, batch, batch_idx):
        edge_index_ui = batch['edge_index_ui']
        edge_index_kg = batch.get('edge_index_kg', None)
        edge_type_kg = batch.get('edge_type_kg', None)
        
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
        self.log('train_loss', loss, prog_bar=True)
        
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
            
            # Get ground truth
            test_items = batch['test_items'][idx]
            
            # Calculate metrics at different K
            for k in [10, 20, 50]:
                _, top_indices = torch.topk(scores, k)
                recommended = set(top_indices.cpu().numpy())
                ground_truth = set(test_items.cpu().numpy())
                
                # Recall@K
                recall = len(recommended & ground_truth) / len(ground_truth) if ground_truth else 0
                metrics[f'recall@{k}'].append(recall)
                
                # Precision@K
                precision = len(recommended & ground_truth) / k
                metrics[f'precision@{k}'].append(precision)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(top_indices.cpu().numpy(), ground_truth, k)
                metrics[f'ndcg@{k}'].append(ndcg)
        
        # Average metrics
        avg_metrics = {f'{stage}_{key}': np.mean(values) 
                      for key, values in metrics.items()}
        
        self.log_dict(avg_metrics, prog_bar=True)
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_recall@20',
                'frequency': 1
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