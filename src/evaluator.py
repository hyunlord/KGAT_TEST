import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import heapq


class Evaluator:
    def __init__(self, model, data_loader, device='cpu', top_k=[20, 40, 60]):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.top_k = top_k
        
    def evaluate_standard(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        """Standard evaluation: recommend items based on user-item similarity only"""
        self.model.eval()
        
        metrics = defaultdict(lambda: defaultdict(float))
        
        with torch.no_grad():
            for batch_users, batch_test_items in tqdm(self.data_loader.generate_test_batch(), 
                                                     desc="Evaluating (Standard)"):
                # Get all user and item indices
                all_users = torch.arange(self.data_loader.n_users).to(self.device)
                all_items = torch.arange(self.data_loader.n_items).to(self.device)
                
                # Forward pass
                user_embeds, item_embeds = self.model(
                    all_users, all_items, 
                    edge_index_ui, edge_index_kg, edge_type_kg
                )
                
                # Calculate scores for batch users
                for idx, user in enumerate(batch_users):
                    user_embed = user_embeds[user].unsqueeze(0)
                    
                    # Calculate similarity with all items
                    scores = torch.matmul(user_embed, item_embeds.t()).squeeze()
                    
                    # Exclude training items
                    train_items = self.data_loader.train_data.get(user, [])
                    scores[train_items] = -float('inf')
                    
                    # Get top-k recommendations
                    test_items = set(batch_test_items[idx])
                    
                    for k in self.top_k:
                        _, top_indices = torch.topk(scores, k)
                        recommended = set(top_indices.cpu().numpy())
                        
                        # Calculate metrics
                        hits = recommended.intersection(test_items)
                        
                        metrics['recall'][k] += len(hits) / len(test_items)
                        metrics['precision'][k] += len(hits) / k
                        metrics['ndcg'][k] += self._calculate_ndcg(
                            top_indices.cpu().numpy(), test_items, k
                        )
        
        # Average metrics
        n_test_users = len(self.data_loader.test_data)
        for metric in metrics:
            for k in metrics[metric]:
                metrics[metric][k] /= n_test_users
                
        return dict(metrics)
    
    def evaluate_with_relations(self, edge_index_ui, edge_index_kg, edge_type_kg, 
                              relation_weights=None):
        """Enhanced evaluation: recommend items based on user+relation-item similarity"""
        self.model.eval()
        
        metrics = defaultdict(lambda: defaultdict(float))
        
        with torch.no_grad():
            for batch_users, batch_test_items in tqdm(self.data_loader.generate_test_batch(), 
                                                     desc="Evaluating (With Relations)"):
                # Get all user and item indices
                all_users = torch.arange(self.data_loader.n_users).to(self.device)
                all_items = torch.arange(self.data_loader.n_items).to(self.device)
                
                # Forward pass
                user_embeds, item_embeds = self.model(
                    all_users, all_items, 
                    edge_index_ui, edge_index_kg, edge_type_kg
                )
                
                # Get relation embeddings
                relation_embeds = self.model.relation_embedding.weight
                
                # Calculate scores for batch users
                for idx, user in enumerate(batch_users):
                    user_embed = user_embeds[user]
                    
                    # Standard user-item scores
                    user_item_scores = torch.matmul(user_embed, item_embeds.t())
                    
                    # Enhanced scores with relations
                    enhanced_scores = user_item_scores.clone()
                    
                    # For each relation, calculate user-relation-item scores
                    for rel_id in range(self.model.n_relations):
                        rel_embed = relation_embeds[rel_id]
                        
                        # Combine user and relation embeddings
                        user_rel_embed = user_embed + rel_embed
                        
                        # Calculate similarity with items
                        user_rel_item_scores = torch.matmul(user_rel_embed, item_embeds.t())
                        
                        # Weight the relation-based scores
                        weight = relation_weights[rel_id] if relation_weights else 1.0
                        enhanced_scores += weight * user_rel_item_scores
                    
                    # Normalize enhanced scores
                    enhanced_scores = enhanced_scores / (1 + self.model.n_relations)
                    
                    # Exclude training items
                    train_items = self.data_loader.train_data.get(user, [])
                    enhanced_scores[train_items] = -float('inf')
                    
                    # Get top-k recommendations
                    test_items = set(batch_test_items[idx])
                    
                    for k in self.top_k:
                        _, top_indices = torch.topk(enhanced_scores, k)
                        recommended = set(top_indices.cpu().numpy())
                        
                        # Calculate metrics
                        hits = recommended.intersection(test_items)
                        
                        metrics['recall'][k] += len(hits) / len(test_items)
                        metrics['precision'][k] += len(hits) / k
                        metrics['ndcg'][k] += self._calculate_ndcg(
                            top_indices.cpu().numpy(), test_items, k
                        )
        
        # Average metrics
        n_test_users = len(self.data_loader.test_data)
        for metric in metrics:
            for k in metrics[metric]:
                metrics[metric][k] /= n_test_users
                
        return dict(metrics)
    
    def _calculate_ndcg(self, ranked_list, ground_truth, k):
        """Calculate NDCG@k"""
        dcg = 0.0
        for i, item in enumerate(ranked_list[:k]):
            if item in ground_truth:
                dcg += 1 / np.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def get_item_popularity(self):
        """Calculate item popularity from training data"""
        item_popularity = defaultdict(int)
        
        for user, items in self.data_loader.train_data.items():
            for item in items:
                item_popularity[item] += 1
                
        return item_popularity
    
    def analyze_recommendations(self, user_id, edge_index_ui, edge_index_kg, edge_type_kg, k=20):
        """Analyze recommendations for a specific user using both methods"""
        self.model.eval()
        
        with torch.no_grad():
            # Get embeddings
            all_users = torch.arange(self.data_loader.n_users).to(self.device)
            all_items = torch.arange(self.data_loader.n_items).to(self.device)
            
            user_embeds, item_embeds = self.model(
                all_users, all_items, 
                edge_index_ui, edge_index_kg, edge_type_kg
            )
            
            user_embed = user_embeds[user_id]
            
            # Standard method
            standard_scores = torch.matmul(user_embed, item_embeds.t())
            
            # Enhanced method with relations
            relation_embeds = self.model.relation_embedding.weight
            enhanced_scores = standard_scores.clone()
            
            for rel_id in range(self.model.n_relations):
                rel_embed = relation_embeds[rel_id]
                user_rel_embed = user_embed + rel_embed
                user_rel_item_scores = torch.matmul(user_rel_embed, item_embeds.t())
                enhanced_scores += user_rel_item_scores
            
            enhanced_scores = enhanced_scores / (1 + self.model.n_relations)
            
            # Exclude training items
            train_items = self.data_loader.train_data.get(user_id, [])
            standard_scores[train_items] = -float('inf')
            enhanced_scores[train_items] = -float('inf')
            
            # Get top-k recommendations
            _, standard_top = torch.topk(standard_scores, k)
            _, enhanced_top = torch.topk(enhanced_scores, k)
            
            standard_recs = standard_top.cpu().numpy()
            enhanced_recs = enhanced_top.cpu().numpy()
            
            # Get test items
            test_items = set(self.data_loader.test_data.get(user_id, []))
            
            analysis = {
                'user_id': user_id,
                'standard_recommendations': standard_recs.tolist(),
                'enhanced_recommendations': enhanced_recs.tolist(),
                'standard_hits': list(set(standard_recs) & test_items),
                'enhanced_hits': list(set(enhanced_recs) & test_items),
                'overlap': len(set(standard_recs) & set(enhanced_recs)),
                'test_items': list(test_items)
            }
            
            return analysis