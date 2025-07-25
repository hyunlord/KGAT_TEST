import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional


class KGATDataset(Dataset):
    def __init__(self, user_item_dict, n_users, n_items, n_entities, 
                 kg_dict=None, neg_sample_size=1, is_training=True):
        self.user_item_dict = user_item_dict
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.kg_dict = kg_dict
        self.neg_sample_size = neg_sample_size
        self.is_training = is_training
        
        # Create user list for iteration
        self.users = list(user_item_dict.keys())
        
        # Create item set for negative sampling
        self.all_items = set(range(n_items))
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = self.user_item_dict[user]
        
        if self.is_training:
            # Sample one positive item
            pos_item = np.random.choice(pos_items)
            
            # Sample negative items
            neg_items = []
            user_items = set(pos_items)
            
            for _ in range(self.neg_sample_size):
                neg_item = np.random.randint(0, self.n_items)
                while neg_item in user_items:
                    neg_item = np.random.randint(0, self.n_items)
                neg_items.append(neg_item)
            
            return {
                'user_id': user,
                'pos_item_id': pos_item,
                'neg_item_ids': neg_items[0] if self.neg_sample_size == 1 else neg_items
            }
        else:
            # For evaluation, return user and their items
            return {
                'user_id': user,
                'pos_items': pos_items
            }


class KGATDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.neg_sample_size = config.neg_sample_size
        
        # Data statistics
        self.n_users = 0
        self.n_items = 0
        self.n_entities = 0
        self.n_relations = 0
        
        # Data storage
        self.train_user_dict = defaultdict(list)
        self.test_user_dict = defaultdict(list)
        self.kg_data = []
        
        # Graph structures
        self.edge_index_ui = None
        self.edge_index_kg = None
        self.edge_type_kg = None
        
    def prepare_data(self):
        """Download data if needed"""
        # This is where you would download data if it doesn't exist
        pass
    
    def setup(self, stage=None):
        """Load and process data"""
        # Load CF data
        self._load_cf_data()
        
        # Load KG data
        self._load_kg_data()
        
        # Create graph structures
        self._create_graph_structures()
        
        print(f"Data loaded successfully:")
        print(f"  Users: {self.n_users}")
        print(f"  Items: {self.n_items}")
        print(f"  Entities: {self.n_entities}")
        print(f"  Relations: {self.n_relations}")
        print(f"  Train interactions: {sum(len(items) for items in self.train_user_dict.values())}")
        print(f"  Test interactions: {sum(len(items) for items in self.test_user_dict.values())}")
        print(f"  KG triples: {len(self.kg_data)}")
    
    def _load_cf_data(self):
        """Load collaborative filtering data"""
        train_file = os.path.join(self.data_dir, 'train.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')
        
        # Load training data
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    items = [int(item) for item in parts[1:]]
                    self.train_user_dict[user] = items
                    self.n_users = max(self.n_users, user)
                    self.n_items = max(self.n_items, max(items))
        
        # Load test data
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    items = [int(item) for item in parts[1:]]
                    self.test_user_dict[user] = items
                    self.n_items = max(self.n_items, max(items))
        
        self.n_users += 1
        self.n_items += 1
        self.n_entities = self.n_items  # Initially, entities are items
    
    def _load_kg_data(self):
        """Load knowledge graph data"""
        kg_file = os.path.join(self.data_dir, 'kg_final.txt')
        
        if os.path.exists(kg_file):
            with open(kg_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        head, relation, tail = int(parts[0]), int(parts[1]), int(parts[2])
                        self.kg_data.append((head, relation, tail))
                        self.n_entities = max(self.n_entities, head, tail)
                        self.n_relations = max(self.n_relations, relation)
            
            self.n_entities += 1
            self.n_relations += 1
        else:
            print(f"Warning: KG file {kg_file} not found. Proceeding without KG data.")
            self.n_relations = 1  # Dummy relation
    
    def _create_graph_structures(self):
        """Create edge indices for user-item and knowledge graphs"""
        # User-item bipartite graph
        edge_list_ui = []
        for user, items in self.train_user_dict.items():
            for item in items:
                # Add bidirectional edges
                edge_list_ui.append([user, self.n_users + item])
                edge_list_ui.append([self.n_users + item, user])
        
        self.edge_index_ui = torch.tensor(edge_list_ui, dtype=torch.long).t()
        
        # Knowledge graph
        if self.kg_data:
            edge_list_kg = []
            edge_types = []
            
            for head, relation, tail in self.kg_data:
                if head < self.n_items and tail < self.n_items:
                    edge_list_kg.append([head, tail])
                    edge_types.append(relation)
            
            if edge_list_kg:
                self.edge_index_kg = torch.tensor(edge_list_kg, dtype=torch.long).t()
                self.edge_type_kg = torch.tensor(edge_types, dtype=torch.long)
    
    def train_dataloader(self):
        """Create training dataloader"""
        dataset = KGATDataset(
            self.train_user_dict,
            self.n_users,
            self.n_items,
            self.n_entities,
            neg_sample_size=self.neg_sample_size,
            is_training=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._train_collate_fn
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        # Use a subset of test users for validation
        val_users = list(self.test_user_dict.keys())[:1000]
        val_dict = {u: self.test_user_dict[u] for u in val_users}
        
        dataset = KGATDataset(
            val_dict,
            self.n_users,
            self.n_items,
            self.n_entities,
            is_training=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._eval_collate_fn
        )
    
    def test_dataloader(self):
        """Create test dataloader"""
        dataset = KGATDataset(
            self.test_user_dict,
            self.n_users,
            self.n_items,
            self.n_entities,
            is_training=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._eval_collate_fn
        )
    
    def _train_collate_fn(self, batch):
        """Collate function for training"""
        users = torch.tensor([b['user_id'] for b in batch], dtype=torch.long)
        pos_items = torch.tensor([b['pos_item_id'] for b in batch], dtype=torch.long)
        neg_items = torch.tensor([b['neg_item_ids'] for b in batch], dtype=torch.long)
        
        return {
            'user_ids': users,
            'pos_item_ids': pos_items,
            'neg_item_ids': neg_items,
            'edge_index_ui': self.edge_index_ui,
            'edge_index_kg': self.edge_index_kg,
            'edge_type_kg': self.edge_type_kg
        }
    
    def _eval_collate_fn(self, batch):
        """Collate function for evaluation"""
        users = [b['user_id'] for b in batch]
        
        # Get train and test items for each user
        train_items = [self.train_user_dict.get(u, []) for u in users]
        test_items = [b['pos_items'] for b in batch]
        
        return {
            'eval_user_ids': torch.tensor(users, dtype=torch.long),
            'train_items': train_items,
            'test_items': test_items,
            'edge_index_ui': self.edge_index_ui,
            'edge_index_kg': self.edge_index_kg,
            'edge_type_kg': self.edge_type_kg
        }
    
    def get_statistics(self):
        """Get data statistics"""
        return {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_entities': self.n_entities,
            'n_relations': self.n_relations,
            'n_train_interactions': sum(len(items) for items in self.train_user_dict.values()),
            'n_test_interactions': sum(len(items) for items in self.test_user_dict.values()),
            'n_kg_triples': len(self.kg_data)
        }