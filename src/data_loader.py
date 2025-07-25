import os
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse as sp


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.n_users = 0
        self.n_items = 0
        self.n_entities = 0
        self.n_relations = 0
        
        self.train_data = defaultdict(list)
        self.test_data = defaultdict(list)
        self.kg_data = []
        
    def load_cf_data(self):
        """Load collaborative filtering data (user-item interactions)"""
        train_file = os.path.join(self.data_path, 'train.txt')
        test_file = os.path.join(self.data_path, 'test.txt')
        
        # Load training data
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    items = [int(item) for item in parts[1:]]
                    self.train_data[user] = items
                    self.n_users = max(self.n_users, user)
                    self.n_items = max(self.n_items, max(items))
        
        # Load test data
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    items = [int(item) for item in parts[1:]]
                    self.test_data[user] = items
                    self.n_items = max(self.n_items, max(items))
        
        self.n_users += 1
        self.n_items += 1
        self.n_entities = self.n_items
        
        print(f"CF Data loaded: {self.n_users} users, {self.n_items} items")
        print(f"Train interactions: {sum(len(items) for items in self.train_data.values())}")
        print(f"Test interactions: {sum(len(items) for items in self.test_data.values())}")
        
    def load_kg_data(self):
        """Load knowledge graph data"""
        kg_file = os.path.join(self.data_path, 'kg_final.txt')
        
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
        
        print(f"KG Data loaded: {self.n_entities} entities, {self.n_relations} relations")
        print(f"KG triples: {len(self.kg_data)}")
        
    def create_adjacency_matrix(self):
        """Create adjacency matrix for user-item bipartite graph"""
        row = []
        col = []
        
        for user, items in self.train_data.items():
            for item in items:
                row.append(user)
                col.append(self.n_users + item)
                
        # Add symmetric edges
        row_symmetric = [c for c in col]
        col_symmetric = [r for r in row]
        
        row.extend(row_symmetric)
        col.extend(col_symmetric)
        
        data = np.ones(len(row))
        adj_mat = sp.csr_matrix((data, (row, col)), 
                               shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        
        return adj_mat
    
    def create_kg_adjacency_matrix(self):
        """Create adjacency matrices for knowledge graph"""
        kg_dict = defaultdict(list)
        relation_dict = defaultdict(list)
        
        for head, relation, tail in self.kg_data:
            kg_dict[head].append((tail, relation))
            
        return kg_dict, relation_dict
    
    def generate_test_batch(self, batch_size=1024):
        """Generate batches for testing"""
        test_users = list(self.test_data.keys())
        n_test_users = len(test_users)
        
        for start_idx in range(0, n_test_users, batch_size):
            end_idx = min(start_idx + batch_size, n_test_users)
            batch_users = test_users[start_idx:end_idx]
            
            yield batch_users, [self.test_data[u] for u in batch_users]