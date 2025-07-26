"""
원본 KGAT 논문의 데이터 로더 재현
"""
import numpy as np
import random as rd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd


class DataLoaderOriginal(object):
    """원본 KGAT의 데이터 로더 재현"""
    
    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.dataset
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        
        self.logging = logging
        
        # 파일 경로
        data_directory = 'data/' + self.data_name + '/'
        self.train_file = data_directory + 'train.txt'
        self.test_file = data_directory + 'test.txt'
        self.kg_file = data_directory + 'kg_final.txt'
        
        # 데이터 로드
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
        self.statistic_cf()
        
        # KG 데이터 로드
        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)
        
        # user-item graph 생성
        self.create_adjacency_dict()
        self.create_laplacian_dict()
        
    def load_cf(self, filename):
        """CF 데이터 로드"""
        user = []
        item = []
        user_dict = dict()
        
        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]
            
            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))
                
                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids
        
        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict
    
    def statistic_cf(self):
        """CF 데이터 통계"""
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
    
    def load_kg(self, filename):
        """KG 데이터 로드"""
        kg_data = []
        lines = open(filename, 'r').readlines()
        
        for l in lines:
            line = l.strip()
            if len(line) > 0:
                h, r, t = line.split()
                h, r, t = int(h), int(r), int(t)
                kg_data.append([h, r, t])
        
        return np.array(kg_data, dtype=np.int32)
    
    def construct_data(self, kg_data):
        """KG 데이터 구성"""
        # kg dict 재구성
        n_relations = max(kg_data[:, 1]) + 1
        
        # inverse kg dict
        n_users = self.n_users
        n_entities = max(max(kg_data[:, 0]), max(kg_data[:, 2])) + 1
        self.n_entities = n_entities
        self.n_relations = n_relations
        
        # cf 데이터와 kg 데이터 정렬
        self.cf_train_data = (
            np.array(list(map(lambda d: d + n_entities, self.cf_train_data[0]))).astype(np.int32),
            self.cf_train_data[1].astype(np.int32)
        )
        self.cf_test_data = (
            np.array(list(map(lambda d: d + n_entities, self.cf_test_data[0]))).astype(np.int32),
            self.cf_test_data[1].astype(np.int32)
        )
        
        self.train_user_dict = {k + n_entities: np.unique(v).astype(np.int32) 
                                for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + n_entities: np.unique(v).astype(np.int32) 
                               for k, v in self.test_user_dict.items()}
        
        # kg에 user 추가
        cf2kg_train_data = pd.DataFrame(
            np.zeros((self.n_cf_train, 3), dtype=np.int32), 
            columns=['h', 'r', 't']
        )
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        
        inverse_cf2kg_train_data = pd.DataFrame(
            np.ones((self.n_cf_train, 3), dtype=np.int32), 
            columns=['h', 'r', 't']
        )
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]
        
        self.kg_train_data = pd.concat([
            cf2kg_train_data, inverse_cf2kg_train_data, 
            pd.DataFrame(kg_data, columns=['h', 'r', 't'])
        ], ignore_index=True)
        
        self.n_kg_train = len(self.kg_train_data)
        
        # 관계별 kg dict 구성
        self.train_kg_dict = defaultdict(list)
        self.train_relation_dict = defaultdict(list)
        
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
    
    def create_adjacency_dict(self):
        """인접 행렬 생성"""
        self.adjacency_dict = {}
        
        # user-item 상호작용을 위한 행렬
        def _bi_norm_lap(adj):
            """Bi-normalized Laplacian"""
            rowsum = np.array(adj.sum(1))
            
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()
        
        def _si_norm_lap(adj):
            """Simple normalized Laplacian"""
            rowsum = np.array(adj.sum(1))
            
            # Avoid divide by zero warning
            d_inv = np.zeros_like(rowsum, dtype=np.float32).flatten()
            d_inv[rowsum > 0] = np.power(rowsum[rowsum > 0], -1).flatten()
            d_mat_inv = sp.diags(d_inv)
            
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()
        
        # 기본 인접 행렬 생성
        cf = self.cf_train_data
        cf = np.vstack([cf, [cf[1], cf[0]]])  # 양방향
        
        vals = [1.] * len(cf[0])
        mat = sp.coo_matrix((vals, (cf[0], cf[1])), 
                           shape=(self.n_users + self.n_entities, self.n_users + self.n_entities))
        
        if self.args.adj_type == 'bi':
            self.adjacency_dict['plain_adj'] = _bi_norm_lap(mat)
        else:
            self.adjacency_dict['plain_adj'] = _si_norm_lap(mat)
    
    def create_laplacian_dict(self):
        """KG를 위한 Laplacian 생성"""
        def _build_kg_mat(kg_dict):
            """KG adjacency matrix 생성"""
            n_nodes = self.n_users + self.n_entities
            
            # 기본 상호작용
            a_rows = self.cf_train_data[0]
            a_cols = self.cf_train_data[1]
            a_vals = [1.] * len(a_rows)
            
            # Inverse 상호작용
            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)
            
            # KG 트리플
            kg_rows = []
            kg_cols = []
            kg_vals = []
            
            for h in kg_dict:
                for t, r in kg_dict[h]:
                    kg_rows.append(h)
                    kg_cols.append(t)
                    kg_vals.append(1.)
            
            rows = np.concatenate([a_rows, b_rows, kg_rows], axis=0)
            cols = np.concatenate([a_cols, b_cols, kg_cols], axis=0)
            vals = np.concatenate([a_vals, b_vals, kg_vals], axis=0)
            
            mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
            return mat
        
        kg_mat = _build_kg_mat(self.train_kg_dict)
        self.laplacian_dict = {'kg_mat': kg_mat}
    
    def sample_pos_items_for_u(self, u, num):
        """사용자를 위한 positive 아이템 샘플링"""
        pos_items = self.train_user_dict[u]
        n_pos_items = len(pos_items)
        
        pos_batch = []
        while True:
            if len(pos_batch) == num:
                break
            
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]
            
            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        
        return pos_batch
    
    def sample_neg_items_for_u(self, u, num):
        """사용자를 위한 negative 아이템 샘플링"""
        neg_items = []
        while True:
            if len(neg_items) == num:
                break
            neg_i_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            
            if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                neg_items.append(neg_i_id)
        
        return neg_items
    
    def generate_cf_batch(self, user_list, batch_size):
        """CF 배치 생성"""
        all_size = len(user_list)
        total_batch = all_size // batch_size + 1
        
        user_list_batch = []
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = start + batch_size
            
            user_list_batch.append(user_list[start:end])
        
        return user_list_batch
    
    def print_info(self, logging):
        """데이터 정보 출력"""
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_relations:       %d' % self.n_relations)
        
        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)
        
        logging.info('n_kg_train:        %d' % self.n_kg_train)