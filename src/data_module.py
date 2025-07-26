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
        
        # 반복을 위한 사용자 리스트 생성
        self.users = list(user_item_dict.keys())
        
        # 부정 샘플링을 위한 아이템 집합 생성
        self.all_items = set(range(n_items))
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = self.user_item_dict[user]
        
        if self.is_training:
            # 하나의 긍정 아이템 샘플링
            pos_item = np.random.choice(pos_items)
            
            # 부정 아이템 샘플링 (더 효율적인 방법)
            user_items = set(pos_items)
            # 가능한 모든 부정 아이템 후보
            neg_candidates = list(self.all_items - user_items)
            
            # 충분한 후보가 있는 경우 샘플링
            if len(neg_candidates) >= self.neg_sample_size:
                neg_items = np.random.choice(neg_candidates, 
                                           size=self.neg_sample_size, 
                                           replace=False).tolist()
            else:
                # 후보가 부족한 경우 반복 허용
                neg_items = np.random.choice(neg_candidates, 
                                           size=self.neg_sample_size, 
                                           replace=True).tolist()
            
            return {
                'user_id': user,
                'pos_item_id': pos_item,
                'neg_item_ids': neg_items[0] if self.neg_sample_size == 1 else neg_items
            }
        else:
            # 평가를 위해 사용자와 해당 아이템 반환
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
        
        # 데이터 통계
        self.n_users = 0
        self.n_items = 0
        self.n_entities = 0
        self.n_relations = 0
        
        # 데이터 저장소
        self.train_user_dict = defaultdict(list)
        self.test_user_dict = defaultdict(list)
        self.kg_data = []
        
        # 그래프 구조
        self.edge_index_ui = None
        self.edge_index_kg = None
        self.edge_type_kg = None
        
    def prepare_data(self):
        """필요 시 데이터 다운로드"""
        # 데이터가 없을 경우 여기서 다운로드
        pass
    
    def setup(self, stage=None):
        """데이터 로드 및 처리"""
        # CF 데이터 로드
        self._load_cf_data()
        
        # KG 데이터 로드
        self._load_kg_data()
        
        # 그래프 구조 생성
        self._create_graph_structures()
        
        print(f"데이터 로드 성공:")
        print(f"  사용자: {self.n_users}")
        print(f"  아이템: {self.n_items}")
        print(f"  엔티티: {self.n_entities}")
        print(f"  관계: {self.n_relations}")
        print(f"  학습 상호작용: {sum(len(items) for items in self.train_user_dict.values())}")
        print(f"  테스트 상호작용: {sum(len(items) for items in self.test_user_dict.values())}")
        print(f"  KG 트리플: {len(self.kg_data)}")
    
    def _load_cf_data(self):
        """협업 필터링 데이터 로드"""
        train_file = os.path.join(self.data_dir, 'train.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')
        
        # 파일 존재 확인
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"테스트 데이터 파일을 찾을 수 없습니다: {test_file}")
        
        # 학습 데이터 로드
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
        self.n_entities = self.n_items  # 초기에는 엔티티가 아이템
        
        # 데이터 검증
        if self.n_users == 1:  # 0에서 1을 더했으므로
            raise ValueError("학습 데이터에서 사용자를 찾을 수 없습니다")
        if self.n_items == 1:
            raise ValueError("학습 데이터에서 아이템을 찾을 수 없습니다")
    
    def _load_kg_data(self):
        """지식 그래프 데이터 로드"""
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
            print(f"경고: KG 파일 {kg_file}을 찾을 수 없습니다. KG 데이터 없이 진행합니다.")
            self.n_relations = 1  # 더미 관계
    
    def _create_graph_structures(self):
        """사용자-아이템 및 지식 그래프를 위한 엣지 인덱스 생성"""
        # 사용자-아이템 이분 그래프
        edge_list_ui = []
        for user, items in self.train_user_dict.items():
            for item in items:
                # 양방향 엣지 추가
                edge_list_ui.append([user, self.n_users + item])
                edge_list_ui.append([self.n_users + item, user])
        
        self.edge_index_ui = torch.tensor(edge_list_ui, dtype=torch.long).t()
        
        # 지식 그래프
        if self.kg_data:
            edge_list_kg = []
            edge_types = []
            
            for head, relation, tail in self.kg_data:
                # 현재는 아이템 간의 관계만 사용
                # TODO: 모든 엔티티를 활용하려면 모델 아키텍처 수정 필요
                if head < self.n_items and tail < self.n_items:
                    edge_list_kg.append([head, tail])
                    edge_types.append(relation)
            
            if edge_list_kg:
                self.edge_index_kg = torch.tensor(edge_list_kg, dtype=torch.long).t()
                self.edge_type_kg = torch.tensor(edge_types, dtype=torch.long)
    
    def train_dataloader(self):
        """학습 데이터로더 생성"""
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
        """검증 데이터로더 생성"""
        # 테스트 데이터가 있는 사용자 중 일부를 검증용으로 사용
        test_users_with_data = [u for u in self.test_user_dict.keys() 
                               if u in self.train_user_dict and len(self.test_user_dict[u]) > 0]
        
        # 최대 1000명의 사용자를 검증용으로 사용
        val_users = test_users_with_data[:min(1000, len(test_users_with_data))]
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
        """테스트 데이터로더 생성"""
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
        """학습용 collate 함수"""
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
        """평가용 collate 함수"""
        users = [b['user_id'] for b in batch]
        
        # 각 사용자의 학습 및 테스트 아이템 가져오기
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
        """데이터 통계 가져오기"""
        return {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_entities': self.n_entities,
            'n_relations': self.n_relations,
            'n_train_interactions': sum(len(items) for items in self.train_user_dict.values()),
            'n_test_interactions': sum(len(items) for items in self.test_user_dict.values()),
            'n_kg_triples': len(self.kg_data)
        }