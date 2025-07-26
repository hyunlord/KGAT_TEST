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
        
        # 변환 행렬
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)
        ])
        
        # 어텐션 파라미터
        if aggregator == 'bi-interaction':
            self.attention = nn.Linear(out_dim, 1)
            
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_type=None):
        return self.propagate(edge_index, x=x, edge_type=edge_type)
    
    def message(self, x_i, x_j, edge_type, index):
        # 이웃 임베딩 변환
        if edge_type is not None:
            # 관계가 있는 KG 엣지
            x_j_list = []
            for idx, rel in enumerate(edge_type):
                x_j_list.append(self.W_r[rel](x_j[idx].unsqueeze(0)))
            x_j = torch.cat(x_j_list, dim=0)
        else:
            # 사용자-아이템 엣지
            x_j = self.W(x_j)
        
        # bi-interaction 사용 시 어텐션 계산
        if self.aggregator == 'bi-interaction':
            x_i = self.W(x_i)
            
            # 어텐션을 위한 요소별 곱셈
            attention_input = x_i * x_j
            attention_scores = self.attention(attention_input)
            attention_scores = self.leaky_relu(attention_scores)
            
            # 가중치가 적용된 메시지 반환
            return attention_scores * x_j
        else:
            return x_j
    
    def update(self, aggr_out):
        return self.dropout_layer(aggr_out)


class KGATLightning(pl.LightningModule):
    def __init__(self, config):
        super(KGATLightning, self).__init__()
        self.save_hyperparameters()
        
        # 모델 파라미터
        self.n_users = config.n_users
        self.n_entities = config.n_entities
        self.n_relations = config.n_relations
        self.embed_dim = config.embed_dim
        self.layer_dims = config.layer_dims
        self.dropout = config.dropout
        self.reg_weight = config.reg_weight
        self.lr = config.lr
        self.aggregator = config.aggregator
        
        # 임베딩
        self.user_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embed_dim)
        # relation_embedding은 KGATConv 내부에서 처리되므로 여기서는 제거
        
        # KGAT 레이어
        self.convs = nn.ModuleList()
        in_dim = self.embed_dim
        for out_dim in self.layer_dims:
            self.convs.append(
                KGATConv(in_dim, out_dim, self.n_relations, 
                        self.aggregator, self.dropout)
            )
            in_dim = out_dim
            
        # 최종 차원: 초기 임베딩 + 모든 레이어의 출력 차원
        self.final_dim = self.embed_dim + sum(self.layer_dims)
        
        # 임베딩 초기화
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        
    def forward(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        # 모든 임베딩 가져오기
        user_embeds = self.user_embedding.weight
        entity_embeds = self.entity_embedding.weight
        
        # 초기 임베딩: 사용자와 엔티티 연결
        x = torch.cat([user_embeds, entity_embeds], dim=0)
        
        # 각 레이어의 출력 저장 (초기 임베딩 포함)
        all_embeddings = [x]
        
        # KGAT 컨볼루션 적용
        for conv in self.convs:
            # 사용자-아이템 그래프 컨볼루션
            x_ui = conv(x, edge_index_ui)
            
            # 지식 그래프 컨볼루션 (엔티티만 해당)
            if edge_index_kg is not None and edge_type_kg is not None:
                # 엔티티 임베딩 추출
                entity_x = x[self.n_users:]
                x_kg = conv(entity_x, edge_index_kg, edge_type_kg)
                
                # x의 엔티티 부분 업데이트
                x = torch.cat([x_ui[:self.n_users], x_ui[self.n_users:] + x_kg], dim=0)
            else:
                x = x_ui
                
            # 활성화 함수 및 드롭아웃 적용
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # L2 정규화 적용 (스케일 팩터 추가)
            x = F.normalize(x, p=2, dim=1) * 5.0  # 스케일 업으로 점수 범위 확대
            
            # 현재 레이어 출력 저장
            all_embeddings.append(x)
        
        # 모든 레이어의 임베딩을 연결 (concatenate)하여 최종 임베딩 생성
        # KGAT 원래 방식: 각 레이어의 출력을 연결
        final_embeds = torch.cat(all_embeddings, dim=1)
        
        # 사용자와 엔티티로 다시 분리
        user_final = final_embeds[:self.n_users]
        entity_final = final_embeds[self.n_users:]
        
        return user_final, entity_final
    
    def bpr_loss(self, users, pos_items, neg_items):
        """베이지안 개인화 순위 손실"""
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        
        # L2 정규화
        reg_loss = self.reg_weight * (
            users.norm(2).pow(2) + 
            pos_items.norm(2).pow(2) + 
            neg_items.norm(2).pow(2)
        ) / users.size(0)
        
        return bpr_loss + reg_loss
    
    def training_step(self, batch, batch_idx):
        # 모든 텐서를 현재 디바이스로 이동
        edge_index_ui = batch['edge_index_ui'].to(self.device)
        edge_index_kg = batch.get('edge_index_kg', None)
        edge_type_kg = batch.get('edge_type_kg', None)
        
        if edge_index_kg is not None:
            edge_index_kg = edge_index_kg.to(self.device)
        if edge_type_kg is not None:
            edge_type_kg = edge_type_kg.to(self.device)
        
        # 긍정 및 부정 아이템 샘플링
        user_ids = batch['user_ids']
        pos_item_ids = batch['pos_item_ids']
        neg_item_ids = batch['neg_item_ids']
        
        # 순전파
        user_embeds, entity_embeds = self(edge_index_ui, edge_index_kg, edge_type_kg)
        
        # 특정 임베딩 가져오기
        users = user_embeds[user_ids]
        pos_items = entity_embeds[pos_item_ids]
        neg_items = entity_embeds[neg_item_ids]
        
        # 손실 계산
        loss = self.bpr_loss(users, pos_items, neg_items)
        
        # 로깅
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, 'test')
    
    def _evaluation_step(self, batch, batch_idx, stage):
        # 모든 텐서를 현재 디바이스로 이동
        edge_index_ui = batch['edge_index_ui'].to(self.device)
        edge_index_kg = batch.get('edge_index_kg', None)
        edge_type_kg = batch.get('edge_type_kg', None)
        
        if edge_index_kg is not None:
            edge_index_kg = edge_index_kg.to(self.device)
        if edge_type_kg is not None:
            edge_type_kg = edge_type_kg.to(self.device)
        
        # 순전파
        user_embeds, entity_embeds = self(edge_index_ui, edge_index_kg, edge_type_kg)
        
        # 배치 사용자 평가
        metrics = defaultdict(list)
        
        for idx, user_id in enumerate(batch['eval_user_ids']):
            user_embed = user_embeds[user_id]
            
            # 모든 아이템과의 점수 계산
            scores = torch.matmul(user_embed, entity_embeds.t())
            
            # 학습 아이템 제외
            train_items = batch['train_items'][idx]
            scores[train_items] = -float('inf')
            
            # 정답 데이터
            test_items = batch['test_items'][idx]
            
            # 다양한 K값에서 메트릭 계산
            for k in [10, 20, 50]:
                # 아이템 수가 k보다 적을 수 있으므로 min 사용
                k_actual = min(k, scores.size(0))
                _, top_indices = torch.topk(scores, k_actual)
                recommended = set(top_indices.cpu().numpy())
                
                # test_items가 텐서가 아닐 수 있으므로 처리
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
        """NDCG@k 계산"""
        dcg = 0.0
        hits = 0
        for i, item in enumerate(ranked_list[:k]):
            if item in ground_truth:
                # 관련성 점수는 1 (이진 관련성)
                dcg += 1 / np.log2(i + 2)
                hits += 1
        
        # 이상적인 DCG: ground truth 아이템들이 상위에 랭크된 경우
        idcg = 0.0
        for i in range(min(len(ground_truth), k)):
            idcg += 1 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def configure_optimizers(self):
        # 기본 학습률 사용
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
                'strict': False  # 메트릭이 없을 때 에러 대신 경고만 표시
            }
        }
    
    def get_user_embeddings(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        """모든 사용자 임베딩 가져오기"""
        self.eval()
        with torch.no_grad():
            user_embeds, _ = self(edge_index_ui, edge_index_kg, edge_type_kg)
        return user_embeds
    
    def get_item_embeddings(self, edge_index_ui, edge_index_kg=None, edge_type_kg=None):
        """모든 아이템 임베딩 가져오기"""
        self.eval()
        with torch.no_grad():
            _, entity_embeds = self(edge_index_ui, edge_index_kg, edge_type_kg)
        return entity_embeds