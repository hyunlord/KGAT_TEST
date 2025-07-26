import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
import logging


class KGATConv(MessagePassing):
    """원본 KGAT의 aggregation 방식을 따르는 convolution layer"""
    
    def __init__(self, in_channels, out_channels, aggregator_type='bi', dropout=0.1):
        super(KGATConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        
        # 가중치 행렬
        if aggregator_type == 'bi':
            self.W = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)
        elif aggregator_type == 'gcn':
            self.W = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)
        else:  # graphsage
            self.W = nn.Parameter(torch.randn(2 * in_channels, out_channels) * 0.01)
            
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None):
        # 이웃 정보 수집
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        if self.aggregator_type == 'bi':
            # Bi-interaction: element-wise product
            out = x * out
            out = torch.matmul(out, self.W)
        elif self.aggregator_type == 'gcn':
            # GCN style
            out = torch.matmul(out, self.W)
        else:  # graphsage
            # Concatenate self and neighbor
            out = torch.cat([x, out], dim=1)
            out = torch.matmul(out, self.W)
        
        out = self.leaky_relu(out)
        out = self.dropout_layer(out)
        out = F.normalize(out, p=2, dim=1)
        
        return out
    
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j


class KGATLightningFixed(pl.LightningModule):
    """원본 KGAT 논문의 방식을 충실히 따르는 PyTorch Lightning 구현"""
    
    def __init__(self, config):
        super(KGATLightningFixed, self).__init__()
        self.save_hyperparameters()
        
        # 모델 설정
        self.n_users = config.n_users
        self.n_entities = config.n_entities
        self.n_relations = config.n_relations
        
        self.embedding_size = config.embedding_size
        self.layer_sizes = config.layer_sizes
        self.aggregator = config.aggregator
        self.dropout_rates = config.dropout_rates
        
        self.lr = config.lr
        self.weight_decay = config.reg_weight  # config에서 reg_weight 사용
        self.batch_size = getattr(config, 'batch_size', 1024)  # 기본값 1024
        
        # 임베딩 레이어 (원본과 동일한 초기화)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        
        # 초기화
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.entity_embedding.weight, std=0.01)
        nn.init.normal_(self.relation_embedding.weight, std=0.01)
        
        # KGAT 레이어들
        self.conv_layers = nn.ModuleList()
        in_channels = self.embedding_size
        
        for i, out_channels in enumerate(self.layer_sizes):
            self.conv_layers.append(
                KGATConv(in_channels, out_channels, 
                        aggregator_type=self.aggregator,
                        dropout=self.dropout_rates[i])
            )
            in_channels = out_channels
        
        # 메트릭 추적
        self.train_losses = []
        self.val_metrics = []
        
        # Edge weight를 저장할 버퍼
        self.register_buffer('edge_weight_ui', None)
        self.register_buffer('edge_weight_kg', None)
        
    def compute_edge_weights(self, edge_index, num_nodes):
        """원본 논문의 Laplacian normalization 구현"""
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Compute degree
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()
        
        # Compute normalization: D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Edge weights: D^{-1/2} * A * D^{-1/2}
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return edge_index, edge_weight
    
    def setup(self, stage=None):
        """그래프 정규화 사전 계산"""
        # 이 메서드는 DataModule에서 그래프 구조를 받아서 호출되어야 함
        pass
    
    def forward(self, edge_index_ui, edge_weight_ui=None):
        """전체 사용자/아이템 임베딩 계산"""
        # 사용자와 엔티티 임베딩 결합
        x = torch.cat([self.user_embedding.weight, self.entity_embedding.weight], dim=0)
        all_embeddings = [x]
        
        # 각 레이어를 통과
        for conv in self.conv_layers:
            x = conv(x, edge_index_ui, edge_weight_ui)
            all_embeddings.append(x)
        
        # 모든 레이어의 임베딩을 concat (원본 논문 방식)
        all_embeddings = torch.cat(all_embeddings, dim=1)
        
        # 사용자와 아이템 임베딩 분리
        user_embeddings = all_embeddings[:self.n_users]
        item_embeddings = all_embeddings[self.n_users:self.n_users + self.n_entities]
        
        return user_embeddings, item_embeddings
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        """원본과 동일한 BPR loss"""
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        
        # 원본과 동일한 정규화
        regularizer = (torch.norm(users, 2, dim=1).pow(2).mean() + 
                      torch.norm(pos_items, 2, dim=1).pow(2).mean() + 
                      torch.norm(neg_items, 2, dim=1).pow(2).mean()) / 2
        regularizer = self.weight_decay * regularizer
        
        # 원본과 동일한 loss (logsigmoid 사용)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        return mf_loss + regularizer
    
    def training_step(self, batch, batch_idx):
        edge_index_ui = batch['edge_index_ui']
        
        # Edge weight 계산 (첫 번째 배치에서만)
        if self.edge_weight_ui is None:
            num_nodes = self.n_users + self.n_entities
            _, self.edge_weight_ui = self.compute_edge_weights(edge_index_ui, num_nodes)
        
        # Forward pass
        user_embeddings, item_embeddings = self(edge_index_ui, self.edge_weight_ui)
        
        # 배치에서 임베딩 추출
        users = user_embeddings[batch['user_ids']]
        pos_items = item_embeddings[batch['pos_item_ids']]
        neg_items = item_embeddings[batch['neg_item_ids']]
        
        # Loss 계산
        loss = self.create_bpr_loss(users, pos_items, neg_items)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        edge_index_ui = batch['edge_index_ui']
        
        # Edge weight 계산
        if self.edge_weight_ui is None:
            num_nodes = self.n_users + self.n_entities
            _, self.edge_weight_ui = self.compute_edge_weights(edge_index_ui, num_nodes)
        
        # Forward pass
        user_embeddings, item_embeddings = self(edge_index_ui, self.edge_weight_ui)
        
        # 평가를 위한 데이터
        eval_users = batch['eval_user_ids']
        train_items = batch['train_items']
        test_items = batch['test_items']
        
        # 평가 메트릭 계산
        metrics = self._compute_metrics(
            user_embeddings[eval_users], 
            item_embeddings,
            train_items,
            test_items,
            k_list=[20, 40]
        )
        
        # 로깅
        for k in [20, 40]:
            self.log(f'val_recall@{k}', metrics[f'recall@{k}'], on_epoch=True, prog_bar=True)
            self.log(f'val_ndcg@{k}', metrics[f'ndcg@{k}'], on_epoch=True)
            self.log(f'val_precision@{k}', metrics[f'precision@{k}'], on_epoch=True)
        
        return metrics
    
    def get_all_embeddings(self):
        """모든 노드의 임베딩을 반환 (관계 비교용)"""
        with torch.no_grad():
            # 전체 임베딩 계산
            ego_embed = torch.cat([self.user_embedding.weight, self.entity_embedding.weight], dim=0)
            all_embed = [ego_embed]
            
            # 각 레이어별로 전파
            for i, conv in enumerate(self.conv_layers):
                if hasattr(self, 'edge_index_ui') and self.edge_index_ui is not None:
                    ego_embed = conv(ego_embed, self.edge_index_ui, self.edge_weight_ui)
                else:
                    # 더미 엣지로 계산 (평가용)
                    n_nodes = ego_embed.size(0)
                    edge_index = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)]).to(ego_embed.device)
                    ego_embed = conv(ego_embed, edge_index)
                
                all_embed.append(ego_embed)
            
            # 모든 레이어 결합
            all_embed = torch.cat(all_embed, dim=1)
            return all_embed
    
    def set_graph_data(self, adj_mat, kg_mat=None):
        """그래프 데이터 설정 (평가용)"""
        # scipy sparse matrix를 edge_index로 변환
        coo = adj_mat.tocoo()
        edge_index = torch.LongTensor(np.vstack((coo.row, coo.col)))
        self.edge_index_ui = edge_index.to(self.device)
        
        # edge weight 계산
        num_nodes = adj_mat.shape[0]
        _, self.edge_weight_ui = self.compute_edge_weights(self.edge_index_ui, num_nodes)
    
    def _compute_metrics(self, user_embeddings, item_embeddings, train_items, test_items, k_list):
        """평가 메트릭 계산"""
        metrics = {}
        
        # 각 사용자에 대해
        recalls, ndcgs, precisions = {k: [] for k in k_list}, {k: [] for k in k_list}, {k: [] for k in k_list}
        
        for i, user_emb in enumerate(user_embeddings):
            # 점수 계산
            scores = torch.matmul(user_emb, item_embeddings.t())
            
            # 학습 데이터에서 본 아이템 제외
            if i < len(train_items) and len(train_items[i]) > 0:
                scores[train_items[i]] = -float('inf')
            
            # Top-k 추천
            for k in k_list:
                _, topk_indices = torch.topk(scores, k)
                topk_indices = topk_indices.cpu().numpy()
                
                # 테스트 아이템과 비교
                if i < len(test_items) and len(test_items[i]) > 0:
                    test_set = set(test_items[i])
                    hits = [1 if idx in test_set else 0 for idx in topk_indices]
                    
                    # Recall@k
                    recalls[k].append(sum(hits) / len(test_set))
                    
                    # Precision@k
                    precisions[k].append(sum(hits) / k)
                    
                    # NDCG@k
                    dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
                    idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(test_set), k))])
                    ndcgs[k].append(dcg / idcg if idcg > 0 else 0)
        
        # 평균 계산
        for k in k_list:
            metrics[f'recall@{k}'] = np.mean(recalls[k]) if recalls[k] else 0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs[k]) if ndcgs[k] else 0
            metrics[f'precision@{k}'] = np.mean(precisions[k]) if precisions[k] else 0
        
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_train_epoch_end(self):
        # 에폭 종료 시 로깅
        avg_loss = torch.stack([self.trainer.callback_metrics.get('train_loss', torch.tensor(0.))]).mean()
        self.train_losses.append(avg_loss.item())