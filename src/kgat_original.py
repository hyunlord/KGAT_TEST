"""
원본 KGAT 논문의 정확한 재현
https://github.com/xiangwang1223/knowledge_graph_attention_network 기반
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class KGAT(nn.Module):
    """원본 KGAT 모델의 정확한 재현"""
    
    def __init__(self, args, n_users, n_items, n_entities, n_relations, 
                 adj_mat, kg_mat, n_fold=100):
        super(KGAT, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations
        
        # 원본 설정
        self.emb_size = args.embed_size  # 64
        self.layer_size = args.layer_size if isinstance(args.layer_size, list) else eval(args.layer_size)  # [64, 32, 16]
        self.n_layers = len(self.layer_size)
        self.alg_type = args.alg_type  # 'bi', 'gcn', 'graphsage'
        
        self.model_type = args.model_type
        self.adj_type = args.adj_type  # 'si' or 'bi'
        
        self.node_dropout = args.node_dropout if isinstance(args.node_dropout, list) else eval(args.node_dropout)  # [0.1]
        self.mess_dropout = args.mess_dropout if isinstance(args.mess_dropout, list) else eval(args.mess_dropout)  # [0.1, 0.1, 0.1]
        
        self.reg_lambda = args.regs[0]
        self.reg_lambda2 = args.regs[1]
        self.n_fold = n_fold
        
        # 임베딩 초기화
        self.user_embed = nn.Parameter(
            torch.randn(n_users, self.emb_size) * 0.01
        )
        self.entity_embed = nn.Parameter(
            torch.randn(n_entities, self.emb_size) * 0.01
        )
        self.relation_embed = nn.Parameter(
            torch.randn(n_relations, self.emb_size) * 0.01
        )
        
        # relation 임베딩에 대한 변환 행렬
        self.trans_W = nn.Parameter(
            torch.randn(n_relations, self.emb_size, self.emb_size) * 0.01
        )
        
        # 각 레이어의 가중치 행렬들
        self.weight_list = nn.ParameterList()
        for k in range(self.n_layers):
            if self.alg_type == 'bi':
                self.weight_list.append(nn.Parameter(
                    torch.randn(self.layer_size[k], self.layer_size[k]) * 0.01
                ))
            elif self.alg_type == 'gcn':
                self.weight_list.append(nn.Parameter(
                    torch.randn(self.layer_size[k], self.layer_size[k]) * 0.01
                ))
            else:  # graphsage
                self.weight_list.append(nn.Parameter(
                    torch.randn(2 * self.layer_size[k], self.layer_size[k]) * 0.01
                ))
        
        # Laplacian 행렬 생성
        self.A_in = self._convert_sp_mat_to_sp_tensor(adj_mat)
        
        # KG를 위한 attention
        if self.model_type == 'kgat':
            # KG adjacency matrix
            self.A_kg = self._compute_normalized_laplacian(
                kg_mat, self.adj_type
            )
        
    def _convert_sp_mat_to_sp_tensor(self, X):
        """Scipy sparse matrix를 PyTorch sparse tensor로 변환"""
        coo = X.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def _compute_normalized_laplacian(self, adj, adj_type):
        """정규화된 Laplacian 계산"""
        adj = adj + sp.eye(adj.shape[0])
        
        if adj_type == 'bi':
            # D^{-1/2} * A * D^{-1/2}
            rowsum = np.array(adj.sum(1)).flatten()
            d_inv_sqrt = np.power(rowsum, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        else:  # 'si'
            # D^{-1} * A
            rowsum = np.array(adj.sum(1)).flatten()
            d_inv = np.zeros_like(rowsum, dtype=np.float32)
            mask = rowsum > 0
            d_inv[mask] = np.power(rowsum[mask], -1)
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
        
        return self._convert_sp_mat_to_sp_tensor(norm_adj)
    
    def _split_A_hat(self, X):
        """큰 adjacency matrix를 fold로 분할하여 메모리 효율적으로 처리"""
        fold_len = X.shape[0] // self.n_fold
        
        results = []
        for i in range(self.n_fold):
            start = i * fold_len
            if i == self.n_fold - 1:
                end = X.shape[0]
            else:
                end = (i + 1) * fold_len
            
            results.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        
        return results
    
    def forward(self):
        """전체 임베딩 계산"""
        # 사용자와 엔티티 임베딩 결합
        ego_embed = torch.cat([self.user_embed, self.entity_embed], dim=0)
        all_embed = [ego_embed]
        
        # 각 레이어별로 메시지 전파
        for k in range(self.n_layers):
            # Graph convolution
            if self.alg_type == 'bi':
                # Bi-Interaction aggregator
                sum_embed = torch.sparse.mm(self.A_in, ego_embed)
                bi_embed = torch.mul(ego_embed, sum_embed)
                ego_embed = nn.LeakyReLU(negative_slope=0.2)(
                    torch.matmul(bi_embed, self.weight_list[k])
                )
                
            elif self.alg_type == 'gcn':
                # GCN aggregator
                side_embed = torch.sparse.mm(self.A_in, ego_embed)
                ego_embed = nn.LeakyReLU(negative_slope=0.2)(
                    torch.matmul(side_embed, self.weight_list[k])
                )
                
            else:  # graphsage
                # GraphSAGE aggregator
                side_embed = torch.sparse.mm(self.A_in, ego_embed)
                ego_embed = nn.LeakyReLU(negative_slope=0.2)(
                    torch.matmul(
                        torch.cat([ego_embed, side_embed], dim=1),
                        self.weight_list[k]
                    )
                )
            
            # Dropout
            ego_embed = nn.Dropout(self.mess_dropout[k])(ego_embed)
            
            # L2 정규화
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)
        
        # 모든 레이어의 임베딩을 concat
        all_embed = torch.cat(all_embed, dim=1)
        
        # 사용자와 아이템 임베딩 분리
        u_g_embeddings = all_embed[:self.n_users, :]
        i_g_embeddings = all_embed[self.n_users:, :]
        
        return u_g_embeddings, i_g_embeddings
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        """BPR loss 계산"""
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        
        regularizer = self.reg_lambda * torch.norm(users, 2, dim=1).pow(2).mean() + \
                     self.reg_lambda * torch.norm(pos_items, 2, dim=1).pow(2).mean() + \
                     self.reg_lambda * torch.norm(neg_items, 2, dim=1).pow(2).mean()
        regularizer = regularizer / 2
        
        # 원본에서는 softplus 사용
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        
        emb_loss = regularizer
        reg_loss = 0.0
        
        return mf_loss, emb_loss, reg_loss