"""
Fixed KGAT model with proper item indexing
"""
from .kgat_lightning import KGATLightning
import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


class KGATLightningFixed(KGATLightning):
    """수정된 KGAT 모델 - 평가 시 아이템만 사용"""
    
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
        
        # 아이템 임베딩만 추출 (엔티티에는 아이템 외의 것들도 포함됨)
        # 처음 n_items개가 아이템 임베딩
        item_embeds = entity_embeds[:self.n_items] if hasattr(self, 'n_items') else entity_embeds
        
        # 배치 사용자 평가
        metrics = defaultdict(list)
        
        for idx, user_id in enumerate(batch['eval_user_ids']):
            user_embed = user_embeds[user_id]
            
            # 모든 아이템과의 점수 계산 (아이템만!)
            scores = torch.matmul(user_embed, item_embeds.t())
            
            # 학습 아이템 제외
            train_items = batch['train_items'][idx]
            if len(train_items) > 0:
                # 아이템 인덱스가 범위 내에 있는지 확인
                valid_train_items = [item for item in train_items if item < scores.size(0)]
                if valid_train_items:
                    scores[valid_train_items] = -float('inf')
            
            # 정답 데이터
            test_items = batch['test_items'][idx]
            
            # 유효한 테스트 아이템만 사용
            if torch.is_tensor(test_items):
                test_items = test_items.cpu().numpy()
            valid_test_items = [item for item in test_items if item < scores.size(0)]
            
            if len(valid_test_items) == 0:
                continue  # 이 사용자는 건너뛰기
            
            # 다양한 K값에서 메트릭 계산
            for k in [10, 20, 50]:
                # 아이템 수가 k보다 적을 수 있으므로 min 사용
                k_actual = min(k, scores.size(0))
                _, top_indices = torch.topk(scores, k_actual)
                recommended = set(top_indices.cpu().numpy())
                ground_truth = set(valid_test_items)
                
                # Recall@K
                recall = len(recommended & ground_truth) / len(ground_truth)
                metrics[f'recall@{k}'].append(recall)
                
                # Precision@K
                precision = len(recommended & ground_truth) / k_actual
                metrics[f'precision@{k}'].append(precision)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(top_indices.cpu().numpy(), ground_truth, k_actual)
                metrics[f'ndcg@{k}'].append(ndcg)
        
        # Average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            if len(values) > 0:
                avg_metrics[f'{stage}_{key}'] = np.mean(values)
            else:
                avg_metrics[f'{stage}_{key}'] = 0.0
        
        self.log_dict(avg_metrics, prog_bar=True, sync_dist=True)
        
        return avg_metrics