"""
KGAT 모델을 사용한 표준 추천 vs 관계 강화 추천 비교
TransR 방식을 올바르게 적용한 버전
"""

import os
import torch
import numpy as np
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import defaultdict

from kgat_original import KGAT
from data_loader_original import DataLoaderOriginal


class RelationEnhancedComparisonTransR:
    """TransR을 올바르게 적용한 관계 강화 추천"""
    
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
    def get_standard_recommendations(self, user_ids, k=20):
        """표준 방식: user-item 유사도만 사용"""
        self.model.eval()
        
        with torch.no_grad():
            # 전체 임베딩 계산
            u_embed, i_embed = self.model()
            
            recommendations = {}
            for u in user_ids:
                # 사용자 ID 변환 (엔티티 공간에서 원래 공간으로)
                u_original = u - self.data_loader.n_entities
                
                # 사용자 임베딩
                user_emb = u_embed[u_original]
                
                # 모든 아이템과의 점수 계산
                # i_embed에서 아이템만 추출 (전체 엔티티 중 아이템만)
                item_embed = i_embed[:self.data_loader.n_items]
                scores = torch.matmul(user_emb, item_embed.t())
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations
    
    def get_relation_enhanced_recommendations_transr(self, user_ids, k=20):
        """TransR 방식 관계 강화 추천"""
        self.model.eval()
        
        with torch.no_grad():
            # 전체 임베딩 계산
            u_embed, i_embed = self.model()
            
            # 원본 임베딩 (관계 변환용)
            u_embed_base = self.model.user_embed
            i_embed_base = self.model.entity_embed[:self.data_loader.n_items]
            
            # TransR 변환 행렬
            trans_W = self.model.trans_W  # [n_relations, emb_size, emb_size]
            
            recommendations = {}
            
            for u in user_ids:
                # 사용자 ID 변환
                u_original = u - self.data_loader.n_entities
                
                # 기본 사용자 임베딩 (첫 번째 레이어)
                user_emb_base = u_embed_base[u_original]
                
                # 관계별 강화된 임베딩 계산
                enhanced_scores = torch.zeros(self.data_loader.n_items).to(self.device)
                
                # 사용자가 상호작용한 아이템들
                if u in self.data_loader.train_user_dict:
                    interacted_items = self.data_loader.train_user_dict[u]
                    
                    # 각 상호작용한 아이템에서 다른 아이템으로의 관계 고려
                    for item in interacted_items:
                        # 이 아이템과 연결된 다른 아이템들 찾기
                        if item in self.data_loader.train_kg_dict:
                            # 아이템의 임베딩
                            item_emb = i_embed_base[item]
                            
                            for target, relation in self.data_loader.train_kg_dict[item]:
                                if target < self.data_loader.n_items:  # 아이템인 경우만
                                    # TransR: 관계 공간으로 변환
                                    # h_r = W_r * h
                                    # t_r = W_r * t
                                    W_r = trans_W[relation]  # [emb_size, emb_size]
                                    
                                    # 사용자와 아이템을 관계 공간으로 변환
                                    user_r = torch.matmul(user_emb_base, W_r)
                                    item_r = torch.matmul(item_emb, W_r)
                                    
                                    # 타겟 아이템도 관계 공간으로 변환
                                    target_emb = i_embed_base[target]
                                    target_r = torch.matmul(target_emb, W_r)
                                    
                                    # TransR 점수: ||h_r + r - t_r||
                                    # 여기서는 유사도로 변환 (거리가 가까울수록 높은 점수)
                                    relation_vec = self.model.relation_embed[relation]
                                    
                                    # 방법 1: 변환된 공간에서의 유사도
                                    # score = (user_r + relation_vec) · target_r
                                    enhanced_user_r = user_r + relation_vec
                                    score = torch.dot(enhanced_user_r, target_r)
                                    
                                    # 방법 2: TransR 거리 기반 (선택적)
                                    # distance = torch.norm(user_r + relation_vec - target_r, p=2)
                                    # score = 1.0 / (1.0 + distance)  # 거리를 유사도로 변환
                                    
                                    # 점수 누적
                                    enhanced_scores[target] += score
                
                # 기본 점수와 관계 강화 점수 결합 (전체 임베딩 사용)
                user_emb_full = u_embed[u_original]
                item_embed_full = i_embed[:self.data_loader.n_items]
                base_scores = torch.matmul(user_emb_full, item_embed_full.t())
                final_scores = base_scores + 0.3 * enhanced_scores  # 가중치 조절 가능
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    final_scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(final_scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations
    
    def get_relation_enhanced_recommendations_simple(self, user_ids, k=20):
        """단순 더하기 방식 (기존 구현)"""
        self.model.eval()
        
        with torch.no_grad():
            # 전체 임베딩 계산
            u_embed, i_embed = self.model()
            
            # 원본 임베딩 (관계 임베딩과 동일 차원)
            u_embed_base = self.model.user_embed
            i_embed_base = self.model.entity_embed[:self.data_loader.n_items]
            
            recommendations = {}
            
            for u in user_ids:
                # 사용자 ID 변환
                u_original = u - self.data_loader.n_entities
                
                # 기본 사용자 임베딩 (첫 번째 레이어)
                user_emb_base = u_embed_base[u_original]
                
                # 관계별 강화된 임베딩 계산
                enhanced_scores = torch.zeros(self.data_loader.n_items).to(self.device)
                
                # 사용자가 상호작용한 아이템들
                if u in self.data_loader.train_user_dict:
                    interacted_items = self.data_loader.train_user_dict[u]
                    
                    # 각 상호작용한 아이템에서 다른 아이템으로의 관계 고려
                    for item in interacted_items:
                        # 이 아이템과 연결된 다른 아이템들 찾기
                        if item in self.data_loader.train_kg_dict:
                            for target, relation in self.data_loader.train_kg_dict[item]:
                                if target < self.data_loader.n_items:  # 아이템인 경우만
                                    # 관계 임베딩 가져오기
                                    rel_emb = self.model.relation_embed[relation]
                                    
                                    # user + relation 임베딩 (같은 차원)
                                    enhanced_user = user_emb_base + 0.1 * rel_emb  # 가중치 조절 가능
                                    
                                    # 타겟 아이템과의 유사도 (기본 임베딩 사용)
                                    target_emb = i_embed_base[target]
                                    score = torch.dot(enhanced_user, target_emb)
                                    
                                    # 점수 누적
                                    enhanced_scores[target] += score
                
                # 기본 점수와 관계 강화 점수 결합 (전체 임베딩 사용)
                user_emb_full = u_embed[u_original]
                item_embed_full = i_embed[:self.data_loader.n_items]
                base_scores = torch.matmul(user_emb_full, item_embed_full.t())
                final_scores = base_scores + 0.3 * enhanced_scores  # 가중치 조절 가능
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    final_scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(final_scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations
    
    def evaluate_recommendations(self, recommendations, test_user_dict, k_list=[20]):
        """추천 결과 평가"""
        metrics = defaultdict(list)
        
        for u, rec_items in recommendations.items():
            if u not in test_user_dict or len(test_user_dict[u]) == 0:
                continue
                
            test_items = set(test_user_dict[u])
            
            for k in k_list:
                rec_k = rec_items[:k]
                hits = [1 if item in test_items else 0 for item in rec_k]
                
                # Recall@K
                recall = sum(hits) / len(test_items)
                metrics[f'recall@{k}'].append(recall)
                
                # Precision@K
                precision = sum(hits) / k
                metrics[f'precision@{k}'].append(precision)
                
                # NDCG@K
                dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
                idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(test_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f'ndcg@{k}'].append(ndcg)
                
                # Hit Ratio@K
                hit_ratio = 1.0 if sum(hits) > 0 else 0.0
                metrics[f'hit@{k}'].append(hit_ratio)
        
        # 평균 계산
        avg_metrics = {}
        for metric, values in metrics.items():
            avg_metrics[metric] = np.mean(values) if values else 0.0
            
        return avg_metrics
    
    def compare_methods(self, n_users=1000, k_list=[10, 20, 30, 50]):
        """세 가지 방법 비교"""
        # 테스트 사용자 샘플링
        test_users = list(self.data_loader.test_user_dict.keys())[:n_users]
        
        print(f"\n{n_users}명의 사용자에 대해 비교 중...")
        
        # 표준 추천
        print("표준 방식 추천 생성 중...")
        standard_recs = self.get_standard_recommendations(test_users, max(k_list))
        standard_metrics = self.evaluate_recommendations(
            standard_recs, self.data_loader.test_user_dict, k_list
        )
        
        # 단순 더하기 방식
        print("단순 더하기 방식 추천 생성 중...")
        simple_recs = self.get_relation_enhanced_recommendations_simple(test_users, max(k_list))
        simple_metrics = self.evaluate_recommendations(
            simple_recs, self.data_loader.test_user_dict, k_list
        )
        
        # TransR 방식
        print("TransR 방식 추천 생성 중...")
        transr_recs = self.get_relation_enhanced_recommendations_transr(test_users, max(k_list))
        transr_metrics = self.evaluate_recommendations(
            transr_recs, self.data_loader.test_user_dict, k_list
        )
        
        return {
            'standard': standard_metrics,
            'simple': simple_metrics,
            'transr': transr_metrics
        }
    
    def print_comparison(self, results):
        """비교 결과 출력"""
        print("\n" + "="*60)
        print("표준 vs 단순 더하기 vs TransR 추천 비교 결과")
        print("="*60)
        
        # 메트릭별 비교 테이블
        metrics = ['recall', 'precision', 'ndcg', 'hit']
        k_values = [10, 20, 30, 50]
        
        for metric in metrics:
            print(f"\n{metric.upper()} 비교:")
            
            headers = ['Method'] + [f'@{k}' for k in k_values]
            rows = []
            
            # 표준 방식
            std_row = ['Standard']
            for k in k_values:
                key = f'{metric}@{k}'
                if key in results['standard']:
                    std_row.append(f"{results['standard'][key]:.4f}")
                else:
                    std_row.append("N/A")
            rows.append(std_row)
            
            # 단순 더하기 방식
            simple_row = ['Simple Add']
            for k in k_values:
                key = f'{metric}@{k}'
                if key in results['simple']:
                    simple_row.append(f"{results['simple'][key]:.4f}")
                else:
                    simple_row.append("N/A")
            rows.append(simple_row)
            
            # TransR 방식
            transr_row = ['TransR']
            for k in k_values:
                key = f'{metric}@{k}'
                if key in results['transr']:
                    transr_row.append(f"{results['transr'][key]:.4f}")
                else:
                    transr_row.append("N/A")
            rows.append(transr_row)
            
            # 개선율
            for method_name, method_key in [('Simple Add', 'simple'), ('TransR', 'transr')]:
                imp_row = [f'{method_name} Improvement']
                for k in k_values:
                    key = f'{metric}@{k}'
                    if key in results['standard'] and key in results[method_key]:
                        std_val = results['standard'][key]
                        method_val = results[method_key][key]
                        if std_val > 0:
                            improvement = ((method_val - std_val) / std_val) * 100
                            imp_row.append(f"{improvement:+.1f}%")
                        else:
                            imp_row.append("N/A")
                    else:
                        imp_row.append("N/A")
                rows.append(imp_row)
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))


def main():
    parser = argparse.ArgumentParser(description='KGAT TransR 방식 비교')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 Original KGAT 모델 체크포인트')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='비교할 사용자 수')
    parser.add_argument('--output-dir', type=str, default='results/transr_comparison',
                        help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # Original 모델 설정
    class OriginalArgs:
        def __init__(self):
            self.dataset = 'amazon-book'
            self.data_path = 'data/'
            self.model_type = 'kgat'
            self.adj_type = 'si'
            self.alg_type = 'bi'
            self.embed_size = 64
            self.layer_size = [64, 32, 16]
            self.node_dropout = [0.1]
            self.mess_dropout = [0.1, 0.1, 0.1]
            self.regs = [1e-5, 1e-5]
            self.cf_batch_size = 1024
            self.kg_batch_size = 2048
            self.test_batch_size = 10000
            self.use_pretrain = 0
            self.pretrain_embedding_dir = 'pretrain/'
    
    original_args = OriginalArgs()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로드
    print("데이터 로딩 중...")
    data_loader = DataLoaderOriginal(original_args, None)
    
    # 모델 로드
    print("모델 로딩 중...")
    model = KGAT(
        original_args,
        data_loader.n_users,
        data_loader.n_items,
        data_loader.n_entities,
        data_loader.n_relations,
        data_loader.adjacency_dict['plain_adj'],
        data_loader.laplacian_dict['kg_mat']
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    print(f"모델 로드 완료: {args.checkpoint}")
    
    # 비교 실행
    comparison = RelationEnhancedComparisonTransR(model, data_loader, device)
    results = comparison.compare_methods(n_users=args.n_users)
    
    # 결과 출력
    comparison.print_comparison(results)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'n_users': args.n_users,
        'metrics': results
    }
    
    result_file = os.path.join(args.output_dir, f'transr_comparison_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {result_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()