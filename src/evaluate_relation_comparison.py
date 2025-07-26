"""
KGAT 모델을 사용한 표준 추천 vs 관계 강화 추천 비교
- 표준: user 임베딩과 item 임베딩의 유사도만 사용
- 관계 강화: user + relation 임베딩과 item 임베딩의 유사도 사용
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


class RelationEnhancedComparison:
    """표준 추천과 관계 강화 추천 비교"""
    
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
                scores = torch.matmul(user_emb, i_embed.t())
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations
    
    def get_relation_enhanced_recommendations(self, user_ids, k=20):
        """관계 강화 방식: user + relation 임베딩을 사용"""
        self.model.eval()
        
        with torch.no_grad():
            # 전체 임베딩 계산
            u_embed, i_embed = self.model()
            
            # KG에서 관계 정보 활용
            recommendations = {}
            
            for u in user_ids:
                # 사용자 ID 변환
                u_original = u - self.data_loader.n_entities
                
                # 기본 사용자 임베딩
                user_emb = u_embed[u_original]
                
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
                                    
                                    # user + relation 임베딩
                                    enhanced_user = user_emb + 0.1 * rel_emb  # 가중치 조절 가능
                                    
                                    # 타겟 아이템과의 유사도
                                    target_emb = i_embed[target]
                                    score = torch.dot(enhanced_user, target_emb)
                                    
                                    # 점수 누적
                                    enhanced_scores[target] += score
                
                # 기본 점수와 관계 강화 점수 결합
                base_scores = torch.matmul(user_emb, i_embed.t())
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
        """두 방법 비교"""
        # 테스트 사용자 샘플링
        test_users = list(self.data_loader.test_user_dict.keys())[:n_users]
        
        print(f"\n{n_users}명의 사용자에 대해 비교 중...")
        
        # 표준 추천
        print("표준 방식 추천 생성 중...")
        standard_recs = self.get_standard_recommendations(test_users, max(k_list))
        standard_metrics = self.evaluate_recommendations(
            standard_recs, self.data_loader.test_user_dict, k_list
        )
        
        # 관계 강화 추천
        print("관계 강화 방식 추천 생성 중...")
        enhanced_recs = self.get_relation_enhanced_recommendations(test_users, max(k_list))
        enhanced_metrics = self.evaluate_recommendations(
            enhanced_recs, self.data_loader.test_user_dict, k_list
        )
        
        return {
            'standard': standard_metrics,
            'enhanced': enhanced_metrics,
            'standard_recs': standard_recs,
            'enhanced_recs': enhanced_recs
        }
    
    def print_comparison(self, results):
        """비교 결과 출력"""
        print("\n" + "="*60)
        print("표준 vs 관계 강화 추천 비교 결과")
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
            
            # 관계 강화 방식
            enh_row = ['Enhanced']
            for k in k_values:
                key = f'{metric}@{k}'
                if key in results['enhanced']:
                    enh_row.append(f"{results['enhanced'][key]:.4f}")
                else:
                    enh_row.append("N/A")
            rows.append(enh_row)
            
            # 개선율
            imp_row = ['Improvement']
            for k in k_values:
                key = f'{metric}@{k}'
                if key in results['standard'] and key in results['enhanced']:
                    std_val = results['standard'][key]
                    enh_val = results['enhanced'][key]
                    if std_val > 0:
                        improvement = ((enh_val - std_val) / std_val) * 100
                        imp_row.append(f"{improvement:+.1f}%")
                    else:
                        imp_row.append("N/A")
                else:
                    imp_row.append("N/A")
            rows.append(imp_row)
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    def analyze_sample_users(self, results, n_samples=10):
        """샘플 사용자 분석"""
        print(f"\n\n샘플 사용자 분석 ({n_samples}명)")
        print("="*60)
        
        standard_recs = results['standard_recs']
        enhanced_recs = results['enhanced_recs']
        
        sample_users = list(standard_recs.keys())[:n_samples]
        
        for u in sample_users:
            print(f"\n사용자 {u - self.data_loader.n_entities}:")
            
            # 테스트 아이템
            test_items = self.data_loader.test_user_dict.get(u, [])
            print(f"  실제 선호 아이템: {test_items[:5]}{'...' if len(test_items) > 5 else ''}")
            
            # 표준 추천
            std_rec = standard_recs[u][:10]
            std_hits = [item for item in std_rec if item in test_items]
            print(f"  표준 추천 (hits={len(std_hits)}): {std_rec}")
            
            # 관계 강화 추천
            enh_rec = enhanced_recs[u][:10]
            enh_hits = [item for item in enh_rec if item in test_items]
            print(f"  관계 강화 추천 (hits={len(enh_hits)}): {enh_rec}")
            
            # 추천 차이
            overlap = len(set(std_rec) & set(enh_rec))
            print(f"  추천 겹침: {overlap}/10 아이템")
            
            if len(enh_hits) > len(std_hits):
                print(f"  ✓ 관계 강화가 더 나은 성능 (+{len(enh_hits) - len(std_hits)} hits)")
            elif len(enh_hits) < len(std_hits):
                print(f"  ✗ 표준 방식이 더 나은 성능 (+{len(std_hits) - len(enh_hits)} hits)")
            else:
                print(f"  = 동일한 성능")


def main():
    parser = argparse.ArgumentParser(description='KGAT 표준 vs 관계 강화 추천 비교')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 Original KGAT 모델 체크포인트')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='비교할 사용자 수')
    parser.add_argument('--output-dir', type=str, default='results/relation_comparison',
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
    comparison = RelationEnhancedComparison(model, data_loader, device)
    results = comparison.compare_methods(n_users=args.n_users)
    
    # 결과 출력
    comparison.print_comparison(results)
    
    # 샘플 사용자 분석
    comparison.analyze_sample_users(results, n_samples=20)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'n_users': args.n_users,
        'metrics': {
            'standard': results['standard'],
            'enhanced': results['enhanced']
        }
    }
    
    result_file = os.path.join(args.output_dir, f'comparison_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {result_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()