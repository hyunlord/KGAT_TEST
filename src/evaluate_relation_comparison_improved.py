"""
KGAT 모델을 사용한 관계별 추천 비교 (개선된 버전)
- 관계 종류별로 다른 가중치 적용
- 특정 관계만 선택하여 비교 가능
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


class RelationAwareComparison:
    """관계 종류를 고려한 추천 비교"""
    
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # 관계별 통계 분석
        self.analyze_relations()
        
    def analyze_relations(self):
        """KG의 관계 종류별 통계 분석"""
        relation_counts = defaultdict(int)
        relation_item_pairs = defaultdict(set)
        
        for h, pairs in self.data_loader.train_kg_dict.items():
            if h < self.data_loader.n_items:  # head가 아이템인 경우
                for t, r in pairs:
                    if t < self.data_loader.n_items:  # tail도 아이템인 경우
                        relation_counts[r] += 1
                        relation_item_pairs[r].add((h, t))
        
        print("\n=== 관계 종류별 통계 ===")
        for r in sorted(relation_counts.keys()):
            count = relation_counts[r]
            print(f"관계 {r}: {count}개 연결 (아이템-아이템)")
        
        self.relation_stats = {
            'counts': dict(relation_counts),
            'item_pairs': {r: len(pairs) for r, pairs in relation_item_pairs.items()}
        }
        
        # 주요 관계 식별 (상위 5개)
        self.top_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n주요 관계 (상위 5개): {[r for r, _ in self.top_relations]}")
    
    def get_standard_recommendations(self, user_ids, k=20):
        """표준 방식: user-item 유사도만 사용"""
        self.model.eval()
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            # 원본 임베딩 (관계 임베딩과 동일 차원)
            u_embed_base = self.model.user_embed
            i_embed_base = self.model.entity_embed[:self.data_loader.n_items]
            
            recommendations = {}
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb_full = u_embed[u_original]  # 전체 임베딩
                user_emb_base = u_embed_base[u_original]  # 기본 임베딩
                
                # i_embed에서 아이템만 추출
                item_embed = i_embed[:self.data_loader.n_items]
                scores = torch.matmul(user_emb_full, item_embed.t())
                
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                _, topk_indices = torch.topk(scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations
    
    def get_relation_specific_recommendations(self, user_ids, k=20, 
                                            target_relations=None, 
                                            relation_weights=None):
        """특정 관계만 사용한 추천"""
        self.model.eval()
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            # 기본값 설정
            if target_relations is None:
                # 모든 관계 사용
                target_relations = list(range(self.data_loader.n_relations))
            
            if relation_weights is None:
                # 균등 가중치
                relation_weights = {r: 1.0 for r in target_relations}
            
            print(f"\n사용할 관계: {target_relations}")
            print(f"관계별 가중치: {relation_weights}")
            
            recommendations = {}
            
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb = u_embed[u_original]
                
                # 기본 점수
                base_scores = torch.matmul(user_emb, i_embed.t())
                
                # 관계별 강화 점수
                enhanced_scores = torch.zeros(self.data_loader.n_items).to(self.device)
                
                if u in self.data_loader.train_user_dict:
                    interacted_items = self.data_loader.train_user_dict[u]
                    
                    for item in interacted_items:
                        if item in self.data_loader.train_kg_dict:
                            for target, relation in self.data_loader.train_kg_dict[item]:
                                # 특정 관계만 고려
                                if relation in target_relations and target < self.data_loader.n_items:
                                    rel_emb = self.model.relation_embed[relation]
                                    
                                    # 관계별 가중치 적용
                                    weight = relation_weights.get(relation, 1.0)
                                    enhanced_user = user_emb_base + weight * rel_emb  # 같은 차원
                                    
                                    target_emb = i_embed_base[target]  # 기본 임베딩 사용
                                    score = torch.dot(enhanced_user, target_emb)
                                    
                                    enhanced_scores[target] += score
                
                # 최종 점수 결합
                # i_embed에서 아이템만 추출
                item_embed = i_embed[:self.data_loader.n_items]
                base_scores = torch.matmul(user_emb_full, item_embed.t())
                final_scores = base_scores + 0.3 * enhanced_scores
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    final_scores[train_items] = -float('inf')
                
                _, topk_indices = torch.topk(final_scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations
    
    def get_adaptive_relation_recommendations(self, user_ids, k=20):
        """적응적 관계 가중치를 사용한 추천"""
        self.model.eval()
        
        # 관계별 가중치 학습 (빈도 기반)
        relation_importance = {}
        total_count = sum(self.relation_stats['counts'].values())
        
        for r, count in self.relation_stats['counts'].items():
            # 빈도가 높은 관계는 더 일반적이므로 낮은 가중치
            # 빈도가 낮은 관계는 더 특별하므로 높은 가중치
            importance = np.log(total_count / (count + 1))
            relation_importance[r] = importance * 0.1  # 스케일 조정
        
        print("\n적응적 관계 가중치:")
        for r in sorted(relation_importance.keys())[:10]:
            print(f"  관계 {r}: {relation_importance[r]:.4f}")
        
        return self.get_relation_specific_recommendations(
            user_ids, k, 
            relation_weights=relation_importance
        )
    
    def compare_relation_strategies(self, n_users=1000, k_list=[10, 20, 30, 50]):
        """다양한 관계 전략 비교"""
        test_users = list(self.data_loader.test_user_dict.keys())[:n_users]
        
        print(f"\n{n_users}명의 사용자에 대해 비교 중...")
        
        results = {}
        
        # 1. 표준 방식
        print("\n1. 표준 방식 (관계 미사용)")
        standard_recs = self.get_standard_recommendations(test_users, max(k_list))
        results['standard'] = self.evaluate_recommendations(
            standard_recs, self.data_loader.test_user_dict, k_list
        )
        
        # 2. 모든 관계 균등 사용
        print("\n2. 모든 관계 균등 가중치")
        all_equal_recs = self.get_relation_specific_recommendations(
            test_users, max(k_list)
        )
        results['all_equal'] = self.evaluate_recommendations(
            all_equal_recs, self.data_loader.test_user_dict, k_list
        )
        
        # 3. 적응적 가중치
        print("\n3. 적응적 관계 가중치")
        adaptive_recs = self.get_adaptive_relation_recommendations(
            test_users, max(k_list)
        )
        results['adaptive'] = self.evaluate_recommendations(
            adaptive_recs, self.data_loader.test_user_dict, k_list
        )
        
        # 4. 주요 관계만 사용
        if self.top_relations:
            print(f"\n4. 주요 관계만 사용 (상위 3개)")
            top_3_relations = [r for r, _ in self.top_relations[:3]]
            top_relation_recs = self.get_relation_specific_recommendations(
                test_users, max(k_list),
                target_relations=top_3_relations
            )
            results['top_3_relations'] = self.evaluate_recommendations(
                top_relation_recs, self.data_loader.test_user_dict, k_list
            )
        
        # 5. 개별 관계별 성능
        print("\n5. 개별 관계별 성능 분석")
        for r, count in self.top_relations[:3]:
            print(f"\n  관계 {r}만 사용 (연결 수: {count})")
            single_rel_recs = self.get_relation_specific_recommendations(
                test_users, max(k_list),
                target_relations=[r]
            )
            results[f'relation_{r}'] = self.evaluate_recommendations(
                single_rel_recs, self.data_loader.test_user_dict, k_list
            )
        
        return results
    
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
                
                recall = sum(hits) / len(test_items)
                metrics[f'recall@{k}'].append(recall)
                
                precision = sum(hits) / k
                metrics[f'precision@{k}'].append(precision)
                
                dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
                idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(test_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f'ndcg@{k}'].append(ndcg)
                
                hit_ratio = 1.0 if sum(hits) > 0 else 0.0
                metrics[f'hit@{k}'].append(hit_ratio)
        
        avg_metrics = {}
        for metric, values in metrics.items():
            avg_metrics[metric] = np.mean(values) if values else 0.0
            
        return avg_metrics
    
    def print_comparison_results(self, results):
        """비교 결과 출력"""
        print("\n" + "="*80)
        print("관계 전략별 추천 성능 비교")
        print("="*80)
        
        # 전략 이름 매핑
        strategy_names = {
            'standard': '표준 (관계 미사용)',
            'all_equal': '모든 관계 균등',
            'adaptive': '적응적 가중치',
            'top_3_relations': '주요 관계 3개'
        }
        
        # 관계별 결과도 포함
        for key in results:
            if key.startswith('relation_'):
                r = key.split('_')[1]
                strategy_names[key] = f'관계 {r}만'
        
        metrics = ['recall', 'precision', 'ndcg', 'hit']
        k_values = [10, 20, 30, 50]
        
        for metric in metrics:
            print(f"\n{metric.upper()} 비교:")
            
            headers = ['전략'] + [f'@{k}' for k in k_values]
            rows = []
            
            # 표준 방식을 기준으로
            standard_values = {}
            for k in k_values:
                key = f'{metric}@{k}'
                if key in results.get('standard', {}):
                    standard_values[k] = results['standard'][key]
            
            for strategy_key, strategy_name in strategy_names.items():
                if strategy_key not in results:
                    continue
                    
                row = [strategy_name]
                for k in k_values:
                    key = f'{metric}@{k}'
                    if key in results[strategy_key]:
                        value = results[strategy_key][key]
                        
                        # 표준 대비 개선율 계산
                        if strategy_key != 'standard' and k in standard_values:
                            improvement = ((value - standard_values[k]) / standard_values[k]) * 100
                            row.append(f"{value:.4f} ({improvement:+.1f}%)")
                        else:
                            row.append(f"{value:.4f}")
                    else:
                        row.append("N/A")
                rows.append(row)
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    def visualize_relation_impact(self, results, output_dir):
        """관계별 영향력 시각화"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 관계별 Recall@20 성능
        relation_performance = {}
        for key, metrics in results.items():
            if key.startswith('relation_'):
                r = key.split('_')[1]
                relation_performance[f'관계 {r}'] = metrics.get('recall@20', 0)
        
        if relation_performance:
            plt.figure(figsize=(10, 6))
            relations = list(relation_performance.keys())
            performances = list(relation_performance.values())
            
            plt.bar(relations, performances)
            plt.axhline(y=results['standard']['recall@20'], color='r', linestyle='--', 
                       label='표준 방식')
            plt.xlabel('관계 종류')
            plt.ylabel('Recall@20')
            plt.title('관계별 추천 성능')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'relation_impact.png'))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='KGAT 관계별 추천 비교 (개선된 버전)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 Original KGAT 모델 체크포인트')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='비교할 사용자 수')
    parser.add_argument('--output-dir', type=str, default='results/relation_analysis',
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
    print(f"관계 종류 수: {data_loader.n_relations}")
    
    # 비교 실행
    comparison = RelationAwareComparison(model, data_loader, device)
    results = comparison.compare_relation_strategies(n_users=args.n_users)
    
    # 결과 출력
    comparison.print_comparison_results(results)
    
    # 시각화
    comparison.visualize_relation_impact(results, args.output_dir)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'n_users': args.n_users,
        'n_relations': data_loader.n_relations,
        'relation_stats': comparison.relation_stats,
        'results': results
    }
    
    result_file = os.path.join(args.output_dir, f'relation_analysis_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {result_file}에 저장되었습니다.")
    print(f"시각화 결과는 {args.output_dir}/relation_impact.png에서 확인하세요.")


if __name__ == "__main__":
    main()