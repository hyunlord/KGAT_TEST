"""
단일 관계 기반 추천 평가
특정 관계 하나만을 사용하여 타겟팅된 추천 생성
"""

import os
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from tabulate import tabulate
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from kgat_original import KGAT
from data_loader_original import DataLoaderOriginal


class SingleRelationRecommender:
    """단일 관계 기반 추천"""
    
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # 관계별 통계 분석
        self.analyze_relations()
    
    def analyze_relations(self):
        """각 관계의 특성 분석"""
        self.relation_info = defaultdict(lambda: {
            'count': 0,
            'unique_items': set(),
            'item_pairs': [],
            'coverage': 0.0
        })
        
        # 아이템-아이템 관계만 분석
        for h, pairs in self.data_loader.train_kg_dict.items():
            if h < self.data_loader.n_items:
                for t, r in pairs:
                    if t < self.data_loader.n_items:
                        self.relation_info[r]['count'] += 1
                        self.relation_info[r]['unique_items'].add(h)
                        self.relation_info[r]['unique_items'].add(t)
                        self.relation_info[r]['item_pairs'].append((h, t))
        
        # 관계별 커버리지 계산
        total_items = self.data_loader.n_items
        for r, info in self.relation_info.items():
            info['coverage'] = len(info['unique_items']) / total_items
            info['unique_items'] = len(info['unique_items'])  # set을 크기로 변환
        
        print("\n=== 관계별 특성 분석 ===")
        print(f"{'관계ID':>6} | {'연결수':>8} | {'고유아이템':>10} | {'커버리지':>8}")
        print("-" * 45)
        
        for r in sorted(self.relation_info.keys()):
            info = self.relation_info[r]
            print(f"{r:>6} | {info['count']:>8} | {info['unique_items']:>10} | {info['coverage']:>7.2%}")
    
    def get_single_relation_recommendations(self, user_ids, relation_id, k=20, 
                                          use_base_score=True, base_weight=0.7):
        """단일 관계만 사용한 추천
        
        Args:
            user_ids: 추천할 사용자 리스트
            relation_id: 사용할 관계 ID
            k: 추천 아이템 수
            use_base_score: 기본 유사도 점수 포함 여부
            base_weight: 기본 점수의 가중치 (0~1)
        """
        self.model.eval()
        
        print(f"\n관계 {relation_id}만을 사용한 추천 생성 중...")
        print(f"기본 점수 사용: {use_base_score}, 가중치: {base_weight}")
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            recommendations = {}
            relation_stats = {
                'users_with_paths': 0,
                'total_paths': 0,
                'avg_paths_per_user': 0
            }
            
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb = u_embed[u_original]
                
                # 관계 기반 점수만 계산
                relation_scores = torch.zeros(self.data_loader.n_items).to(self.device)
                path_count = 0
                
                if u in self.data_loader.train_user_dict:
                    interacted_items = self.data_loader.train_user_dict[u]
                    
                    for item in interacted_items:
                        if item in self.data_loader.train_kg_dict:
                            for target, r in self.data_loader.train_kg_dict[item]:
                                # 지정된 관계만 사용
                                if r == relation_id and target < self.data_loader.n_items:
                                    # relation 임베딩
                                    rel_emb = self.model.relation_embed[r]
                                    
                                    # user + relation 결합
                                    enhanced_user = user_emb + rel_emb
                                    
                                    # 타겟 아이템과의 유사도
                                    target_emb = i_embed[target]
                                    score = torch.dot(enhanced_user, target_emb)
                                    
                                    relation_scores[target] += score
                                    path_count += 1
                
                if path_count > 0:
                    relation_stats['users_with_paths'] += 1
                    relation_stats['total_paths'] += path_count
                
                # 최종 점수 계산
                if use_base_score:
                    base_scores = torch.matmul(user_emb, i_embed.t())
                    final_scores = base_weight * base_scores + (1 - base_weight) * relation_scores
                else:
                    final_scores = relation_scores
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    final_scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(final_scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
            
            # 통계 계산
            if relation_stats['users_with_paths'] > 0:
                relation_stats['avg_paths_per_user'] = (
                    relation_stats['total_paths'] / relation_stats['users_with_paths']
                )
            
            print(f"\n관계 {relation_id} 사용 통계:")
            print(f"  - 경로가 있는 사용자: {relation_stats['users_with_paths']}/{len(user_ids)}")
            print(f"  - 평균 경로 수: {relation_stats['avg_paths_per_user']:.2f}")
            
        return recommendations, relation_stats
    
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
                
                if len(test_items) > 0:
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
    
    def compare_all_relations(self, n_users=1000, k_list=[10, 20, 30, 50]):
        """모든 관계를 개별적으로 비교"""
        test_users = list(self.data_loader.test_user_dict.keys())[:n_users]
        
        print(f"\n{n_users}명의 사용자에 대해 모든 관계 비교 중...")
        
        results = {}
        
        # 기준선: 표준 추천 (관계 미사용)
        print("\n표준 추천 (기준선) 생성 중...")
        standard_recs = self.get_standard_recommendations(test_users, max(k_list))
        results['standard'] = {
            'metrics': self.evaluate_recommendations(
                standard_recs, self.data_loader.test_user_dict, k_list
            ),
            'stats': {'users_with_paths': len(test_users)}
        }
        
        # 각 관계별 추천
        for r in sorted(self.relation_info.keys()):
            if self.relation_info[r]['count'] < 100:  # 너무 적은 관계는 제외
                continue
                
            print(f"\n관계 {r} 평가 중...")
            recs, stats = self.get_single_relation_recommendations(
                test_users, r, max(k_list), 
                use_base_score=True, base_weight=0.5
            )
            
            results[f'relation_{r}'] = {
                'metrics': self.evaluate_recommendations(
                    recs, self.data_loader.test_user_dict, k_list
                ),
                'stats': stats,
                'info': self.relation_info[r]
            }
        
        # 순수 관계만 사용 (기본 점수 제외)
        print("\n주요 관계들의 순수 성능 평가 중...")
        top_relations = sorted(self.relation_info.items(), 
                             key=lambda x: x[1]['count'], reverse=True)[:3]
        
        for r, _ in top_relations:
            print(f"\n관계 {r} (순수) 평가 중...")
            recs, stats = self.get_single_relation_recommendations(
                test_users, r, max(k_list), 
                use_base_score=False
            )
            
            results[f'pure_relation_{r}'] = {
                'metrics': self.evaluate_recommendations(
                    recs, self.data_loader.test_user_dict, k_list
                ),
                'stats': stats,
                'info': self.relation_info[r]
            }
        
        return results
    
    def get_standard_recommendations(self, user_ids, k=20):
        """표준 추천 (관계 미사용)"""
        self.model.eval()
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            recommendations = {}
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb = u_embed[u_original]
                
                scores = torch.matmul(user_emb, i_embed.t())
                
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                _, topk_indices = torch.topk(scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations
    
    def print_comparison_results(self, results):
        """결과 출력"""
        print("\n" + "="*100)
        print("단일 관계 기반 추천 성능 비교")
        print("="*100)
        
        # Recall@20 기준으로 정렬
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['metrics'].get('recall@20', 0), 
                              reverse=True)
        
        # 성능 테이블
        headers = ['관계', '연결수', '커버리지', 'Recall@20', 'Precision@20', 
                   'NDCG@20', '개선율', '경로있는사용자']
        rows = []
        
        standard_recall = results['standard']['metrics']['recall@20']
        
        for key, data in sorted_results:
            if key == 'standard':
                row = ['표준(기준)', '-', '-']
            elif key.startswith('pure_relation_'):
                r = key.split('_')[-1]
                info = data['info']
                row = [f'관계{r}(순수)', f"{info['count']:,}", f"{info['coverage']:.1%}"]
            elif key.startswith('relation_'):
                r = key.split('_')[-1]
                info = data['info']
                row = [f'관계{r}(혼합)', f"{info['count']:,}", f"{info['coverage']:.1%}"]
            else:
                continue
            
            metrics = data['metrics']
            recall = metrics.get('recall@20', 0)
            precision = metrics.get('precision@20', 0)
            ndcg = metrics.get('ndcg@20', 0)
            
            improvement = ((recall - standard_recall) / standard_recall * 100) if standard_recall > 0 else 0
            
            row.extend([
                f"{recall:.4f}",
                f"{precision:.4f}",
                f"{ndcg:.4f}",
                f"{improvement:+.1f}%",
                f"{data['stats'].get('users_with_paths', 0)}"
            ])
            
            rows.append(row)
        
        print("\n성능 비교:")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # 관계별 특성과 성능 상관관계
        print("\n\n관계 특성과 성능의 상관관계:")
        relation_data = []
        for key, data in results.items():
            if key.startswith('relation_') and not key.startswith('pure_'):
                r = key.split('_')[-1]
                relation_data.append({
                    '관계ID': r,
                    '연결수': data['info']['count'],
                    '커버리지': data['info']['coverage'],
                    'Recall@20': data['metrics']['recall@20']
                })
        
        if relation_data:
            # 연결수와 성능의 상관관계
            counts = [d['연결수'] for d in relation_data]
            recalls = [d['Recall@20'] for d in relation_data]
            
            if len(counts) > 2:
                correlation = np.corrcoef(counts, recalls)[0, 1]
                print(f"  - 연결수와 Recall@20의 상관계수: {correlation:.3f}")
    
    def visualize_results(self, results, output_dir):
        """결과 시각화"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 관계별 성능 막대 그래프
        plt.figure(figsize=(12, 8))
        
        relation_names = []
        recalls = []
        colors = []
        
        for key, data in sorted(results.items(), 
                               key=lambda x: x[1]['metrics'].get('recall@20', 0), 
                               reverse=True):
            if key == 'standard':
                relation_names.append('표준')
                colors.append('red')
            elif key.startswith('pure_relation_'):
                r = key.split('_')[-1]
                relation_names.append(f'관계{r}(순수)')
                colors.append('orange')
            elif key.startswith('relation_'):
                r = key.split('_')[-1]
                relation_names.append(f'관계{r}')
                colors.append('blue')
            else:
                continue
            
            recalls.append(data['metrics'].get('recall@20', 0))
        
        plt.bar(range(len(relation_names)), recalls, color=colors)
        plt.xticks(range(len(relation_names)), relation_names, rotation=45)
        plt.ylabel('Recall@20')
        plt.title('단일 관계 기반 추천 성능')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'single_relation_performance.png'))
        plt.close()
        
        # 2. 관계 특성 vs 성능 산점도
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 연결수 vs 성능
        counts = []
        recalls = []
        labels = []
        
        for key, data in results.items():
            if key.startswith('relation_') and not key.startswith('pure_'):
                r = key.split('_')[-1]
                counts.append(data['info']['count'])
                recalls.append(data['metrics']['recall@20'])
                labels.append(f'R{r}')
        
        ax1.scatter(counts, recalls)
        for i, label in enumerate(labels):
            ax1.annotate(label, (counts[i], recalls[i]))
        ax1.set_xlabel('관계 연결 수')
        ax1.set_ylabel('Recall@20')
        ax1.set_title('관계 연결 수와 성능')
        ax1.set_xscale('log')
        
        # 커버리지 vs 성능
        coverages = []
        recalls = []
        
        for key, data in results.items():
            if key.startswith('relation_') and not key.startswith('pure_'):
                coverages.append(data['info']['coverage'])
                recalls.append(data['metrics']['recall@20'])
        
        ax2.scatter(coverages, recalls)
        for i, label in enumerate(labels):
            ax2.annotate(label, (coverages[i], recalls[i]))
        ax2.set_xlabel('아이템 커버리지')
        ax2.set_ylabel('Recall@20')
        ax2.set_title('아이템 커버리지와 성능')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relation_characteristics.png'))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='단일 관계 기반 KGAT 추천 평가')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 Original KGAT 모델 체크포인트')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='평가할 사용자 수')
    parser.add_argument('--target-relation', type=int, default=None,
                        help='특정 관계만 평가 (None이면 모든 관계)')
    parser.add_argument('--output-dir', type=str, default='results/single_relation',
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
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    print(f"모델 로드 완료: {args.checkpoint}")
    
    # 추천 및 평가
    recommender = SingleRelationRecommender(model, data_loader, device)
    
    if args.target_relation is not None:
        # 특정 관계만 평가
        print(f"\n관계 {args.target_relation}만 평가합니다.")
        test_users = list(data_loader.test_user_dict.keys())[:args.n_users]
        
        recs, stats = recommender.get_single_relation_recommendations(
            test_users, args.target_relation, k=50,
            use_base_score=True, base_weight=0.5
        )
        
        metrics = recommender.evaluate_recommendations(
            recs, data_loader.test_user_dict, [10, 20, 30, 50]
        )
        
        print(f"\n관계 {args.target_relation} 성능:")
        for k in [10, 20, 30, 50]:
            print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    else:
        # 모든 관계 비교
        results = recommender.compare_all_relations(n_users=args.n_users)
        
        # 결과 출력
        recommender.print_comparison_results(results)
        
        # 시각화
        recommender.visualize_results(results, args.output_dir)
        
        # 결과 저장
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_data = {
            'timestamp': timestamp,
            'checkpoint': args.checkpoint,
            'n_users': args.n_users,
            'n_relations': data_loader.n_relations,
            'results': results
        }
        
        result_file = os.path.join(args.output_dir, f'single_relation_{timestamp}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n결과가 {result_file}에 저장되었습니다.")
        print(f"시각화는 {args.output_dir}/ 디렉토리에서 확인하세요.")


if __name__ == "__main__":
    main()