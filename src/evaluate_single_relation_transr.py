"""
단일 관계 기반 추천 평가 (TransR 적용 버전)
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


class SingleRelationRecommenderTransR:
    """단일 관계 기반 추천 (TransR 적용)"""
    
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
    
    def get_single_relation_recommendations_transr(self, user_ids, relation_id, k=20, 
                                                  use_base_score=True, base_weight=0.5):
        """단일 관계만 사용한 추천 (TransR 적용)"""
        self.model.eval()
        
        print(f"\n관계 {relation_id}만을 사용한 추천 생성 중 (TransR)...")
        print(f"기본 점수 사용: {use_base_score}, 가중치: {base_weight}")
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            # 원본 임베딩 (관계 임베딩과 동일 차원)
            u_embed_base = self.model.user_embed
            i_embed_base = self.model.entity_embed[:self.data_loader.n_items]
            
            # TransR 변환 행렬
            W_r = self.model.trans_W[relation_id]  # [emb_size, emb_size]
            
            recommendations = {}
            relation_stats = {
                'users_with_paths': 0,
                'total_paths': 0,
                'avg_paths_per_user': 0
            }
            
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb_full = u_embed[u_original]  # 전체 임베딩 (concat된 것)
                user_emb_base = u_embed_base[u_original]  # 기본 임베딩
                
                # 사용자를 관계 공간으로 변환
                user_r = torch.matmul(user_emb_base, W_r)
                
                # 관계 기반 점수만 계산
                relation_scores = torch.zeros(self.data_loader.n_items).to(self.device)
                path_count = 0
                
                if u in self.data_loader.train_user_dict:
                    interacted_items = self.data_loader.train_user_dict[u]
                    
                    for item in interacted_items:
                        if item in self.data_loader.train_kg_dict:
                            # 상호작용한 아이템도 관계 공간으로 변환
                            item_emb = i_embed_base[item]
                            item_r = torch.matmul(item_emb, W_r)
                            
                            for target, r in self.data_loader.train_kg_dict[item]:
                                # 지정된 관계만 사용
                                if r == relation_id and target < self.data_loader.n_items:
                                    # relation 임베딩
                                    rel_emb = self.model.relation_embed[r]
                                    
                                    # 타겟 아이템도 관계 공간으로 변환
                                    target_emb = i_embed_base[target]
                                    target_r = torch.matmul(target_emb, W_r)
                                    
                                    # TransR: user_r + relation -> target_r
                                    enhanced_user_r = user_r + rel_emb
                                    score = torch.dot(enhanced_user_r, target_r)
                                    
                                    relation_scores[target] += score
                                    path_count += 1
                
                if path_count > 0:
                    relation_stats['users_with_paths'] += 1
                    relation_stats['total_paths'] += path_count
                
                # 최종 점수 계산
                if use_base_score:
                    # 전체 임베딩으로 기본 점수 계산
                    item_embed_full = i_embed[:self.data_loader.n_items]
                    base_scores = torch.matmul(user_emb_full, item_embed_full.t())
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
    
    def get_single_relation_recommendations_simple(self, user_ids, relation_id, k=20, 
                                                  use_base_score=True, base_weight=0.5):
        """단일 관계만 사용한 추천 (단순 더하기)"""
        self.model.eval()
        
        print(f"\n관계 {relation_id}만을 사용한 추천 생성 중 (단순 더하기)...")
        print(f"기본 점수 사용: {use_base_score}, 가중치: {base_weight}")
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            # 원본 임베딩
            u_embed_base = self.model.user_embed
            i_embed_base = self.model.entity_embed[:self.data_loader.n_items]
            
            recommendations = {}
            relation_stats = {
                'users_with_paths': 0,
                'total_paths': 0,
                'avg_paths_per_user': 0
            }
            
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb_full = u_embed[u_original]
                user_emb_base = u_embed_base[u_original]
                
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
                                    
                                    # 단순 더하기
                                    enhanced_user = user_emb_base + rel_emb
                                    
                                    # 타겟 아이템과의 유사도
                                    target_emb = i_embed_base[target]
                                    score = torch.dot(enhanced_user, target_emb)
                                    
                                    relation_scores[target] += score
                                    path_count += 1
                
                if path_count > 0:
                    relation_stats['users_with_paths'] += 1
                    relation_stats['total_paths'] += path_count
                
                # 최종 점수 계산
                if use_base_score:
                    item_embed_full = i_embed[:self.data_loader.n_items]
                    base_scores = torch.matmul(user_emb_full, item_embed_full.t())
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
    
    def get_standard_recommendations(self, user_ids, k=20):
        """표준 추천 (관계 미사용)"""
        self.model.eval()
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            recommendations = {}
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb = u_embed[u_original]
                
                # i_embed에서 아이템만 추출
                item_embed = i_embed[:self.data_loader.n_items]
                scores = torch.matmul(user_emb, item_embed.t())
                
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                _, topk_indices = torch.topk(scores, k)
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
    
    def compare_transr_vs_simple(self, n_users=1000, k_list=[10, 20, 30, 50]):
        """TransR vs 단순 더하기 비교"""
        test_users = list(self.data_loader.test_user_dict.keys())[:n_users]
        
        print(f"\n{n_users}명의 사용자에 대해 비교 중...")
        
        results = {}
        
        # 기준선: 표준 추천
        print("\n표준 추천 (기준선) 생성 중...")
        standard_recs = self.get_standard_recommendations(test_users, max(k_list))
        results['standard'] = {
            'metrics': self.evaluate_recommendations(
                standard_recs, self.data_loader.test_user_dict, k_list
            )
        }
        
        # 주요 관계들에 대해 비교
        top_relations = sorted(self.relation_info.items(), 
                             key=lambda x: x[1]['count'], reverse=True)[:3]
        
        for r, info in top_relations:
            if info['count'] < 100:  # 너무 적은 관계는 제외
                continue
            
            # 단순 더하기
            print(f"\n관계 {r} - 단순 더하기 평가 중...")
            simple_recs, simple_stats = self.get_single_relation_recommendations_simple(
                test_users, r, max(k_list), use_base_score=True, base_weight=0.5
            )
            
            results[f'relation_{r}_simple'] = {
                'metrics': self.evaluate_recommendations(
                    simple_recs, self.data_loader.test_user_dict, k_list
                ),
                'stats': simple_stats,
                'info': info
            }
            
            # TransR
            print(f"\n관계 {r} - TransR 평가 중...")
            transr_recs, transr_stats = self.get_single_relation_recommendations_transr(
                test_users, r, max(k_list), use_base_score=True, base_weight=0.5
            )
            
            results[f'relation_{r}_transr'] = {
                'metrics': self.evaluate_recommendations(
                    transr_recs, self.data_loader.test_user_dict, k_list
                ),
                'stats': transr_stats,
                'info': info
            }
        
        return results
    
    def print_comparison_results(self, results):
        """결과 출력"""
        print("\n" + "="*100)
        print("단일 관계 기반 추천: TransR vs 단순 더하기")
        print("="*100)
        
        # Recall@20 기준으로 정리
        headers = ['관계/방법', '연결수', '커버리지', 'Recall@20', 'Precision@20', 
                   'NDCG@20', '개선율', '경로있는사용자']
        rows = []
        
        standard_recall = results['standard']['metrics']['recall@20']
        
        # 표준 추천
        rows.append(['표준(기준)', '-', '-', f"{standard_recall:.4f}", 
                    f"{results['standard']['metrics']['precision@20']:.4f}",
                    f"{results['standard']['metrics']['ndcg@20']:.4f}",
                    '-', '-'])
        
        # 관계별 결과
        for key, data in sorted(results.items()):
            if key == 'standard':
                continue
                
            parts = key.split('_')
            if len(parts) >= 3:
                r = parts[1]
                method = parts[2]
                
                if 'metrics' in data:
                    metrics = data['metrics']
                    info = data['info']
                    stats = data.get('stats', {})
                    
                    recall = metrics.get('recall@20', 0)
                    improvement = ((recall - standard_recall) / standard_recall * 100) if standard_recall > 0 else 0
                    
                    row = [
                        f'관계{r}({method})',
                        f"{info['count']:,}",
                        f"{info['coverage']:.1%}",
                        f"{recall:.4f}",
                        f"{metrics.get('precision@20', 0):.4f}",
                        f"{metrics.get('ndcg@20', 0):.4f}",
                        f"{improvement:+.1f}%",
                        f"{stats.get('users_with_paths', 0)}"
                    ]
                    rows.append(row)
        
        print("\n성능 비교:")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # TransR vs Simple 직접 비교
        print("\n\nTransR vs 단순 더하기 직접 비교:")
        for r in range(40):  # 관계 ID 범위
            simple_key = f'relation_{r}_simple'
            transr_key = f'relation_{r}_transr'
            
            if simple_key in results and transr_key in results:
                simple_recall = results[simple_key]['metrics']['recall@20']
                transr_recall = results[transr_key]['metrics']['recall@20']
                
                if simple_recall > 0:
                    improvement = ((transr_recall - simple_recall) / simple_recall) * 100
                    print(f"  관계 {r}: TransR가 단순 더하기 대비 {improvement:+.1f}% {'개선' if improvement > 0 else '하락'}")


def main():
    parser = argparse.ArgumentParser(description='단일 관계 기반 KGAT 추천 (TransR 비교)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 Original KGAT 모델 체크포인트')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='평가할 사용자 수')
    parser.add_argument('--output-dir', type=str, default='results/single_relation_transr',
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
    recommender = SingleRelationRecommenderTransR(model, data_loader, device)
    results = recommender.compare_transr_vs_simple(n_users=args.n_users)
    
    # 결과 출력
    recommender.print_comparison_results(results)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'n_users': args.n_users,
        'results': {k: {
            'metrics': v['metrics'],
            'stats': v.get('stats', {}),
            'info': v.get('info', {})
        } for k, v in results.items()}
    }
    
    result_file = os.path.join(args.output_dir, f'single_relation_transr_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n결과가 {result_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()