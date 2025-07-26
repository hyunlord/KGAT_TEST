"""
KGAT 모델의 올바른 평가 방식
- 표준: 학습된 임베딩으로 사용자-아이템 추천
- 분석: KG가 임베딩 학습에 미친 영향 분석
"""

import os
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from tabulate import tabulate
from collections import defaultdict

from kgat_original import KGAT
from data_loader_original import DataLoaderOriginal


class KGATCorrectEvaluation:
    """KGAT의 올바른 평가"""
    
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
    def get_recommendations(self, user_ids, k=20):
        """표준 KGAT 추천: 학습된 임베딩 사용"""
        self.model.eval()
        
        with torch.no_grad():
            # 전체 임베딩 계산 (KG 정보가 이미 반영됨)
            u_embed, i_embed = self.model()
            
            recommendations = {}
            scores_dict = {}
            
            for u in user_ids:
                # 사용자 ID 변환
                u_original = u - self.data_loader.n_entities
                
                # 사용자 임베딩
                user_emb = u_embed[u_original]
                
                # 모든 아이템과의 점수 계산
                item_embed = i_embed[:self.data_loader.n_items]
                scores = torch.matmul(user_emb, item_embed.t())
                
                # 점수 저장 (분석용)
                scores_dict[u] = scores.cpu().numpy()
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
                
        return recommendations, scores_dict
    
    def analyze_kg_influence(self, user_ids, k=20):
        """KG가 추천에 미치는 영향 분석"""
        self.model.eval()
        
        with torch.no_grad():
            # 전체 임베딩
            u_embed, i_embed = self.model()
            
            # 분석 결과 저장
            analysis = {
                'kg_connected_items': {},
                'recommendation_overlap': {},
                'kg_item_scores': {}
            }
            
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb = u_embed[u_original]
                
                # KG로 연결된 아이템 찾기
                kg_connected = set()
                if u in self.data_loader.train_user_dict:
                    for item in self.data_loader.train_user_dict[u]:
                        if item in self.data_loader.train_kg_dict:
                            for target, _ in self.data_loader.train_kg_dict[item]:
                                if target < self.data_loader.n_items:
                                    kg_connected.add(target)
                
                analysis['kg_connected_items'][u] = list(kg_connected)
                
                # 추천 점수 계산
                item_embed = i_embed[:self.data_loader.n_items]
                scores = torch.matmul(user_emb, item_embed.t())
                
                # KG 연결 아이템들의 평균 점수
                if kg_connected:
                    kg_scores = scores[list(kg_connected)].mean().item()
                    non_kg_scores = scores[~torch.tensor([i in kg_connected for i in range(len(scores))])].mean().item()
                    analysis['kg_item_scores'][u] = {
                        'kg_connected': kg_scores,
                        'non_kg_connected': non_kg_scores,
                        'difference': kg_scores - non_kg_scores
                    }
                
                # Top-K 추천에서 KG 연결 아이템 비율
                scores_copy = scores.clone()
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores_copy[train_items] = -float('inf')
                
                _, topk_indices = torch.topk(scores_copy, k)
                topk_items = topk_indices.cpu().numpy().tolist()
                
                kg_in_topk = sum(1 for item in topk_items if item in kg_connected)
                analysis['recommendation_overlap'][u] = {
                    'total_recommendations': k,
                    'kg_connected_in_topk': kg_in_topk,
                    'percentage': (kg_in_topk / k) * 100
                }
        
        return analysis
    
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
    
    def print_analysis(self, analysis, n_samples=10):
        """KG 영향 분석 결과 출력"""
        print("\n" + "="*60)
        print("KG가 추천에 미치는 영향 분석")
        print("="*60)
        
        # 전체 통계
        total_users = len(analysis['recommendation_overlap'])
        avg_kg_percentage = np.mean([
            data['percentage'] 
            for data in analysis['recommendation_overlap'].values()
        ])
        
        print(f"\n전체 통계:")
        print(f"  - 분석 사용자 수: {total_users}")
        print(f"  - Top-20 추천 중 KG 연결 아이템 평균 비율: {avg_kg_percentage:.1f}%")
        
        # KG 연결 아이템 점수 분석
        if analysis['kg_item_scores']:
            avg_diff = np.mean([
                data['difference'] 
                for data in analysis['kg_item_scores'].values()
            ])
            print(f"  - KG 연결 아이템의 평균 점수 차이: {avg_diff:.4f}")
        
        # 샘플 사용자 상세 분석
        print(f"\n\n샘플 사용자 분석 ({n_samples}명):")
        print("-"*60)
        
        sample_users = list(analysis['kg_connected_items'].keys())[:n_samples]
        
        for u in sample_users:
            u_display = u - self.data_loader.n_entities
            print(f"\n사용자 {u_display}:")
            
            # KG 연결 아이템 수
            kg_items = analysis['kg_connected_items'][u]
            print(f"  - KG로 연결된 아이템 수: {len(kg_items)}")
            
            # 추천에서의 KG 아이템 비율
            overlap = analysis['recommendation_overlap'][u]
            print(f"  - Top-20 추천 중 KG 연결 아이템: {overlap['kg_connected_in_topk']}/20 ({overlap['percentage']:.1f}%)")
            
            # 점수 차이
            if u in analysis['kg_item_scores']:
                scores = analysis['kg_item_scores'][u]
                print(f"  - KG 연결 아이템 평균 점수: {scores['kg_connected']:.4f}")
                print(f"  - 비연결 아이템 평균 점수: {scores['non_kg_connected']:.4f}")
                print(f"  - 점수 차이: {scores['difference']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='KGAT 올바른 평가 및 분석')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 KGAT 모델 체크포인트')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='평가할 사용자 수')
    parser.add_argument('--output-dir', type=str, default='results/kgat_analysis',
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
    
    # 평가 실행
    evaluator = KGATCorrectEvaluation(model, data_loader, device)
    
    # 테스트 사용자
    test_users = list(data_loader.test_user_dict.keys())[:args.n_users]
    
    # 1. 표준 추천 및 평가
    print("\nKGAT 추천 생성 중...")
    recommendations, scores = evaluator.get_recommendations(test_users, k=50)
    
    metrics = evaluator.evaluate_recommendations(
        recommendations, data_loader.test_user_dict, [10, 20, 30, 50]
    )
    
    print("\n추천 성능:")
    for k in [10, 20, 30, 50]:
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
        print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
        print(f"  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    
    # 2. KG 영향 분석
    print("\nKG 영향 분석 중...")
    analysis = evaluator.analyze_kg_influence(test_users, k=20)
    evaluator.print_analysis(analysis)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'n_users': args.n_users,
        'metrics': metrics,
        'kg_analysis': {
            'avg_kg_percentage_in_topk': np.mean([
                data['percentage'] 
                for data in analysis['recommendation_overlap'].values()
            ]),
            'total_users_analyzed': len(analysis['recommendation_overlap'])
        }
    }
    
    result_file = os.path.join(args.output_dir, f'kgat_analysis_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {result_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()