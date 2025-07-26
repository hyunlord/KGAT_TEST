"""
구매 관계(relation 0)를 직접 사용한 추천 평가
user + purchase_relation → item 방식으로 추천
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


class PurchaseRelationRecommender:
    """구매 관계 기반 직접 추천"""
    
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # 구매 관계 ID 확인
        self.purchase_relation_id = 0  # user → item
        self.inverse_purchase_relation_id = 1  # item → user
        
        print(f"\n=== 관계 정보 ===")
        print(f"전체 관계 수: {data_loader.n_relations}")
        print(f"구매 관계 ID: {self.purchase_relation_id}")
        print(f"역구매 관계 ID: {self.inverse_purchase_relation_id}")
    
    def get_purchase_relation_recommendations_transr(self, user_ids, k=20):
        """구매 관계를 사용한 직접 추천 (TransR 적용)"""
        self.model.eval()
        
        print(f"\n구매 관계 기반 직접 추천 생성 중 (TransR)...")
        
        with torch.no_grad():
            # 전체 임베딩
            u_embed, i_embed = self.model()
            
            # 기본 임베딩 (관계 임베딩과 동일 차원)
            u_embed_base = self.model.user_embed
            i_embed_base = self.model.entity_embed[:self.data_loader.n_items]
            
            # 구매 관계의 임베딩과 변환 행렬
            purchase_rel_emb = self.model.relation_embed[self.purchase_relation_id]
            W_purchase = self.model.trans_W[self.purchase_relation_id]  # [emb_size, emb_size]
            
            # 모든 아이템을 구매 관계 공간으로 변환
            all_items_r = torch.matmul(i_embed_base, W_purchase)  # [n_items, emb_size]
            
            recommendations = {}
            
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb_base = u_embed_base[u_original]
                
                # 사용자를 구매 관계 공간으로 변환
                user_r = torch.matmul(user_emb_base, W_purchase)
                
                # TransR: user_r + purchase_relation
                enhanced_user_r = user_r + purchase_rel_emb
                
                # 모든 아이템과의 점수 계산
                scores = torch.matmul(enhanced_user_r, all_items_r.t())
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
            
        return recommendations
    
    def get_purchase_relation_recommendations_simple(self, user_ids, k=20):
        """구매 관계를 사용한 직접 추천 (단순 더하기)"""
        self.model.eval()
        
        print(f"\n구매 관계 기반 직접 추천 생성 중 (단순 더하기)...")
        
        with torch.no_grad():
            # 기본 임베딩
            u_embed_base = self.model.user_embed
            i_embed_base = self.model.entity_embed[:self.data_loader.n_items]
            
            # 구매 관계 임베딩
            purchase_rel_emb = self.model.relation_embed[self.purchase_relation_id]
            
            recommendations = {}
            
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb_base = u_embed_base[u_original]
                
                # 단순 더하기: user + purchase_relation
                enhanced_user = user_emb_base + purchase_rel_emb
                
                # 모든 아이템과의 점수 계산
                scores = torch.matmul(enhanced_user, i_embed_base.t())
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    scores[train_items] = -float('inf')
                
                # Top-K 추천
                _, topk_indices = torch.topk(scores, k)
                recommendations[u] = topk_indices.cpu().numpy().tolist()
            
        return recommendations
    
    def get_standard_recommendations(self, user_ids, k=20):
        """표준 KGAT 추천 (기준선)"""
        self.model.eval()
        
        print(f"\n표준 KGAT 추천 생성 중...")
        
        with torch.no_grad():
            u_embed, i_embed = self.model()
            
            recommendations = {}
            for u in user_ids:
                u_original = u - self.data_loader.n_entities
                user_emb = u_embed[u_original]
                
                # 전체 임베딩으로 계산
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
        
        avg_metrics = {}
        for metric, values in metrics.items():
            avg_metrics[metric] = np.mean(values) if values else 0.0
            
        return avg_metrics
    
    def compare_all_methods(self, n_users=1000, k_list=[10, 20, 30, 50]):
        """모든 방법 비교"""
        test_users = list(self.data_loader.test_user_dict.keys())[:n_users]
        
        print(f"\n{n_users}명의 사용자에 대해 방법 비교 중...")
        
        results = {}
        
        # 1. 표준 KGAT 추천
        standard_recs = self.get_standard_recommendations(test_users, max(k_list))
        results['standard'] = {
            'metrics': self.evaluate_recommendations(
                standard_recs, self.data_loader.test_user_dict, k_list
            )
        }
        
        # 2. 구매 관계 + 단순 더하기
        purchase_simple_recs = self.get_purchase_relation_recommendations_simple(
            test_users, max(k_list)
        )
        results['purchase_simple'] = {
            'metrics': self.evaluate_recommendations(
                purchase_simple_recs, self.data_loader.test_user_dict, k_list
            )
        }
        
        # 3. 구매 관계 + TransR
        purchase_transr_recs = self.get_purchase_relation_recommendations_transr(
            test_users, max(k_list)
        )
        results['purchase_transr'] = {
            'metrics': self.evaluate_recommendations(
                purchase_transr_recs, self.data_loader.test_user_dict, k_list
            )
        }
        
        return results
    
    def print_comparison_results(self, results):
        """결과 출력"""
        print("\n" + "="*80)
        print("구매 관계 기반 직접 추천 비교")
        print("="*80)
        
        # 각 k값에 대한 결과
        for k in [10, 20, 30, 50]:
            print(f"\n=== K={k} ===")
            headers = ['방법', 'Recall', 'Precision', 'NDCG', 'Hit Ratio']
            rows = []
            
            # 표준 KGAT
            standard_metrics = results['standard']['metrics']
            rows.append([
                '표준 KGAT',
                f"{standard_metrics[f'recall@{k}']:.4f}",
                f"{standard_metrics[f'precision@{k}']:.4f}",
                f"{standard_metrics[f'ndcg@{k}']:.4f}",
                f"{standard_metrics[f'hit@{k}']:.4f}"
            ])
            
            # 구매관계 + 단순 더하기
            simple_metrics = results['purchase_simple']['metrics']
            simple_improvement = ((simple_metrics[f'recall@{k}'] - standard_metrics[f'recall@{k}']) 
                                / standard_metrics[f'recall@{k}'] * 100)
            rows.append([
                f'구매관계+단순더하기 ({simple_improvement:+.1f}%)',
                f"{simple_metrics[f'recall@{k}']:.4f}",
                f"{simple_metrics[f'precision@{k}']:.4f}",
                f"{simple_metrics[f'ndcg@{k}']:.4f}",
                f"{simple_metrics[f'hit@{k}']:.4f}"
            ])
            
            # 구매관계 + TransR
            transr_metrics = results['purchase_transr']['metrics']
            transr_improvement = ((transr_metrics[f'recall@{k}'] - standard_metrics[f'recall@{k}']) 
                                / standard_metrics[f'recall@{k}'] * 100)
            rows.append([
                f'구매관계+TransR ({transr_improvement:+.1f}%)',
                f"{transr_metrics[f'recall@{k}']:.4f}",
                f"{transr_metrics[f'precision@{k}']:.4f}",
                f"{transr_metrics[f'ndcg@{k}']:.4f}",
                f"{transr_metrics[f'hit@{k}']:.4f}"
            ])
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # 전체 요약
        print("\n\n=== 전체 요약 (Recall@20 기준) ===")
        standard_recall = results['standard']['metrics']['recall@20']
        simple_recall = results['purchase_simple']['metrics']['recall@20']
        transr_recall = results['purchase_transr']['metrics']['recall@20']
        
        print(f"표준 KGAT: {standard_recall:.4f} (기준선)")
        print(f"구매관계+단순더하기: {simple_recall:.4f} "
              f"({((simple_recall - standard_recall) / standard_recall * 100):+.1f}%)")
        print(f"구매관계+TransR: {transr_recall:.4f} "
              f"({((transr_recall - standard_recall) / standard_recall * 100):+.1f}%)")
        
        # TransR vs 단순 더하기
        if simple_recall > 0:
            transr_vs_simple = ((transr_recall - simple_recall) / simple_recall * 100)
            print(f"\nTransR는 단순 더하기 대비 {transr_vs_simple:+.1f}% "
                  f"{'개선' if transr_vs_simple > 0 else '하락'}")


def main():
    parser = argparse.ArgumentParser(description='구매 관계 기반 직접 추천')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 KGAT 모델 체크포인트')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='평가할 사용자 수')
    parser.add_argument('--output-dir', type=str, default='results/purchase_relation',
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
    recommender = PurchaseRelationRecommender(model, data_loader, device)
    results = recommender.compare_all_methods(n_users=args.n_users)
    
    # 결과 출력
    recommender.print_comparison_results(results)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'n_users': args.n_users,
        'results': results
    }
    
    result_file = os.path.join(args.output_dir, f'purchase_relation_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {result_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()