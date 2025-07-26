"""
KGAT 모델을 사용한 표준 추천 vs 관계 강화 추천 비교
Original과 Fixed 모델 모두 지원
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
import pytorch_lightning as pl


class RelationEnhancedComparisonAll:
    """표준 추천과 관계 강화 추천 비교 (모든 모델 지원)"""
    
    def __init__(self, model, data_loader, device, model_type='original'):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.model_type = model_type
        
    def get_embeddings(self):
        """모델 타입에 따라 임베딩 추출"""
        if self.model_type == 'original':
            # Original KGAT
            return self.model()
        else:
            # Fixed KGAT (PyTorch Lightning)
            self.model.eval()
            with torch.no_grad():
                # 전체 노드 임베딩 계산
                all_embed = self.model.get_all_embeddings()
                
                # 사용자와 아이템 임베딩 분리
                u_embed = all_embed[:self.model.n_users]
                i_embed = all_embed[self.model.n_users:]
                
                return u_embed, i_embed
    
    def get_relation_embeddings(self):
        """관계 임베딩 추출"""
        if self.model_type == 'original':
            return self.model.relation_embed
        else:
            return self.model.relation_embedding.weight
        
    def get_standard_recommendations(self, user_ids, k=20):
        """표준 방식: user-item 유사도만 사용"""
        self.model.eval()
        
        with torch.no_grad():
            # 전체 임베딩 계산
            u_embed, i_embed = self.get_embeddings()
            
            recommendations = {}
            for u in user_ids:
                # 사용자 ID 변환 (엔티티 공간에서 원래 공간으로)
                if self.model_type == 'original':
                    u_original = u - self.data_loader.n_entities
                else:
                    u_original = u  # Fixed는 이미 원래 공간
                
                # 사용자 임베딩
                user_emb = u_embed[u_original]
                
                # 모든 아이템과의 점수 계산
                scores = torch.matmul(user_emb, i_embed.t())
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    # Fixed 모델의 경우 아이템 ID 조정
                    if self.model_type != 'original':
                        train_items = [item - self.data_loader.n_users for item in train_items 
                                     if item >= self.data_loader.n_users]
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
            u_embed, i_embed = self.get_embeddings()
            relation_embed = self.get_relation_embeddings()
            
            # KG에서 관계 정보 활용
            recommendations = {}
            
            for u in user_ids:
                # 사용자 ID 변환
                if self.model_type == 'original':
                    u_original = u - self.data_loader.n_entities
                else:
                    u_original = u
                
                # 기본 사용자 임베딩
                user_emb = u_embed[u_original]
                
                # 관계별 강화된 임베딩 계산
                n_items = self.data_loader.n_items if self.model_type == 'original' else i_embed.size(0)
                enhanced_scores = torch.zeros(n_items).to(self.device)
                
                # 사용자가 상호작용한 아이템들
                if u in self.data_loader.train_user_dict:
                    interacted_items = self.data_loader.train_user_dict[u]
                    
                    # 각 상호작용한 아이템에서 다른 아이템으로의 관계 고려
                    for item in interacted_items:
                        # Fixed 모델의 경우 item ID 조정
                        if self.model_type != 'original':
                            if item < self.data_loader.n_users:
                                continue  # 사용자 노드는 건너뜀
                            item_adjusted = item
                        else:
                            item_adjusted = item
                            
                        # 이 아이템과 연결된 다른 아이템들 찾기
                        if item_adjusted in self.data_loader.train_kg_dict:
                            for target, relation in self.data_loader.train_kg_dict[item_adjusted]:
                                # 타겟이 아이템인지 확인
                                if self.model_type == 'original':
                                    if target < self.data_loader.n_items:
                                        target_idx = target
                                    else:
                                        continue
                                else:
                                    if target >= self.data_loader.n_users:
                                        target_idx = target - self.data_loader.n_users
                                    else:
                                        continue
                                
                                # 관계 임베딩 가져오기
                                rel_emb = relation_embed[relation]
                                
                                # user + relation 임베딩
                                enhanced_user = user_emb + 0.1 * rel_emb  # 가중치 조절 가능
                                
                                # 타겟 아이템과의 유사도
                                target_emb = i_embed[target_idx]
                                score = torch.dot(enhanced_user, target_emb)
                                
                                # 점수 누적
                                enhanced_scores[target_idx] += score
                
                # 기본 점수와 관계 강화 점수 결합
                base_scores = torch.matmul(user_emb, i_embed.t())
                final_scores = base_scores + 0.3 * enhanced_scores  # 가중치 조절 가능
                
                # 이미 본 아이템 제외
                if u in self.data_loader.train_user_dict:
                    train_items = self.data_loader.train_user_dict[u]
                    if self.model_type != 'original':
                        train_items = [item - self.data_loader.n_users for item in train_items 
                                     if item >= self.data_loader.n_users]
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
            
            # Fixed 모델의 경우 테스트 아이템 ID 조정
            if self.model_type != 'original':
                test_items = {item - self.data_loader.n_users for item in test_items 
                            if item >= self.data_loader.n_users}
            
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


def load_original_model(checkpoint_path, data_loader, device):
    """Original KGAT 모델 로드"""
    from kgat_original import KGAT
    
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
            self.use_pretrain = 0
    
    args = OriginalArgs()
    
    model = KGAT(
        args,
        data_loader.n_users,
        data_loader.n_items,
        data_loader.n_entities,
        data_loader.n_relations,
        data_loader.adjacency_dict['plain_adj'],
        data_loader.laplacian_dict['kg_mat']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    return model, 'original'


def load_fixed_model(checkpoint_path, data_loader, device):
    """Fixed KGAT 모델 로드"""
    from kgat_lightning_fixed import KGATLightningFixed
    
    # checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 설정 재구성
    class FixedConfig:
        def __init__(self, hparams, data_loader):
            # checkpoint의 hyperparameters 사용
            for key, value in hparams.items():
                setattr(self, key, value)
            
            # 데이터 로더에서 필요한 정보 추가
            self.n_users = data_loader.n_users
            self.n_entities = data_loader.n_entities
            self.n_relations = data_loader.n_relations
            
            # 필수 속성 확인 및 기본값 설정
            if not hasattr(self, 'embedding_size'):
                self.embedding_size = 64
            if not hasattr(self, 'layer_sizes'):
                self.layer_sizes = [64, 32, 16]
            if not hasattr(self, 'aggregator'):
                self.aggregator = 'bi'
            if not hasattr(self, 'dropout_rates'):
                self.dropout_rates = [0.1, 0.1, 0.1]
            if not hasattr(self, 'batch_size'):
                self.batch_size = 1024
    
    # hyperparameters 추출
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
    else:
        # Lightning checkpoint 구조
        hparams = checkpoint.get('hparams', {})
    
    config = FixedConfig(hparams, data_loader)
    
    # 모델 생성
    model = KGATLightningFixed(config)
    
    # state_dict 로드
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 그래프 구조 설정 (Fixed 모델용)
    model.set_graph_data(
        data_loader.adjacency_dict['plain_adj'],
        data_loader.laplacian_dict.get('kg_mat', None)
    )
    
    return model, 'fixed'


def main():
    parser = argparse.ArgumentParser(description='KGAT 표준 vs 관계 강화 추천 비교 (모든 모델)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 KGAT 모델 체크포인트')
    parser.add_argument('--model-type', type=str, choices=['original', 'fixed', 'auto'], 
                        default='auto', help='모델 타입')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='비교할 사용자 수')
    parser.add_argument('--output-dir', type=str, default='results/relation_comparison',
                        help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로드
    print("데이터 로딩 중...")
    from data_loader_original import DataLoaderOriginal
    
    class DataArgs:
        def __init__(self):
            self.dataset = 'amazon-book'
            self.data_path = 'data/'
            self.cf_batch_size = 1024
            self.kg_batch_size = 2048
            self.test_batch_size = 10000
            self.use_pretrain = 0
            self.pretrain_embedding_dir = 'pretrain/'
    
    data_args = DataArgs()
    data_loader = DataLoaderOriginal(data_args, None)
    
    # 모델 타입 자동 감지
    if args.model_type == 'auto':
        if args.checkpoint.endswith('.pth'):
            args.model_type = 'original'
        elif args.checkpoint.endswith('.ckpt'):
            args.model_type = 'fixed'
        else:
            # 파일 내용으로 판단
            try:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                if 'state_dict' in checkpoint or 'hyper_parameters' in checkpoint:
                    args.model_type = 'fixed'
                else:
                    args.model_type = 'original'
            except:
                print("모델 타입을 자동으로 감지할 수 없습니다. --model-type을 명시해주세요.")
                return
    
    print(f"모델 타입: {args.model_type}")
    
    # 모델 로드
    print("모델 로딩 중...")
    if args.model_type == 'original':
        model, model_type = load_original_model(args.checkpoint, data_loader, device)
    else:
        model, model_type = load_fixed_model(args.checkpoint, data_loader, device)
    
    print(f"모델 로드 완료: {args.checkpoint}")
    
    # 비교 실행
    comparison = RelationEnhancedComparisonAll(model, data_loader, device, model_type)
    results = comparison.compare_methods(n_users=args.n_users)
    
    # 결과 출력
    comparison.print_comparison(results)
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'model_type': model_type,
        'n_users': args.n_users,
        'metrics': {
            'standard': results['standard'],
            'enhanced': results['enhanced']
        }
    }
    
    result_file = os.path.join(args.output_dir, f'comparison_{model_type}_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {result_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()