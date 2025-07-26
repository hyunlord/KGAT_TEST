"""
여러 KGAT 모델 구현체들의 성능을 비교하는 스크립트
Original, Lightning, Fixed 버전의 결과를 비교 분석
"""

import os
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from kgat_original import KGAT as KGATOriginal
from kgat_lightning import KGATLightning
from kgat_lightning_fixed import KGATLightningFixed
from kgat_improved import KGATImproved
from data_module import KGATDataModule
from data_loader_original import DataLoaderOriginal
from evaluate_original import test as test_original
from omegaconf import OmegaConf


class ModelComparison:
    """여러 KGAT 구현체의 성능을 비교하는 클래스"""
    
    def __init__(self, data_config):
        self.data_config = data_config
        self.results = {}
        self.models = {}
        
    def load_original_model(self, checkpoint_path, args):
        """Original KGAT 모델 로드"""
        print("Original KGAT 모델 로딩...")
        
        # 데이터 로더 초기화
        data_loader = DataLoaderOriginal(args, None)
        
        # 모델 초기화
        model = KGATOriginal(
            args,
            data_loader.n_users,
            data_loader.n_items,
            data_loader.n_entities,
            data_loader.n_relations,
            data_loader.adjacency_dict['plain_adj'],
            data_loader.laplacian_dict['kg_mat']
        )
        
        # 체크포인트 로드
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            print(f"Original 모델 체크포인트 로드 완료: {checkpoint_path}")
        
        return model, data_loader
    
    def load_lightning_model(self, checkpoint_path, config_path, model_class):
        """Lightning 기반 모델 로드"""
        print(f"{model_class.__name__} 모델 로딩...")
        
        # 설정 로드
        config = OmegaConf.load(config_path)
        
        # 데이터 모듈 초기화
        data_module = KGATDataModule(config.data)
        data_module.setup()
        
        # 모델 설정 업데이트
        stats = data_module.get_statistics()
        config.model.n_users = stats['n_users']
        config.model.n_entities = stats['n_entities']
        config.model.n_relations = stats['n_relations']
        
        # 모델 로드
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                model = model_class.load_from_checkpoint(
                    checkpoint_path,
                    config=config.model,
                    map_location='cpu'
                )
            except:
                # 호환성 문제 처리
                model = model_class(config.model)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict, strict=False)
        else:
            model = model_class(config.model)
        
        return model, data_module
    
    def evaluate_original(self, model, data_loader, device, k_list=[20, 40, 60, 80, 100]):
        """Original 모델 평가"""
        print("Original 모델 평가 중...")
        model.eval()
        
        with torch.no_grad():
            u_embed, i_embed = model()
            
            # 테스트 사용자 선택
            test_users = list(data_loader.test_user_dict.keys())[:5000]
            
            # 평가
            results = test_original(model, data_loader, test_users, k_list, device)
            
        return results
    
    def evaluate_lightning(self, model, data_module, device, k_list=[20, 40, 60, 80, 100]):
        """Lightning 모델 평가"""
        print(f"{model.__class__.__name__} 모델 평가 중...")
        model.eval()
        model = model.to(device)
        
        results = defaultdict(list)
        
        # 테스트 데이터로더
        test_loader = data_module.test_dataloader()
        
        with torch.no_grad():
            for batch in test_loader:
                # 그래프 데이터 준비
                edge_index_ui = batch['edge_index_ui'].to(device)
                
                # Forward pass
                if hasattr(model, 'compute_edge_weights'):
                    # Fixed 모델의 경우
                    num_nodes = model.n_users + model.n_entities
                    _, edge_weight = model.compute_edge_weights(edge_index_ui, num_nodes)
                    user_embeddings, item_embeddings = model(edge_index_ui, edge_weight)
                else:
                    # 일반 Lightning 모델
                    user_embeddings, item_embeddings = model(edge_index_ui)
                
                # 평가
                eval_users = batch['eval_user_ids']
                train_items = batch['train_items']
                test_items = batch['test_items']
                
                for i, user_id in enumerate(eval_users):
                    if i >= len(test_items) or len(test_items[i]) == 0:
                        continue
                    
                    # 사용자 임베딩과 모든 아이템의 점수 계산
                    user_emb = user_embeddings[user_id]
                    scores = torch.matmul(user_emb, item_embeddings.t())
                    
                    # 학습 데이터에서 본 아이템 제외
                    if i < len(train_items) and len(train_items[i]) > 0:
                        scores[train_items[i]] = -float('inf')
                    
                    # Top-K 추천
                    for k in k_list:
                        _, topk_indices = torch.topk(scores, k)
                        topk_indices = topk_indices.cpu().numpy()
                        
                        # 메트릭 계산
                        test_set = set(test_items[i])
                        hits = [1 if idx in test_set else 0 for idx in topk_indices]
                        
                        # Recall
                        recall = sum(hits) / len(test_set)
                        results[f'recall@{k}'].append(recall)
                        
                        # Precision
                        precision = sum(hits) / k
                        results[f'precision@{k}'].append(precision)
                        
                        # NDCG
                        dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
                        idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(test_set), k))])
                        ndcg = dcg / idcg if idcg > 0 else 0
                        results[f'ndcg@{k}'].append(ndcg)
                        
                        # Hit Ratio
                        hit_ratio = 1.0 if sum(hits) > 0 else 0.0
                        results[f'hit_ratio@{k}'].append(hit_ratio)
        
        # 평균 계산
        avg_results = {}
        for metric, values in results.items():
            avg_results[metric.replace('@', '')] = np.mean(values) if values else 0.0
        
        # Original 형식으로 변환
        formatted_results = {
            'recall': [avg_results.get(f'recall{k}', 0.0) for k in k_list],
            'precision': [avg_results.get(f'precision{k}', 0.0) for k in k_list],
            'ndcg': [avg_results.get(f'ndcg{k}', 0.0) for k in k_list],
            'hit_ratio': [avg_results.get(f'hit_ratio{k}', 0.0) for k in k_list]
        }
        
        return formatted_results
    
    def compare_all_models(self, model_configs):
        """모든 모델 비교"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k_list = [20, 40, 60, 80, 100]
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*50}")
            print(f"평가 중: {model_name}")
            print('='*50)
            
            if config['type'] == 'original':
                # Original 모델
                args = config['args']
                model, data_loader = self.load_original_model(
                    config.get('checkpoint'), args
                )
                model = model.to(device)
                results = self.evaluate_original(model, data_loader, device, k_list)
                
            else:
                # Lightning 기반 모델
                model_class = config['class']
                model, data_module = self.load_lightning_model(
                    config.get('checkpoint'),
                    config['config_path'],
                    model_class
                )
                results = self.evaluate_lightning(model, data_module, device, k_list)
            
            self.results[model_name] = results
            self.models[model_name] = model
    
    def print_comparison_table(self):
        """비교 결과를 표로 출력"""
        print("\n" + "="*80)
        print("모델 성능 비교 결과")
        print("="*80)
        
        metrics = ['recall', 'precision', 'ndcg', 'hit_ratio']
        k_values = [20, 40, 60, 80, 100]
        
        for metric in metrics:
            print(f"\n{metric.upper()} 비교:")
            
            # 테이블 데이터 준비
            headers = ['Model'] + [f'@{k}' for k in k_values]
            rows = []
            
            for model_name, results in self.results.items():
                if metric in results:
                    row = [model_name] + [f"{val:.4f}" for val in results[metric]]
                    rows.append(row)
            
            # 테이블 출력
            print(tabulate(rows, headers=headers, tablefmt='grid'))
            
            # 최고 성능 모델 표시
            if rows:
                best_scores = []
                for k_idx in range(len(k_values)):
                    scores = [float(row[k_idx + 1]) for row in rows]
                    best_idx = scores.index(max(scores))
                    best_model = rows[best_idx][0]
                    best_score = scores[best_idx]
                    best_scores.append(f"{best_model} ({best_score:.4f})")
                
                print(f"최고 성능: {', '.join([f'@{k}: {bs}' for k, bs in zip(k_values, best_scores)])}")
    
    def visualize_comparison(self, save_path='results/model_comparison'):
        """비교 결과 시각화"""
        os.makedirs(save_path, exist_ok=True)
        
        metrics = ['recall', 'precision', 'ndcg', 'hit_ratio']
        k_values = [20, 40, 60, 80, 100]
        
        # 각 메트릭별 그래프
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            for model_name, results in self.results.items():
                if metric in results:
                    plt.plot(k_values, results[metric], marker='o', label=model_name)
            
            plt.xlabel('K')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} Comparison across Models')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(save_path, f'{metric}_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 종합 비교 히트맵
        plt.figure(figsize=(12, 8))
        
        # 데이터 준비
        data = []
        model_names = list(self.results.keys())
        
        for model_name in model_names:
            model_data = []
            for metric in metrics:
                if metric in self.results[model_name]:
                    # K=20 값만 사용
                    model_data.append(self.results[model_name][metric][0])
                else:
                    model_data.append(0)
            data.append(model_data)
        
        # 히트맵 그리기
        sns.heatmap(data, 
                   xticklabels=[f'{m}@20' for m in metrics],
                   yticklabels=model_names,
                   annot=True, 
                   fmt='.4f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Score'})
        
        plt.title('Model Performance Comparison Heatmap (@K=20)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n시각화 결과가 {save_path}에 저장되었습니다.")
    
    def save_results(self, save_path='results/model_comparison'):
        """결과를 JSON 파일로 저장"""
        os.makedirs(save_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(save_path, f'comparison_results_{timestamp}.json')
        
        # 저장할 데이터 준비
        save_data = {
            'timestamp': timestamp,
            'results': self.results,
            'model_info': {
                model_name: {
                    'type': model.__class__.__name__,
                    'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
                }
                for model_name, model in self.models.items()
            }
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"결과가 {result_file}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='KGAT 모델 구현체 비교')
    parser.add_argument('--original-checkpoint', type=str,
                        help='Original KGAT 체크포인트 경로')
    parser.add_argument('--lightning-checkpoint', type=str,
                        help='Lightning KGAT 체크포인트 경로')
    parser.add_argument('--fixed-checkpoint', type=str,
                        help='Fixed KGAT 체크포인트 경로')
    parser.add_argument('--improved-checkpoint', type=str,
                        help='Improved KGAT 체크포인트 경로')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--output-dir', type=str, default='results/model_comparison',
                        help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # Original 모델용 args 준비
    class OriginalArgs:
        def __init__(self):
            self.dataset = 'amazon-book'
            self.data_path = 'data/'
            self.model_type = 'kgat'
            self.adj_type = 'si'
            self.alg_type = 'bi'
            self.embed_size = 64
            self.layer_size = [64, 32, 16]
            self.lr = 0.0001
            self.regs = [1e-5, 1e-5]
            self.node_dropout = [0.1]
            self.mess_dropout = [0.1, 0.1, 0.1]
            self.seed = 2019
    
    original_args = OriginalArgs()
    
    # 모델 설정
    model_configs = {}
    
    if args.original_checkpoint:
        model_configs['Original KGAT'] = {
            'type': 'original',
            'checkpoint': args.original_checkpoint,
            'args': original_args
        }
    
    if args.lightning_checkpoint:
        model_configs['Lightning KGAT'] = {
            'type': 'lightning',
            'class': KGATLightning,
            'checkpoint': args.lightning_checkpoint,
            'config_path': args.config
        }
    
    if args.fixed_checkpoint:
        model_configs['Fixed KGAT'] = {
            'type': 'lightning',
            'class': KGATLightningFixed,
            'checkpoint': args.fixed_checkpoint,
            'config_path': args.config
        }
    
    if args.improved_checkpoint:
        model_configs['Improved KGAT'] = {
            'type': 'lightning',
            'class': KGATImproved,
            'checkpoint': args.improved_checkpoint,
            'config_path': args.config
        }
    
    # 비교 실행
    comparison = ModelComparison(args.config)
    comparison.compare_all_models(model_configs)
    
    # 결과 출력
    comparison.print_comparison_table()
    
    # 시각화
    comparison.visualize_comparison(args.output_dir)
    
    # 결과 저장
    comparison.save_results(args.output_dir)


if __name__ == "__main__":
    main()