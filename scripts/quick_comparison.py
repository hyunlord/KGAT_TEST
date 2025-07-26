#!/usr/bin/env python
"""
학습된 모델들의 빠른 성능 비교
체크포인트에서 메트릭만 추출하여 표시
"""

import os
import torch
import glob
from tabulate import tabulate


def extract_metrics_from_checkpoint(checkpoint_path):
    """체크포인트에서 메트릭 추출"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # PyTorch Lightning 체크포인트
        if 'callbacks' in checkpoint:
            metrics = {}
            # Callback에서 메트릭 찾기
            for callback_name, callback_state in checkpoint['callbacks'].items():
                if 'best_model_score' in str(callback_state):
                    if isinstance(callback_state, dict) and 'best_model_score' in callback_state:
                        best_score = callback_state['best_model_score']
                        if hasattr(best_score, 'item'):
                            metrics['best_recall@20'] = best_score.item()
                        else:
                            metrics['best_recall@20'] = float(best_score)
            
            # Trainer state에서 찾기
            if 'trainer' in checkpoint and 'logged_metrics' in checkpoint['trainer']:
                logged = checkpoint['trainer']['logged_metrics']
                for key, value in logged.items():
                    if 'val_' in key:
                        metric_name = key.replace('val_', '')
                        metrics[metric_name] = float(value) if hasattr(value, 'item') else value
            
            return metrics
            
        # Original KGAT 체크포인트
        elif 'state_dict' in checkpoint or isinstance(checkpoint, dict):
            # Original 모델은 체크포인트에 메트릭을 저장하지 않음
            return {'type': 'original', 'no_metrics': True}
            
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def find_checkpoints():
    """모든 체크포인트 찾기"""
    checkpoints = {}
    
    # 다양한 패턴으로 체크포인트 찾기
    patterns = [
        'models/*.pth',
        'models/*.ckpt',
        'models/**/*.ckpt',
        'lightning_logs/*/checkpoints/*.ckpt'
    ]
    
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            name = os.path.basename(path)
            
            # 모델 타입 추정
            if 'original' in name.lower():
                model_type = 'Original'
            elif 'fixed' in name.lower():
                model_type = 'Fixed'
            elif 'improved' in name.lower():
                model_type = 'Improved'
            elif 'kgat' in name.lower():
                model_type = 'Lightning'
            else:
                model_type = 'Unknown'
            
            checkpoints[f"{model_type}: {name}"] = path
    
    return checkpoints


def main():
    print("="*80)
    print("KGAT 모델 체크포인트 빠른 비교")
    print("="*80)
    
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("체크포인트를 찾을 수 없습니다.")
        return
    
    # 메트릭 추출
    results = []
    for name, path in checkpoints.items():
        print(f"\n처리 중: {name}")
        metrics = extract_metrics_from_checkpoint(path)
        
        if metrics:
            if 'no_metrics' in metrics:
                results.append({
                    'Model': name,
                    'Recall@20': 'N/A (저장된 메트릭 없음)',
                    'Path': path
                })
            else:
                results.append({
                    'Model': name,
                    'Recall@20': f"{metrics.get('recall@20', metrics.get('best_recall@20', 'N/A')):.4f}" 
                               if isinstance(metrics.get('recall@20', metrics.get('best_recall@20', 0)), (int, float)) 
                               else 'N/A',
                    'Path': path
                })
    
    # 결과 표시
    if results:
        print("\n" + "="*80)
        print("체크포인트 메트릭 비교")
        print("="*80)
        
        # 간단한 표
        simple_results = [{'모델': r['Model'], 'Recall@20': r['Recall@20']} for r in results]
        print(tabulate(simple_results, headers='keys', tablefmt='grid'))
        
        print("\n상세 경로:")
        for r in results:
            print(f"- {r['Model']}: {r['Path']}")
        
        print("\n참고: Original 모델은 체크포인트에 메트릭을 저장하지 않습니다.")
        print("정확한 비교를 위해서는 scripts/compare_all_models.sh를 사용하세요.")
    
    else:
        print("메트릭을 추출할 수 없습니다.")


if __name__ == "__main__":
    main()