"""
디버깅을 위한 스크립트
"""
import torch
import numpy as np
from data_module import KGATDataModule
from omegaconf import OmegaConf

def debug_data_module():
    """데이터 모듈 문제 확인"""
    config = OmegaConf.create({
        'data_dir': 'data/amazon-book',
        'batch_size': 32,
        'num_workers': 0,
        'neg_sample_size': 1
    })
    
    data_module = KGATDataModule(config)
    data_module.setup()
    
    print(f"\n=== 데이터 통계 ===")
    print(f"사용자 수: {data_module.n_users}")
    print(f"아이템 수: {data_module.n_items}")
    print(f"엔티티 수: {data_module.n_entities}")
    print(f"관계 수: {data_module.n_relations}")
    
    # 샘플 사용자 확인
    print(f"\n=== 샘플 사용자 데이터 ===")
    for i, user_id in enumerate(list(data_module.train_user_dict.keys())[:5]):
        train_items = data_module.train_user_dict[user_id]
        test_items = data_module.test_user_dict.get(user_id, [])
        print(f"사용자 {user_id}: 학습 아이템 {len(train_items)}개, 테스트 아이템 {len(test_items)}개")
        print(f"  학습 아이템 샘플: {train_items[:5]}")
        print(f"  테스트 아이템 샘플: {test_items[:5]}")
    
    # 검증 데이터로더 확인
    print(f"\n=== 검증 데이터로더 확인 ===")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    
    print(f"배치 키: {batch.keys()}")
    print(f"평가 사용자 수: {len(batch['eval_user_ids'])}")
    print(f"엣지 인덱스 shape: {batch['edge_index_ui'].shape}")
    
    # 첫 번째 사용자의 데이터 확인
    if len(batch['train_items']) > 0:
        print(f"\n첫 번째 사용자의 학습 아이템: {len(batch['train_items'][0])}개")
        print(f"첫 번째 사용자의 테스트 아이템: {len(batch['test_items'][0])}개")
    
    return data_module


def debug_embeddings(model, data_module):
    """임베딩 문제 확인"""
    model.eval()
    
    with torch.no_grad():
        # 간단한 forward pass
        edge_index_ui = data_module.edge_index_ui
        edge_index_kg = data_module.edge_index_kg
        edge_type_kg = data_module.edge_type_kg
        
        user_embeds, entity_embeds = model(edge_index_ui, edge_index_kg, edge_type_kg)
        
        print(f"\n=== 임베딩 통계 ===")
        print(f"사용자 임베딩 shape: {user_embeds.shape}")
        print(f"엔티티 임베딩 shape: {entity_embeds.shape}")
        print(f"사용자 임베딩 평균: {user_embeds.mean().item():.4f}")
        print(f"사용자 임베딩 표준편차: {user_embeds.std().item():.4f}")
        print(f"엔티티 임베딩 평균: {entity_embeds.mean().item():.4f}")
        print(f"엔티티 임베딩 표준편차: {entity_embeds.std().item():.4f}")
        
        # 점수 계산 테스트
        user_id = 0
        user_embed = user_embeds[user_id]
        scores = torch.matmul(user_embed, entity_embeds.t())
        
        print(f"\n=== 점수 통계 ===")
        print(f"점수 shape: {scores.shape}")
        print(f"점수 평균: {scores.mean().item():.4f}")
        print(f"점수 표준편차: {scores.std().item():.4f}")
        print(f"점수 최대값: {scores.max().item():.4f}")
        print(f"점수 최소값: {scores.min().item():.4f}")
        
        # Top-K 확인
        k = 20
        top_scores, top_indices = torch.topk(scores, k)
        print(f"\nTop-{k} 점수: {top_scores.tolist()}")
        print(f"Top-{k} 인덱스: {top_indices.tolist()}")


if __name__ == "__main__":
    print("데이터 모듈 디버깅 시작...")
    data_module = debug_data_module()
    
    # 간단한 모델로 테스트
    from kgat_lightning import KGATLightning
    model_config = OmegaConf.create({
        'n_users': data_module.n_users,
        'n_entities': data_module.n_entities,
        'n_relations': data_module.n_relations,
        'embed_dim': 64,
        'layer_dims': [32, 16],
        'dropout': 0.1,
        'aggregator': 'bi-interaction',
        'reg_weight': 1e-5,
        'lr': 0.001
    })
    
    model = KGATLightning(model_config)
    debug_embeddings(model, data_module)