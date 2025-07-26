"""
디버깅 버전의 학습 스크립트
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import warnings

# PyTorch 분산 학습 경고 필터링
warnings.filterwarnings("ignore", message="No device id is provided")

from kgat_lightning_fixed import KGATLightningFixed
from data_module import KGATDataModule


class DebugCallback(pl.Callback):
    """디버깅을 위한 콜백"""
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:  # 첫 번째 배치만
            print(f"\n=== Validation Batch {batch_idx} Debug ===")
            print(f"Batch keys: {batch.keys()}")
            print(f"Number of eval users: {len(batch['eval_user_ids'])}")
            
            # 첫 번째 사용자 확인
            if len(batch['train_items']) > 0 and len(batch['test_items']) > 0:
                print(f"First user train items: {len(batch['train_items'][0])}")
                print(f"First user test items: {len(batch['test_items'][0])}")
                print(f"Sample test items: {batch['test_items'][0][:5]}")
            
            # 출력 확인
            if outputs:
                print(f"Output metrics: {outputs}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # 시드 설정
    pl.seed_everything(cfg.training.seed)
    
    # 데이터 모듈 초기화
    data_module = KGATDataModule(cfg.data)
    data_module.setup()
    
    # 데이터 통계 출력
    stats = data_module.get_statistics()
    print(f"\n데이터 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 모델 설정에 데이터 정보 추가
    cfg.model.n_users = stats['n_users']
    cfg.model.n_entities = stats['n_entities']
    cfg.model.n_relations = stats['n_relations']
    
    # 모델 초기화 (수정된 버전 사용)
    model = KGATLightningFixed(cfg.model)
    # n_items 속성 추가
    model.n_items = data_module.n_items
    
    # 콜백 설정
    callbacks = [
        ModelCheckpoint(
            monitor='val_recall@20',
            mode='max',
            save_top_k=3,
            filename='kgat-{epoch:02d}-{val_recall@20:.3f}',
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor='val_recall@20',
            mode='max',
            patience=cfg.training.early_stopping_patience,
            verbose=True
        ),
        RichProgressBar(),
        DebugCallback()  # 디버깅 콜백 추가
    ]
    
    # 로거 설정
    logger = TensorBoardLogger(
        save_dir="logs/",
        name=f"kgat_debug_{cfg.data.dataset}",
        version=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    # 트레이너 설정 (간단하게)
    trainer = pl.Trainer(
        max_epochs=30,  # 빠른 테스트를 위해 줄임
        accelerator='gpu',
        devices=1,  # 디버깅을 위해 단일 GPU
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # 학습
    print("\n학습 시작...")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()