import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from kgat_lightning import KGATLightning
from data_module import KGATDataModule


class MetricsCallback(pl.Callback):
    """메트릭 추적 및 출력을 위한 커스텀 콜백"""
    def __init__(self):
        self.best_metrics = {}
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        
        # 최고 메트릭 추적
        for key, value in metrics.items():
            if 'val_' in key:
                metric_name = key.replace('val_', '')
                if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
        
        # 현재 에포크 메트릭 출력
        if trainer.current_epoch % 5 == 0:
            print(f"\n[에포크 {trainer.current_epoch}] 검증 메트릭:")
            for key, value in metrics.items():
                if 'val_' in key:
                    print(f"  {key}: {value:.4f}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    """메인 학습 함수"""
    print(OmegaConf.to_yaml(cfg))
    
    # 랜덤 시드 설정
    pl.seed_everything(cfg.training.seed)
    
    # 데이터 모듈 초기화
    data_module = KGATDataModule(cfg.data)
    data_module.setup()
    
    # 모델 초기화를 위한 데이터 통계 가져오기
    stats = data_module.get_statistics()
    cfg.model.n_users = stats['n_users']
    cfg.model.n_entities = stats['n_entities']
    cfg.model.n_relations = stats['n_relations']
    
    print(f"\n데이터 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 모델 초기화
    model = KGATLightning(cfg.model)
    
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
        MetricsCallback()
    ]
    
    # 로거 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name=f"kgat_{cfg.data.dataset}",
        version=timestamp
    )
    
    # 트레이너 초기화
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.get('strategy', 'auto'),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        precision=cfg.training.precision,
        sync_batchnorm=cfg.training.get('sync_batchnorm', False),
        replace_sampler_ddp=cfg.training.get('replace_sampler_ddp', True),
        deterministic=True
    )
    
    # 모델 학습
    print("\n학습 시작...")
    trainer.fit(model, data_module)
    
    # 모델 테스트
    print("\n테스트 세트에서 평가 중...")
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    # 최종 결과 출력
    print("\n" + "="*60)
    print("학습 완료")
    print("="*60)
    print(f"\n최고 검증 메트릭:")
    for key, value in callbacks[3].best_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n테스트 결과:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n모델 체크포인트 저장 위치: {trainer.checkpoint_callback.dirpath}")
    print(f"TensorBoard 로그 저장 위치: {logger.log_dir}")
    
    # 최종 모델 저장
    final_model_path = os.path.join(cfg.training.model_save_dir, f"kgat_final_{timestamp}.pth")
    os.makedirs(cfg.training.model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"최종 모델 저장 위치: {final_model_path}")
    
    return test_results[0]


if __name__ == "__main__":
    train()