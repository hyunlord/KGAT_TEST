"""
Hydra 없이 간단한 argparse로 학습하는 스크립트
"""
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

from kgat_lightning import KGATLightning
from data_module import KGATDataModule


def train(args):
    """메인 학습 함수"""
    # 랜덤 시드 설정
    pl.seed_everything(args.seed)
    
    # 데이터 설정
    data_config = type('Config', (), {
        'data_dir': args.data_dir,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'neg_sample_size': 1
    })
    
    # 데이터 모듈 초기화
    data_module = KGATDataModule(data_config)
    data_module.setup()
    
    # 데이터 통계
    stats = data_module.get_statistics()
    print(f"\n데이터 로드 성공:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 모델 설정
    model_config = type('Config', (), {
        'n_users': stats['n_users'],
        'n_entities': stats['n_entities'],
        'n_relations': stats['n_relations'],
        'embed_dim': args.embed_dim,
        'layer_dims': args.layer_dims,
        'dropout': args.dropout,
        'aggregator': args.aggregator,
        'reg_weight': args.reg_weight,
        'lr': args.lr
    })
    
    # 모델 초기화
    model = KGATLightning(model_config)
    
    # 콜백 설정
    callbacks = [
        ModelCheckpoint(
            monitor='val_recall@20',
            mode='max',
            save_top_k=3,
            filename='kgat-{epoch:02d}-{val_recall@20:.3f}',
            save_last=True
        ),
        EarlyStopping(
            monitor='val_recall@20',
            mode='max',
            patience=args.early_stopping_patience
        ),
        RichProgressBar()
    ]
    
    # 로거 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f"kgat_{args.dataset}",
        version=timestamp
    )
    
    # 트레이너 초기화
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        strategy=args.strategy,
        callbacks=callbacks,
        logger=logger,
        precision=args.precision,
        gradient_clip_val=5.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=100
    )
    
    # 학습
    print("\n학습 시작...")
    trainer.fit(model, data_module)
    
    # 테스트
    print("\n테스트 중...")
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    print("\n학습 완료!")
    print(f"체크포인트: {trainer.checkpoint_callback.dirpath}")
    print(f"로그: {logger.log_dir}")
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description='KGAT 학습 (Hydra 없음)')
    
    # 데이터 관련
    parser.add_argument('--data-dir', type=str, default='data/amazon-book',
                        help='데이터 디렉토리 경로')
    parser.add_argument('--dataset', type=str, default='amazon-book',
                        help='데이터셋 이름')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='배치 크기')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='데이터로더 워커 수')
    
    # 모델 관련
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='임베딩 차원')
    parser.add_argument('--layer-dims', type=int, nargs='+', default=[32, 16],
                        help='KGAT 레이어 차원')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='드롭아웃 비율')
    parser.add_argument('--aggregator', type=str, default='bi-interaction',
                        help='어그리게이터 타입')
    parser.add_argument('--reg-weight', type=float, default=1e-5,
                        help='L2 정규화 가중치')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률')
    
    # 학습 관련
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='최대 에포크')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                        help='조기 종료 인내심')
    parser.add_argument('--devices', type=int, default=1,
                        help='GPU 개수')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='학습 전략 (auto, ddp, deepspeed 등)')
    parser.add_argument('--precision', type=int, default=16,
                        help='정밀도 (16 또는 32)')
    parser.add_argument('--check-val-every-n-epoch', type=int, default=5,
                        help='N 에포크마다 검증')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    parser.add_argument('--log-dir', type=str, default='logs/',
                        help='로그 디렉토리')
    
    args = parser.parse_args()
    
    # 학습 실행
    train(args)


if __name__ == "__main__":
    main()