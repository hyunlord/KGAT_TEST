#!/bin/bash
# Amazon-Book 데이터셋으로 KGAT 학습

echo "Amazon-Book 데이터셋으로 KGAT 학습 시작..."

# 단일 GPU 학습
python src/train.py \
    data.data_dir=data/amazon-book \
    data.batch_size=1024 \
    training.max_epochs=200 \
    training.early_stopping_patience=20 \
    training.check_val_every_n_epoch=1 \
    "$@"