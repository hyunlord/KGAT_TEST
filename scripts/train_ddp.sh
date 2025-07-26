#!/bin/bash
# DDP 학습을 위한 스크립트

# CUDA 디바이스 설정 (경고 제거)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# PyTorch 분산 학습 설정
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# 학습 실행
python src/train.py \
    training.devices=4 \
    training.strategy=ddp \
    data.batch_size=2048 \
    "$@"  # 추가 인자 전달