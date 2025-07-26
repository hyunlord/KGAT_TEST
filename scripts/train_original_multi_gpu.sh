#!/bin/bash

# Original KGAT 멀티 GPU 학습 스크립트
# 사용법: ./train_original_multi_gpu.sh [GPU수] [배치크기] [전략]

echo "======================================"
echo "Original KGAT 멀티 GPU 학습"
echo "======================================"

# 기본 설정
NUM_GPUS=${1:-"-1"}  # 기본값: 모든 GPU 사용 (-1)
BATCH_SIZE=${2:-"2048"}  # 기본값: 2048
STRATEGY=${3:-"ddp"}  # 기본값: ddp

# 환경 정보 출력
echo "설정:"
echo "  GPU 수: $NUM_GPUS"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습 전략: $STRATEGY"
echo ""

# GPU 정보 출력
if command -v nvidia-smi &> /dev/null; then
    echo "사용 가능한 GPU:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
fi

# 학습 시작
echo "Original KGAT 모델 멀티 GPU 학습 시작..."
echo "======================================"

# DDP 설정 (PyTorch 기본)
if [ "$NUM_GPUS" = "-1" ]; then
    # 모든 GPU 사용
    python -m torch.distributed.launch --nproc_per_node=auto \
        src/train_original.py \
        --multi_gpu \
        --batch_size $BATCH_SIZE
else
    # 특정 수의 GPU 사용
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
        src/train_original.py \
        --multi_gpu \
        --batch_size $BATCH_SIZE
fi

echo ""
echo "학습 완료!"
echo "결과는 models/ 디렉토리에서 확인할 수 있습니다."