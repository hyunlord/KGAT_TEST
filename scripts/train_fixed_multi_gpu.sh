#!/bin/bash

# Fixed KGAT 모델을 위한 멀티 GPU 학습 스크립트
# 사용법: ./train_fixed_multi_gpu.sh [GPU수] [배치크기] [전략]
# 예시: ./train_fixed_multi_gpu.sh 4 4096 ddp

echo "======================================"
echo "Fixed KGAT 멀티 GPU 학습"
echo "======================================"

# 기본 설정
NUM_GPUS=${1:-"-1"}  # 기본값: 모든 GPU 사용 (-1)
BATCH_SIZE=${2:-"4096"}  # 기본값: 4096
STRATEGY=${3:-"ddp"}  # 기본값: ddp (옵션: ddp, deepspeed_stage_1, deepspeed_stage_2, deepspeed_stage_3)
DATA_DIR=${4:-"data/amazon-book"}  # 기본값: amazon-book

# 환경 정보 출력
echo "설정:"
echo "  GPU 수: $NUM_GPUS"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습 전략: $STRATEGY"
echo "  데이터 디렉토리: $DATA_DIR"
echo ""

# GPU 정보 출력
if command -v nvidia-smi &> /dev/null; then
    echo "사용 가능한 GPU:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
fi

# 학습 시작
echo "Fixed KGAT 모델 멀티 GPU 학습 시작..."
echo "======================================"

python src/train_improved.py \
    model.type=kgat_fixed \
    data.data_dir=$DATA_DIR \
    data.batch_size=$BATCH_SIZE \
    training.devices=$NUM_GPUS \
    training.strategy=$STRATEGY \
    training.precision=16 \
    training.max_epochs=1000 \
    training.check_val_every_n_epoch=10 \
    training.early_stopping_patience=50 \
    model.weight_decay=1e-5 \
    model.lr=0.0001 \
    model.aggregator=bi \
    training.num_workers=4 \
    training.persistent_workers=true

echo ""
echo "학습 완료!"
echo "결과는 logs/ 디렉토리에서 확인할 수 있습니다."
echo "TensorBoard로 확인: tensorboard --logdir logs/"