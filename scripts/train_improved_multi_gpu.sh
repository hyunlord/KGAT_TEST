#!/bin/bash
# 개선된 KGAT 모델을 멀티 GPU로 학습

echo "Training Improved KGAT with Multi-GPU..."
echo "GPUs: ${1:-4}"
echo "Batch Size: ${2:-2048}"

python src/train_improved.py \
    use_improved_model=true \
    training.devices=${1:-4} \
    training.strategy=ddp \
    data.batch_size=${2:-2048} \
    training.precision=16 \
    training.gradient_clip_val=5.0 \
    "${@:3}"  # 추가 인자 전달

# 사용 예시:
# bash scripts/train_improved_multi_gpu.sh 4 2048  # 4 GPU, batch size 2048
# bash scripts/train_improved_multi_gpu.sh 8 4096  # 8 GPU, batch size 4096