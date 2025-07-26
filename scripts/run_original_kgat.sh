#!/bin/bash
# 원본 KGAT 논문의 Amazon-Book 설정으로 실행

echo "Running original KGAT implementation on Amazon-Book dataset..."

python src/train_original.py \
    --dataset amazon-book \
    --model_type kgat \
    --adj_type si \
    --alg_type bi \
    --embed_size 64 \
    --layer_size '[64,32,16]' \
    --lr 0.0001 \
    --regs '[1e-5,1e-5]' \
    --epoch 1000 \
    --batch_size 1024 \
    --node_dropout '[0.1]' \
    --mess_dropout '[0.1,0.1,0.1]' \
    --Ks '[20,40,60,80,100]' \
    --save_flag 1 \
    --test_flag part \
    --gpu_id 0 \
    --seed 2019 \
    "$@"