#!/bin/bash

echo "Training Fixed KGAT Model with PyTorch Lightning..."

python src/train_improved.py \
    ++model.type=kgat_fixed \
    training.max_epochs=1000 \
    training.check_val_every_n_epoch=10 \
    training.early_stopping_patience=50 \
    data.batch_size=1024 \
    model.reg_weight=1e-5 \
    model.lr=0.0001 \
    model.aggregator=bi-interaction