#!/bin/bash

# Multi-GPU training script for original KGAT implementation
# This script supports training with multiple GPUs using PyTorch's DistributedDataParallel (DDP)

# Default values
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1024
DATASET="amazon-book"
EPOCHS=1000
MASTER_PORT=29500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size_per_gpu)
            BATCH_SIZE_PER_GPU="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --num_gpus <N>           Number of GPUs to use (default: 1)"
            echo "  --batch_size_per_gpu <N> Batch size per GPU (default: 1024)"
            echo "  --dataset <name>         Dataset name: amazon-book, last-fm, yelp2018 (default: amazon-book)"
            echo "  --epochs <N>             Number of training epochs (default: 1000)"
            echo "  --master_port <N>        Master port for DDP (default: 29500)"
            echo ""
            echo "Examples:"
            echo "  # Train with 2 GPUs"
            echo "  $0 --num_gpus 2"
            echo ""
            echo "  # Train with 4 GPUs and larger batch size"
            echo "  $0 --num_gpus 4 --batch_size_per_gpu 2048"
            echo ""
            echo "  # Train on last-fm dataset with 2 GPUs"
            echo "  $0 --num_gpus 2 --dataset last-fm"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Calculate total batch sizes
TOTAL_CF_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
TOTAL_KG_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * 2 * NUM_GPUS))  # KG batch is typically 2x CF batch

echo "=========================================="
echo "Multi-GPU Training Configuration"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total CF batch size: $TOTAL_CF_BATCH_SIZE"
echo "Total KG batch size: $TOTAL_KG_BATCH_SIZE"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "Error: Failed to check CUDA availability"
    exit 1
fi

# Check if we have enough GPUs
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs/${DATASET}

# Export environment variables for DDP
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$NUM_GPUS

# Single GPU training
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running single GPU training..."
    python src/train_original.py \
        --dataset $DATASET \
        --cf_batch_size $TOTAL_CF_BATCH_SIZE \
        --kg_batch_size $TOTAL_KG_BATCH_SIZE \
        --epoch $EPOCHS \
        --gpu_id 0
else
    echo "Running multi-GPU training with DDP..."
    
    # First, we need to create the multi-GPU training script
    # Check if the multi-GPU training script exists
    if [ ! -f "src/train_original_multi_gpu.py" ]; then
        echo "Error: Multi-GPU training script not found. Creating it now..."
        
        # We'll need to create this script
        cat > src/train_original_multi_gpu.py << 'EOF'
"""
Multi-GPU training script for original KGAT implementation using DistributedDataParallel
"""
import os
import sys
import random
import logging
import argparse
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from kgat_original import KGAT
from data_loader_original import DataLoaderOriginal
from evaluate_original import test


def parse_args():
    """Parse arguments with multi-GPU support"""
    parser = argparse.ArgumentParser(description="KGAT Multi-GPU")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='amazon-book',
                        help='dataset name: amazon-book, last-fm, yelp2018')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='path of data directory')
    
    # Model
    parser.add_argument('--model_type', type=str, default='kgat',
                        help='model type: kgat, bprmf, fm, nfm, cke, cfkg')
    parser.add_argument('--adj_type', type=str, default='si',
                        help='adjacency matrix type: si, bi')
    parser.add_argument('--alg_type', type=str, default='bi',
                        help='algorithm type: bi, gcn, graphsage')
    
    # Hyperparameters
    parser.add_argument('--embed_size', type=int, default=64,
                        help='embedding size')
    parser.add_argument('--layer_size', type=str, default='[64, 32, 16]',
                        help='layer sizes')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--regs', nargs='?', default='[1e-5, 1e-5]',
                        help='regularization coefficients')
    
    # Training settings
    parser.add_argument('--epoch', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='test batch size')
    
    # Dropout
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='node dropout per layer')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='message dropout per layer')
    
    # Evaluation
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='k values for evaluation')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='save model or not')
    parser.add_argument('--test_flag', type=str, default='part',
                        help='test flag: part, full')
    parser.add_argument('--report_flag', type=int, default=0,
                        help='report flag')
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='use pretrained embeddings')
    parser.add_argument('--pretrain_embedding_dir', type=str, default='pretrain/',
                        help='pretrained embeddings directory')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed')
    
    # Multi-GPU specific
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank for distributed training')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id (for single GPU)')
    
    args = parser.parse_args()
    
    # Convert string to list
    args.layer_size = eval(args.layer_size)
    args.regs = eval(args.regs)
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.Ks = eval(args.Ks)
    
    return args


def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def setup_logger(args, rank):
    """Set up logger for distributed training"""
    log_path = os.path.join('logs', args.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    # Only rank 0 writes to file
    handlers = []
    if rank == 0:
        handlers.append(logging.FileHandler(os.path.join(log_path, 'train_multi_gpu.log')))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        format=f'[Rank {rank}] %(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=handlers
    )
    
    return logging.getLogger()


def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_ddp(rank, world_size, args):
    """Training process for each GPU"""
    # Set up distributed training
    setup(rank, world_size)
    
    # Set seed
    set_seed(args.seed + rank)
    
    # Set up logger
    logger = setup_logger(args, rank)
    if rank == 0:
        logger.info(args)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    logger.info(f'Process {rank} using device: {device}')
    
    # Load data (only rank 0 logs)
    if rank == 0:
        logger.info('Loading data...')
    data_loader = DataLoaderOriginal(args, logger if rank == 0 else None)
    
    # Initialize model
    if rank == 0:
        logger.info('Initializing model...')
    model = KGAT(
        args,
        data_loader.n_users,
        data_loader.n_items,
        data_loader.n_entities,
        data_loader.n_relations,
        data_loader.adjacency_dict['plain_adj'],
        data_loader.laplacian_dict['kg_mat']
    )
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        logger.info(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Tensorboard (only rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(f'runs/{args.dataset}_{args.model_type}_{args.alg_type}_multi_gpu')
    
    # Training loop
    if rank == 0:
        logger.info('Starting distributed training...')
    
    best_recall = 0.0
    best_epoch = 0
    
    # Calculate batch size per GPU
    cf_batch_size_per_gpu = args.cf_batch_size // world_size
    
    for epoch in range(args.epoch):
        t1 = time()
        
        # Training
        model.train()
        
        # CF training
        cf_total_loss = 0.0
        n_cf_batch = data_loader.n_cf_train // args.cf_batch_size + 1
        
        # Prepare user list
        user_list = list(data_loader.train_user_dict.keys())
        
        # Ensure each process gets different random shuffle
        random.Random(args.seed + epoch + rank).shuffle(user_list)
        
        # Distribute batches across GPUs
        for iter in range(n_cf_batch):
            # Each GPU processes its portion of the batch
            start_idx = (iter * args.cf_batch_size + rank * cf_batch_size_per_gpu)
            end_idx = min(start_idx + cf_batch_size_per_gpu, len(user_list))
            
            if start_idx >= len(user_list):
                continue
            
            batch_user = user_list[start_idx:end_idx]
            if len(batch_user) == 0:
                continue
            
            batch_pos_item = []
            batch_neg_item = []
            
            for u in batch_user:
                pos_item = data_loader.sample_pos_items_for_u(u, 1)[0]
                neg_item = data_loader.sample_neg_items_for_u(u, 1)[0]
                batch_pos_item.append(pos_item)
                batch_neg_item.append(neg_item)
            
            # Convert to tensors
            batch_user = torch.LongTensor(batch_user).to(device)
            batch_pos_item = torch.LongTensor(batch_pos_item).to(device)
            batch_neg_item = torch.LongTensor(batch_neg_item).to(device)
            
            # Forward pass
            u_embed, i_embed = model.module()
            
            u_embed = u_embed[batch_user]
            pos_embed = i_embed[batch_pos_item]
            neg_embed = i_embed[batch_neg_item]
            
            # Calculate loss
            mf_loss, emb_loss, reg_loss = model.module.create_bpr_loss(
                u_embed, pos_embed, neg_embed
            )
            loss = mf_loss + emb_loss + reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cf_total_loss += loss.item()
        
        # Synchronize and aggregate loss across GPUs
        cf_total_loss_tensor = torch.tensor(cf_total_loss).to(device)
        dist.all_reduce(cf_total_loss_tensor, op=dist.ReduceOp.SUM)
        cf_total_loss = cf_total_loss_tensor.item() / world_size
        
        # Evaluation (only on rank 0)
        if (epoch + 1) % 10 == 0 and rank == 0:
            model.eval()
            
            with torch.no_grad():
                u_embed, i_embed = model.module()
                
                # Test
                ret = test(
                    model.module, data_loader, user_list[:5000],
                    args.Ks, device
                )
                
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f], recall=[%.5f, %.5f], ' \
                          'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                          (epoch, time() - t1, cf_total_loss,
                           ret['recall'][0], ret['recall'][-1],
                           ret['precision'][0], ret['precision'][-1],
                           ret['hit_ratio'][0], ret['hit_ratio'][-1],
                           ret['ndcg'][0], ret['ndcg'][-1])
                
                logger.info(perf_str)
                
                # Tensorboard logging
                if writer:
                    writer.add_scalar('Loss/train', cf_total_loss, epoch)
                    writer.add_scalar('Eval/recall@20', ret['recall'][0], epoch)
                    writer.add_scalar('Eval/ndcg@20', ret['ndcg'][0], epoch)
                
                # Save best model
                if ret['recall'][0] > best_recall:
                    best_recall = ret['recall'][0]
                    best_epoch = epoch
                    
                    if args.save_flag:
                        save_path = f'models/{args.dataset}_{args.model_type}_{args.alg_type}_multi_gpu.pth'
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(model.module.state_dict(), save_path)
                        logger.info(f'Model saved to {save_path}')
        
        # Synchronize at the end of each epoch
        dist.barrier()
    
    if rank == 0:
        logger.info(f'Best recall@20: {best_recall:.5f} at epoch {best_epoch}')
        if writer:
            writer.close()
    
    cleanup()


def main():
    args = parse_args()
    
    # Get world size from environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size == 1:
        # Single GPU training - use original script
        print("Single GPU training detected. Use train_original.py instead.")
        sys.exit(1)
    else:
        # Multi-GPU training
        mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
EOF
    fi
    
    # Launch distributed training
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=localhost \
        --master_port=$MASTER_PORT \
        src/train_original_multi_gpu.py \
        --dataset $DATASET \
        --cf_batch_size $TOTAL_CF_BATCH_SIZE \
        --kg_batch_size $TOTAL_KG_BATCH_SIZE \
        --epoch $EPOCHS
fi

echo "Training completed!"