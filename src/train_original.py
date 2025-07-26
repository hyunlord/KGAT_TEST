"""
원본 KGAT 논문의 학습 프로세스 재현
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
from torch.utils.tensorboard import SummaryWriter

from kgat_original import KGAT
from data_loader_original import DataLoaderOriginal
from evaluate_original import test


def parse_args():
    """원본 논문의 하이퍼파라미터"""
    parser = argparse.ArgumentParser(description="KGAT")
    
    # 데이터셋
    parser.add_argument('--dataset', type=str, default='amazon-book',
                        help='dataset name: amazon-book, last-fm, yelp2018')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='path of data directory')
    
    # 모델
    parser.add_argument('--model_type', type=str, default='kgat',
                        help='model type: kgat, bprmf, fm, nfm, cke, cfkg')
    parser.add_argument('--adj_type', type=str, default='si',
                        help='adjacency matrix type: si, bi')
    parser.add_argument('--alg_type', type=str, default='bi',
                        help='algorithm type: bi, gcn, graphsage')
    
    # 하이퍼파라미터 (원본 논문 값)
    parser.add_argument('--embed_size', type=int, default=64,
                        help='embedding size')
    parser.add_argument('--layer_size', type=str, default='[64, 32, 16]',
                        help='layer sizes')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--regs', nargs='?', default='[1e-5, 1e-5]',
                        help='regularization coefficients')
    
    # 학습 설정
    parser.add_argument('--epoch', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='CF batch size')
    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='test batch size')
    
    # 드롭아웃
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='node dropout per layer')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='message dropout per layer')
    
    # 기타
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
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed')
    
    args = parser.parse_args()
    
    # 문자열을 리스트로 변환
    args.layer_size = eval(args.layer_size)
    args.regs = eval(args.regs)
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.Ks = eval(args.Ks)
    
    return args


def setup_logger(args):
    """로거 설정"""
    log_path = os.path.join('logs', args.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(log_path, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()


def set_seed(seed):
    """랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    """학습 프로세스"""
    # 시드 설정
    set_seed(args.seed)
    
    # 로거 설정
    logger = setup_logger(args)
    logger.info(args)
    
    # GPU 설정
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    # 데이터 로드
    logger.info('Loading data...')
    data_loader = DataLoaderOriginal(args, logger)
    
    # 모델 초기화
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
    
    logger.info(model)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Tensorboard
    writer = SummaryWriter(f'runs/{args.dataset}_{args.model_type}_{args.alg_type}')
    
    # 학습 루프
    logger.info('Starting training...')
    
    best_recall = 0.0
    best_epoch = 0
    
    for epoch in range(args.epoch):
        t1 = time()
        
        # 학습
        model.train()
        
        # CF 부분 학습
        cf_total_loss = 0.0
        n_cf_batch = data_loader.n_cf_train // args.cf_batch_size + 1
        
        # 사용자 리스트 준비
        user_list = list(data_loader.train_user_dict.keys())
        random.shuffle(user_list)
        
        for iter in range(n_cf_batch):
            # 배치 준비
            batch_user = user_list[iter * args.cf_batch_size:(iter + 1) * args.cf_batch_size]
            batch_pos_item = []
            batch_neg_item = []
            
            for u in batch_user:
                pos_item = data_loader.sample_pos_items_for_u(u, 1)[0]
                neg_item = data_loader.sample_neg_items_for_u(u, 1)[0]
                batch_pos_item.append(pos_item)
                batch_neg_item.append(neg_item)
            
            # 텐서로 변환
            batch_user = torch.LongTensor(batch_user).to(device)
            batch_pos_item = torch.LongTensor(batch_pos_item).to(device)
            batch_neg_item = torch.LongTensor(batch_neg_item).to(device)
            
            # Forward
            u_embed, i_embed = model()
            
            u_embed = u_embed[batch_user]
            pos_embed = i_embed[batch_pos_item]
            neg_embed = i_embed[batch_neg_item]
            
            # Loss 계산
            mf_loss, emb_loss, reg_loss = model.create_bpr_loss(
                u_embed, pos_embed, neg_embed
            )
            loss = mf_loss + emb_loss + reg_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cf_total_loss += loss.item()
        
        # 평가
        if (epoch + 1) % 10 == 0:
            model.eval()
            
            with torch.no_grad():
                u_embed, i_embed = model()
                
                # 테스트
                ret = test(
                    model, data_loader, user_list[:5000],  # 일부만 테스트
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
                
                # Tensorboard 로깅
                writer.add_scalar('Loss/train', cf_total_loss, epoch)
                writer.add_scalar('Eval/recall@20', ret['recall'][0], epoch)
                writer.add_scalar('Eval/ndcg@20', ret['ndcg'][0], epoch)
                
                # Best model 저장
                if ret['recall'][0] > best_recall:
                    best_recall = ret['recall'][0]
                    best_epoch = epoch
                    
                    if args.save_flag:
                        save_path = f'models/{args.dataset}_{args.model_type}_{args.alg_type}.pth'
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(model.state_dict(), save_path)
                        logger.info(f'Model saved to {save_path}')
    
    logger.info(f'Best recall@20: {best_recall:.5f} at epoch {best_epoch}')
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)