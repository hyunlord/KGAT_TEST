"""
원본 KGAT 논문의 평가 메트릭 재현
"""
import numpy as np
import torch
from collections import defaultdict


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    """힙을 사용한 Top-K 추천"""
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    
    K_max = max(Ks)
    K_max_item_score = sorted(item_score.items(), key=lambda x: x[1], reverse=True)[:K_max]
    
    r = []
    for i in K_max_item_score:
        if i[0] in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    """AUC 계산"""
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]
    
    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc


def AUC(ground_truth, prediction):
    """Area Under Curve 계산"""
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except:
        auc = 0.
    return auc


def recall_at_k(r, k, all_pos_num):
    """Recall@K 계산"""
    r = np.asarray(r, dtype=np.float32)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    """Hit@K 계산"""
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.


def ndcg_at_k(r, k, all_pos_num):
    """NDCG@K 계산"""
    r = np.asarray(r, dtype=np.float32)[:k]
    if np.sum(r) > 0:
        return np.sum(r / np.log2(np.arange(2, len(r) + 2)))
    else:
        return 0.


def precision_at_k(r, k):
    """Precision@K 계산"""
    r = np.asarray(r)[:k]
    return np.mean(r)


def get_performance(user_pos_test, r, auc, Ks):
    """모든 메트릭 계산"""
    recall, precision, ndcg, hit_ratio = [], [], [], []
    
    for K in Ks:
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        precision.append(precision_at_k(r, K))
        ndcg.append(ndcg_at_k(r, K, len(user_pos_test)))
        hit_ratio.append(hit_at_k(r, K))
    
    return {
        'recall': np.array(recall),
        'precision': np.array(precision),
        'ndcg': np.array(ndcg),
        'hit_ratio': np.array(hit_ratio),
        'auc': auc
    }


def test_one_user(x):
    """한 사용자에 대한 평가"""
    # unpack
    rating, u, try_pos_test, test_items, Ks = x
    
    # 예측
    r, auc = ranklist_by_heapq(try_pos_test, test_items, rating, Ks)
    
    return get_performance(try_pos_test, r, auc, Ks)


def test(model, data_loader, user_list, Ks, device):
    """전체 테스트"""
    result = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks)),
        'auc': 0.
    }
    
    # 모델에서 임베딩 가져오기
    u_embed, i_embed = model()
    
    # 테스트 아이템 준비
    test_items = list(range(data_loader.n_items))
    
    count = 0
    for u in user_list:
        # 사용자가 test set에 있는지 확인
        if u not in data_loader.test_user_dict:
            continue
        
        # 학습 데이터에서 본 아이템들
        try_pos_items = data_loader.train_user_dict[u] if u in data_loader.train_user_dict else []
        
        # 테스트 아이템들
        pos_test_items = data_loader.test_user_dict[u]
        
        # 추천 점수 계산
        # 사용자 ID는 엔티티 공간에서 원래 공간으로 변환
        u_original = u - data_loader.n_entities
        u_embedding = u_embed[u_original]
        scores = torch.matmul(u_embedding, i_embed.t()).cpu().numpy()
        
        # 학습에서 본 아이템들은 제외
        for i in try_pos_items:
            scores[i] = -np.inf
        
        # 평가
        ret = test_one_user([scores, u, pos_test_items, test_items, Ks])
        
        count += 1
        for k in result:
            result[k] += ret[k]
    
    # 평균
    for k in result:
        result[k] /= count
    
    return result