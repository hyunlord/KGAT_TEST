"""
올바른 train/test 분할을 위한 스크립트
추천 시스템에서는 사용자별로 상호작용을 나누어야 함
"""
import os
import numpy as np
from collections import defaultdict
import argparse


def create_proper_train_test_split(input_file, output_dir, test_ratio=0.2, min_items_per_user=10):
    """
    올바른 train/test 분할 생성
    
    Args:
        input_file: 전체 user-item 상호작용 파일
        output_dir: 출력 디렉토리
        test_ratio: 테스트 데이터 비율
        min_items_per_user: 사용자당 최소 아이템 수
    """
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    user_items = defaultdict(list)
    all_items = set()
    
    print(f"데이터 로드 중: {input_file}")
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user = int(parts[0])
                items = [int(item) for item in parts[1:]]
                user_items[user].extend(items)
                all_items.update(items)
    
    print(f"총 사용자 수: {len(user_items)}")
    print(f"총 아이템 수: {len(all_items)}")
    
    # 필터링: 최소 아이템 수를 가진 사용자만
    filtered_users = {u: items for u, items in user_items.items() 
                     if len(items) >= min_items_per_user}
    print(f"필터링 후 사용자 수: {len(filtered_users)}")
    
    # Train/Test 분할
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    
    for user, items in filtered_users.items():
        # 중복 제거 및 셔플
        unique_items = list(set(items))
        np.random.shuffle(unique_items)
        
        # 사용자별로 분할
        n_test = max(1, int(len(unique_items) * test_ratio))
        test_items = unique_items[:n_test]
        train_items = unique_items[n_test:]
        
        if len(train_items) > 0 and len(test_items) > 0:
            train_data[user] = train_items
            test_data[user] = test_items
    
    # 통계
    train_items_set = set()
    test_items_set = set()
    for items in train_data.values():
        train_items_set.update(items)
    for items in test_data.values():
        test_items_set.update(items)
    
    print(f"\n=== 분할 통계 ===")
    print(f"Train 사용자 수: {len(train_data)}")
    print(f"Test 사용자 수: {len(test_data)}")
    print(f"Train 고유 아이템 수: {len(train_items_set)}")
    print(f"Test 고유 아이템 수: {len(test_items_set)}")
    print(f"겹치는 아이템 수: {len(train_items_set & test_items_set)}")
    print(f"Test 아이템 중 Train에 있는 비율: {len(train_items_set & test_items_set) / len(test_items_set) * 100:.1f}%")
    
    # 파일 저장
    train_file = os.path.join(output_dir, 'train.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    with open(train_file, 'w') as f:
        for user in sorted(train_data.keys()):
            items = ' '.join(map(str, train_data[user]))
            f.write(f"{user} {items}\n")
    
    with open(test_file, 'w') as f:
        for user in sorted(test_data.keys()):
            items = ' '.join(map(str, test_data[user]))
            f.write(f"{user} {items}\n")
    
    print(f"\n파일 저장 완료:")
    print(f"  {train_file}")
    print(f"  {test_file}")
    
    # 샘플 KG 데이터도 생성 (선택사항)
    kg_file = os.path.join(output_dir, 'kg_final.txt')
    if not os.path.exists(kg_file):
        print("\nKG 데이터 생성 중...")
        with open(kg_file, 'w') as f:
            # 간단한 KG 관계 생성 (아이템 간 유사도)
            n_relations = 5
            n_kg_triples = len(train_items_set) * 10
            
            for _ in range(n_kg_triples):
                head = np.random.choice(list(train_items_set))
                tail = np.random.choice(list(train_items_set))
                if head != tail:
                    relation = np.random.randint(0, n_relations)
                    f.write(f"{head} {relation} {tail}\n")
        print(f"  {kg_file}")


def create_realistic_data(output_dir, n_users=10000, n_items=5000, 
                         interactions_per_user=(10, 200), 
                         popularity_alpha=0.5):
    """
    더 현실적인 추천 시스템 데이터 생성
    
    Args:
        output_dir: 출력 디렉토리
        n_users: 사용자 수
        n_items: 아이템 수
        interactions_per_user: 사용자당 상호작용 수 범위
        popularity_alpha: 인기도 분포 파라미터 (낮을수록 더 치우침)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("현실적인 추천 데이터 생성 중...")
    
    # 아이템 인기도 분포 (power law)
    item_popularity = np.random.power(popularity_alpha, n_items)
    item_popularity = item_popularity / item_popularity.sum()
    
    # 전체 상호작용 생성
    all_interactions = defaultdict(list)
    
    for user in range(n_users):
        # 사용자별 상호작용 수
        n_interactions = np.random.randint(*interactions_per_user)
        
        # 인기도 기반 아이템 샘플링
        items = np.random.choice(n_items, size=n_interactions, 
                               replace=True, p=item_popularity)
        
        # 중복 제거
        unique_items = list(set(items))
        if len(unique_items) >= 5:  # 최소 5개 이상
            all_interactions[user] = unique_items
    
    # Train/Test 분할
    create_proper_train_test_split_from_dict(all_interactions, output_dir)


def create_proper_train_test_split_from_dict(user_items_dict, output_dir, test_ratio=0.2):
    """딕셔너리에서 train/test 분할"""
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    
    for user, items in user_items_dict.items():
        if len(items) < 2:
            continue
            
        # 시간순 정렬 시뮬레이션 (랜덤 셔플)
        items_copy = items.copy()
        np.random.shuffle(items_copy)
        
        # 마지막 20%를 테스트로
        n_test = max(1, int(len(items_copy) * test_ratio))
        test_items = items_copy[-n_test:]
        train_items = items_copy[:-n_test]
        
        if len(train_items) > 0 and len(test_items) > 0:
            train_data[user] = train_items
            test_data[user] = test_items
    
    # 파일 저장
    save_data_files(train_data, test_data, output_dir)
    
    # KG 데이터 생성
    create_kg_data(train_data, output_dir)


def save_data_files(train_data, test_data, output_dir):
    """데이터 파일 저장"""
    train_file = os.path.join(output_dir, 'train.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    with open(train_file, 'w') as f:
        for user in sorted(train_data.keys()):
            items = ' '.join(map(str, train_data[user]))
            f.write(f"{user} {items}\n")
    
    with open(test_file, 'w') as f:
        for user in sorted(test_data.keys()):
            items = ' '.join(map(str, test_data[user]))
            f.write(f"{user} {items}\n")
    
    # 통계 출력
    train_items = set()
    test_items = set()
    for items in train_data.values():
        train_items.update(items)
    for items in test_data.values():
        test_items.update(items)
    
    print(f"\n=== 데이터 통계 ===")
    print(f"Train 사용자: {len(train_data)}")
    print(f"Test 사용자: {len(test_data)}")
    print(f"Train 아이템: {len(train_items)}")
    print(f"Test 아이템: {len(test_items)}")
    print(f"겹치는 아이템: {len(train_items & test_items)} ({len(train_items & test_items) / len(test_items) * 100:.1f}%)")


def create_kg_data(train_data, output_dir, n_relations=5):
    """KG 데이터 생성"""
    kg_file = os.path.join(output_dir, 'kg_final.txt')
    
    # Train 아이템 수집
    all_items = set()
    for items in train_data.values():
        all_items.update(items)
    
    items_list = list(all_items)
    n_triples = len(items_list) * 20
    
    with open(kg_file, 'w') as f:
        for _ in range(n_triples):
            head = np.random.choice(items_list)
            tail = np.random.choice(items_list)
            if head != tail:
                relation = np.random.randint(0, n_relations)
                f.write(f"{head} {relation} {tail}\n")
    
    print(f"KG 트리플 생성: {n_triples}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['split', 'generate'], default='generate',
                       help='split: 기존 데이터 분할, generate: 새 데이터 생성')
    parser.add_argument('--input-file', type=str, 
                       help='입력 파일 (split 모드)')
    parser.add_argument('--output-dir', type=str, default='data/amazon-book-fixed',
                       help='출력 디렉토리')
    parser.add_argument('--n-users', type=int, default=5000,
                       help='사용자 수 (generate 모드)')
    parser.add_argument('--n-items', type=int, default=3000,
                       help='아이템 수 (generate 모드)')
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        if not args.input_file:
            raise ValueError("split 모드에서는 --input-file이 필요합니다")
        create_proper_train_test_split(args.input_file, args.output_dir)
    else:
        create_realistic_data(args.output_dir, args.n_users, args.n_items)