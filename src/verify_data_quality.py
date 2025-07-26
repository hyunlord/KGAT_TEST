"""
데이터 품질 검증 스크립트
"""
import os
from collections import defaultdict


def verify_data_split(data_dir):
    """Train/Test 분할이 올바른지 검증"""
    train_file = os.path.join(data_dir, 'train.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    
    # Train 데이터 로드
    train_items = set()
    train_user_items = defaultdict(list)
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user = int(parts[0])
                items = [int(item) for item in parts[1:]]
                train_user_items[user] = items
                train_items.update(items)
    
    # Test 데이터 로드
    test_items = set()
    test_user_items = defaultdict(list)
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user = int(parts[0])
                items = [int(item) for item in parts[1:]]
                test_user_items[user] = items
                test_items.update(items)
    
    # 통계
    print(f"=== {data_dir} 데이터 검증 ===")
    print(f"Train 사용자: {len(train_user_items)}")
    print(f"Test 사용자: {len(test_user_items)}")
    print(f"Train 아이템: {len(train_items)}")
    print(f"Test 아이템: {len(test_items)}")
    
    # 중요: Test 아이템이 Train에 있는지 확인
    overlap = train_items & test_items
    print(f"\n겹치는 아이템: {len(overlap)}")
    print(f"Test 아이템 중 Train에 있는 비율: {len(overlap) / len(test_items) * 100:.1f}%")
    
    # Cold start 아이템 (학습에 없는 테스트 아이템)
    cold_items = test_items - train_items
    print(f"Cold start 아이템: {len(cold_items)} ({len(cold_items) / len(test_items) * 100:.1f}%)")
    
    # 사용자 겹침 확인
    common_users = set(train_user_items.keys()) & set(test_user_items.keys())
    print(f"\n공통 사용자: {len(common_users)}")
    
    # 품질 판단
    print("\n=== 데이터 품질 평가 ===")
    if len(overlap) / len(test_items) < 0.8:
        print("❌ 경고: Test 아이템의 80% 미만이 Train에 있습니다!")
        print("   모델이 학습하지 않은 아이템을 추천할 수 없습니다.")
    else:
        print("✅ 좋음: 대부분의 Test 아이템이 Train에 있습니다.")
    
    if len(common_users) == 0:
        print("❌ 경고: Train과 Test에 공통 사용자가 없습니다!")
    else:
        print(f"✅ 좋음: {len(common_users)}명의 공통 사용자가 있습니다.")
    
    return len(overlap) / len(test_items) if test_items else 0


if __name__ == "__main__":
    # 원본 데이터 검증
    if os.path.exists('data/amazon-book'):
        print("원본 데이터:")
        orig_quality = verify_data_split('data/amazon-book')
    
    # 수정된 데이터 검증
    if os.path.exists('data/amazon-book-fixed'):
        print("\n\n수정된 데이터:")
        fixed_quality = verify_data_split('data/amazon-book-fixed')
        
        if 'orig_quality' in locals():
            print(f"\n개선도: {(fixed_quality - orig_quality) * 100:.1f}% 포인트")