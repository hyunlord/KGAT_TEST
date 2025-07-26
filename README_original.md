# KGAT 원본 논문 구현

원본 KGAT 논문의 정확한 재현을 위한 구현입니다.

## 사용법

### 1. 원본 KGAT 학습 (Original)

원본 논문의 KGAT 구현을 학습시키려면:

```bash
# 단일 GPU
python src/train_original.py

# 멀티 GPU (DDP)
./scripts/train_original_multi_gpu.sh

# 멀티 GPU with 커스텀 설정
./scripts/train_original_multi_gpu.sh 4 2048 deepspeed_stage_2
```

### 2. Fixed KGAT 학습 (PyTorch Lightning)

수정된 버전의 KGAT 학습:

```bash
# Fixed 모델 학습
bash scripts/train_fixed_multi_gpu.sh 4

# 학습 완료 후 체크포인트는 logs/ 디렉토리에 저장됨
```

### 3. 표준 vs 관계 강화 추천 비교

학습된 Original KGAT 모델을 사용하여 두 가지 추천 방식을 비교:
- **표준 방식**: user 임베딩과 item 임베딩의 유사도만 사용
- **관계 강화 방식**: user + relation 임베딩과 item 임베딩의 유사도 사용

```bash
# Original 모델용 (기존 스크립트)
./scripts/run_relation_comparison.sh models/amazon-book_kgat_bi_multi_gpu.pth 1000

# 모든 모델 지원 (Original/Fixed 자동 감지)
./scripts/run_relation_comparison_all.sh models/amazon-book_kgat_bi_multi_gpu.pth 1000

# Fixed 모델 명시적 지정
./scripts/run_relation_comparison_all.sh logs/kgat_fixed/best.ckpt 1000 fixed

# 직접 실행
python src/evaluate_relation_comparison_all.py \
    --checkpoint models/amazon-book_kgat_bi_multi_gpu.pth \
    --model-type auto \
    --n-users 1000
```

### 4. 단일 관계 기반 추천 평가

특정 관계 하나만을 사용한 타겟팅 추천:

```bash
# 모든 관계 개별 평가
./scripts/run_single_relation.sh models/amazon-book_kgat_bi_multi_gpu.pth 1000

# 특정 관계만 평가 (예: 관계 0)
./scripts/run_single_relation.sh models/amazon-book_kgat_bi_multi_gpu.pth 1000 0
```

## 파일 구조

```
src/
├── kgat_original.py              # 원본 KGAT 모델
├── data_loader_original.py       # 원본 데이터 로더
├── train_original.py             # 원본 학습 스크립트
├── evaluate_original.py          # 원본 평가 메트릭
└── evaluate_relation_comparison.py  # 표준 vs 관계 강화 비교

scripts/
├── train_original_multi_gpu.sh   # 멀티 GPU 학습 스크립트
└── run_relation_comparison.sh    # 관계 비교 실행 스크립트
```

## 주요 특징

- 원본 논문의 정확한 재현
- Bi-interaction aggregator 구현
- 사용자 ID 매핑 처리
- 희소 행렬 연산 최적화
- 멀티 GPU 지원 (DDP, DeepSpeed)

## 성능

Amazon-Book 데이터셋에서:
- Recall@20: ~8.9%
- Precision@20: ~0.8%

이는 원본 논문의 성능과 유사합니다.