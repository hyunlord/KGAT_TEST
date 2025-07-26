# Fixed KGAT 모델 가이드

## 개요
Fixed KGAT는 원본 KGAT 논문의 구현을 PyTorch Lightning으로 수정한 버전입니다.

## 주요 특징
- PyTorch Lightning 기반
- 멀티 GPU 지원 (DDP, DeepSpeed)
- 원본과 동일한 성능 목표
- Bi-interaction aggregator 구현

## 학습 방법

### 1. 기본 학습
```bash
bash scripts/train_fixed_multi_gpu.sh 4
```

### 2. 커스텀 설정
```bash
python src/train_improved.py \
    ++model.type=kgat_fixed \
    ++model.embedding_size=64 \
    ++model.layer_sizes=[64,32,16] \
    model.aggregator=bi-interaction \
    ++model.dropout_rates=[0.1,0.1,0.1] \
    model.reg_weight=1e-5 \
    model.lr=0.0001 \
    data.batch_size=4096 \
    training.devices=4 \
    training.strategy=ddp
```

## 설정 파라미터

### 필수 파라미터
- `model.type`: "kgat_fixed"
- `model.embedding_size`: 임베딩 차원 (기본: 64)
- `model.layer_sizes`: 각 레이어 크기 (기본: [64,32,16])
- `model.aggregator`: "bi-interaction", "gcn", "graphsage"
- `model.dropout_rates`: 각 레이어별 드롭아웃 (기본: [0.1,0.1,0.1])

### 학습 파라미터
- `model.lr`: 학습률 (기본: 0.0001)
- `model.reg_weight`: 정규화 가중치 (기본: 1e-5)
- `data.batch_size`: 배치 크기
- `training.devices`: GPU 개수
- `training.strategy`: "ddp", "deepspeed_stage_1" 등

## 평가 방법

### 관계 비교 평가
```bash
# Fixed 모델 평가
./scripts/run_relation_comparison_all.sh logs/kgat_fixed/best.ckpt 1000 fixed

# 자동 감지
./scripts/run_relation_comparison_all.sh logs/kgat_fixed/best.ckpt 1000 auto
```

### 체크포인트 위치
학습된 모델은 다음 위치에 저장됩니다:
- `logs/kgat_baseline/version_*/checkpoints/*.ckpt`
- `models/` (수동 저장 시)

## 주의사항

1. **Hydra 설정**:
   - 새로운 키는 `++` 사용
   - 기존 키는 `+` 없이 사용
   - 배열은 `[값1,값2,값3]` 형식

2. **메모리 관리**:
   - 큰 배치 크기 사용 시 gradient accumulation 고려
   - Mixed precision (16-bit) 사용 권장

3. **성능 최적화**:
   - DDP가 일반적으로 가장 안정적
   - DeepSpeed는 매우 큰 모델에 유용