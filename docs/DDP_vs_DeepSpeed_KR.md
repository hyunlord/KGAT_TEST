# DDP vs DeepSpeed 비교 가이드

## 개요

이 문서는 KGAT를 위한 분산 학습 전략을 비교합니다:
- **DDP (Distributed Data Parallel)**: PyTorch 기본 분산 학습
- **DeepSpeed**: Microsoft의 대규모 학습 최적화 라이브러리

## 빠른 비교

| 기능 | DDP | DeepSpeed Stage 1 | DeepSpeed Stage 2 | DeepSpeed Stage 3 |
|---------|-----|------------------|------------------|------------------|
| 설정 복잡도 | 간단 | 간단 | 보통 | 복잡 |
| 메모리 절약 | 기준 | 20% | 35% | 50%+ |
| 속도 | 기준 (1.0x) | 1.1x | 1.15x | 0.9x |
| 최적화 대상 | - | Optimizer State | + Gradients | + Parameters |
| CPU 오프로딩 | 미지원 | 미지원 | 선택적 | 필수 |
| 권장 모델 크기 | ~1B | ~2B | ~10B | 10B+ |

## 학습 명령어

### DDP 학습
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=ddp \
    training.precision=16 \
    data.batch_size=4096
```

### DeepSpeed Stage 1 (Optimizer 상태 분할)
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=deepspeed_stage_1 \
    training.precision=16 \
    data.batch_size=4096
```

### DeepSpeed Stage 2 (Optimizer + Gradient 분할)
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=deepspeed_stage_2 \
    training.precision=16 \
    data.batch_size=4096
```

### DeepSpeed Stage 3 (전체 분할 + CPU 오프로드)
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=deepspeed_stage_3 \
    training.precision=16 \
    data.batch_size=3072  # 약간 작은 배치 크기
```

## 성능 비교 스크립트

자동화된 비교 실행:
```bash
# 모든 전략 비교 (DDP + DeepSpeed 1,2,3)
python scripts/compare_strategies.py \
    --data-dir data/amazon-book \
    --devices 4 \
    --batch-size 2048 \
    --max-epochs 10

# 특정 전략만 비교
python scripts/compare_strategies.py \
    --strategies ddp deepspeed_stage_2 \
    --devices 4
```

## 예상 결과

### 메모리 사용량 (4x Tesla T4 GPU)
- **DDP**: ~12GB/GPU
- **DeepSpeed Stage 1**: ~10GB/GPU (20% 감소)
- **DeepSpeed Stage 2**: ~8GB/GPU (35% 감소)
- **DeepSpeed Stage 3**: ~6GB/GPU (50% 감소)

### 학습 속도
- **DDP**: 기준선 (1.0x)
- **DeepSpeed Stage 1**: 1.1x 빠름
- **DeepSpeed Stage 2**: 1.15x 빠름
- **DeepSpeed Stage 3**: 0.9x (CPU 오프로드로 인해 약간 느림)

### 언제 무엇을 사용할까?

#### DDP 사용 시기:
- 모델이 GPU 메모리에 편안하게 들어갈 때
- 가장 간단한 설정을 원할 때
- 분산 학습 문제를 디버깅할 때

#### DeepSpeed Stage 1 사용 시기:
- 모델이 메모리에 겨우 들어갈 때
- 10-20% 메모리 절약이 필요할 때
- 최소한의 코드 변경이 필요할 때

#### DeepSpeed Stage 2 사용 시기:
- 상당한 메모리 절약이 필요할 때
- 매우 큰 모델을 학습할 때
- 최고의 속도/메모리 균형을 원할 때

#### DeepSpeed Stage 3 사용 시기:
- Stage 2로도 모델이 안 들어갈 때
- 빠른 CPU-GPU 연결이 있을 때
- 초대형 모델 학습 시

## 문제 해결

### DeepSpeed 설치
```bash
# CUDA 11.x용
pip install deepspeed

# 특정 CUDA 버전용
DS_BUILD_CUDA_EXT=1 pip install deepspeed
```

### 일반적인 문제

1. **DeepSpeed에서 NCCL 오류**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_P2P_DISABLE=1  # P2P 문제 시
   ```

2. **메모리 단편화**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

3. **DeepSpeed 설정 튜닝**
   ```python
   # 커스텀 DeepSpeed 설정
   training.strategy = {
       "type": "deepspeed",
       "config": {
           "stage": 2,
           "offload_optimizer": True,
           "offload_parameters": False
       }
   }
   ```

## 모니터링

### 실시간 GPU 사용량
```bash
# GPU 메모리 및 사용률 확인
watch -n 1 nvidia-smi

# 상세 프로세스 보기
nvidia-smi pmon -i 0,1,2,3
```

### 학습 메트릭
```bash
# TensorBoard
tensorboard --logdir logs/

# http://localhost:6006에서 확인
```

## T4 GPU 서버 권장 사항 (4x T4)

1. **표준 KGAT 학습**: DDP 사용
   - 간단하고 신뢰성 높음
   - 최대 5억 파라미터 모델에 적합

2. **큰 임베딩 테이블**: DeepSpeed Stage 1
   - 임베딩 테이블이 매우 클 때
   - 최소 오버헤드로 메모리 절약

3. **메모리 제약 상황**: DeepSpeed Stage 2
   - 속도와 메모리의 최적 균형
   - 더 큰 배치 크기 가능

4. **초대형 모델**: DeepSpeed Stage 3
   - 다른 방법으로는 불가능한 모델
   - CPU 메모리를 활용한 학습

5. **실험/연구용**: 모든 전략 시도
   - 비교 스크립트로 최적 설정 찾기
   - 재현성을 위한 결과 문서화