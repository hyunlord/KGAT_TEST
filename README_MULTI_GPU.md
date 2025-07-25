# KGAT 멀티 GPU 학습 가이드

이 가이드는 PyTorch Lightning을 사용하여 KGAT를 멀티 GPU로 학습하는 방법을 설명합니다.

## 빠른 시작

### 1. 모든 GPU 사용
```bash
# 멀티 GPU 설정 파일 사용
python src/train.py --config-name config_multi_gpu

# 또는 devices 파라미터 수정
python src/train.py training.devices=-1
```

### 2. 특정 개수의 GPU 사용
```bash
# 2개 GPU 사용
python src/train.py training.devices=2

# 특정 GPU ID 사용
python src/train.py training.devices=[0,2,3]
```

### 3. 편의 스크립트 사용
```bash
./scripts/train_multi_gpu.sh
```

## 설정 옵션

### 분산 학습 전략

1. **DDP (Distributed Data Parallel)** - 권장
   ```yaml
   training:
     strategy: ddp
     devices: -1  # 모든 GPU
   ```

2. **DDP Spawn** - 디버깅용
   ```yaml
   training:
     strategy: ddp_spawn
     devices: 4
   ```

3. **DeepSpeed** - 대규모 모델용
   ```yaml
   training:
     strategy: deepspeed_stage_2
     precision: 16
   ```

### 배치 크기 스케일링

멀티 GPU 사용 시 유효 배치 크기:
```
유효_배치_크기 = batch_size * GPU_개수 * accumulate_grad_batches
```

예시 설정:
- 1 GPU: batch_size=1024
- 4 GPU: batch_size=4096 (또는 1024 유지하고 DDP가 처리하도록 함)

### 학습률 스케일링

코드는 선형 스케일링 규칙으로 학습률을 자동 조정합니다:
```
유효_학습률 = 기본_학습률 * GPU_개수
```

## 성능 팁

1. **혼합 정밀도 사용**
   ```yaml
   training:
     precision: 16  # 메모리 절약, 더 빠른 학습
   ```

2. **배치 크기 증가**
   ```yaml
   data:
     batch_size: 4096  # 4 GPU용
   ```

3. **더 많은 워커**
   ```yaml
   data:
     num_workers: 16  # GPU당 4개
   ```

4. **그래디언트 누적** (메모리 제한 시)
   ```yaml
   training:
     accumulate_grad_batches: 4
   ```

## 모니터링

### GPU 사용량 확인
```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 사용 중인 GPU 확인
echo $CUDA_VISIBLE_DEVICES
```

### TensorBoard
```bash
tensorboard --logdir logs/
```

## 문제 해결

### 메모리 부족
- batch_size 감소
- 그래디언트 누적 사용
- precision=16 사용
- deepspeed 전략 시도

### 느린 데이터 로딩
- num_workers 증가
- 데이터 저장소에 SSD 사용
- 데이터 전처리

### 불균등한 GPU 사용
- batch_size가 GPU 개수로 나누어떨어지는지 확인
- 데이터 분산 확인

## 예시 명령어

### T4 GPU 서버 (현재 설정)
```bash
# 4개 T4 GPU 모두 사용
python src/train.py \
    --config-name config_multi_gpu \
    data.dataset=amazon-book \
    data.batch_size=2048 \
    training.devices=4

# 더 큰 모델로 2개 GPU 사용
python src/train.py \
    training.devices=2 \
    model.embed_dim=128 \
    model.layer_dims=[64,32,16]
```

### 메모리 제약 학습
```bash
python src/train.py \
    training.devices=4 \
    training.strategy=deepspeed_stage_2 \
    training.precision=16 \
    training.accumulate_grad_batches=4 \
    data.batch_size=512
```

## 예상 성능 향상

4개 T4 GPU 사용 시:
- 단일 GPU 대비 ~3.5배 속도 향상 (통신 오버헤드로 인해)
- 더 큰 배치 크기 사용 가능
- 적절한 학습률 스케일링으로 더 빠른 수렴

## 분산 학습 출력

학습 시작 시 다음과 같이 표시됩니다:
```
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
4 GPU용으로 학습률을 0.001에서 0.004로 스케일링
```