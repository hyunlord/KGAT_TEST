# KGAT PyTorch Lightning 구현

추천 시스템을 위한 Knowledge Graph Attention Network (KGAT)의 PyTorch Lightning 기반 구현입니다. 표준 추천 방식과 관계 기반 향상 추천 방식을 비교하는 평가 방법을 포함합니다.

## 주요 기능

- ✅ PyTorch Lightning 기반 구현
- ✅ DataModule을 활용한 모듈식 아키텍처
- ✅ Hydra 설정 관리
- ✅ TensorBoard 로깅
- ✅ 모델 체크포인트 및 조기 종료
- ✅ 두 가지 평가 방법 비교:
  - 표준: 사용자-아이템 유사도만 사용
  - 향상: 사용자+관계-아이템 유사도 사용

## 설치

```bash
# 기본 설치
pip install -r requirements.txt

# DeepSpeed 지원 (선택사항)
pip install deepspeed

# GPU 모니터링 (선택사항)
pip install gputil
```

## 빠른 시작

### 1. 데이터 준비

```bash
# 테스트용 샘플 데이터 생성
python scripts/download_data.py --sample-only

# 특정 데이터셋 다운로드 (지침이 제공됨)
python scripts/download_data.py --dataset amazon-book
```

### 2. KGAT 모델 학습

#### 단일 GPU 학습
```bash
# 기본 설정으로 학습
python src/train.py

# 사용자 정의 설정으로 학습
python src/train.py data.batch_size=512 model.embed_dim=128

# 작은 설정으로 학습 (테스트용)
python src/train.py --config-name config_small
```

#### 멀티 GPU 학습
```bash
# DDP로 모든 GPU 사용 (권장)
python src/train.py training.devices=-1 training.strategy=ddp

# 특정 개수의 GPU 사용
python src/train.py training.devices=4 training.strategy=ddp data.batch_size=4096

# 멀티 GPU 설정 파일 사용
python src/train.py --config-name config_multi_gpu

# 더 나은 메모리 효율을 위한 DeepSpeed 사용
python src/train.py training.devices=-1 training.strategy=deepspeed_stage_2 training.precision=16
```

#### 전체 학습 예제
```bash
# 먼저 데이터셋 다운로드
python scripts/download_data.py --dataset amazon-book

# 4개 GPU로 Amazon-Book 데이터셋 학습
python src/train.py \
    data.data_dir=data/amazon-book \
    data.batch_size=4096 \
    training.devices=4 \
    training.strategy=ddp \
    training.precision=16 \
    training.max_epochs=200
```

### 3. 평가 및 방법 비교

```bash
# 표준 vs 향상 방법 비교
python src/evaluate_comparison.py \
    --checkpoint logs/kgat_amazon-book/version_0/checkpoints/best.ckpt \
    --n-sample-users 20
```

## 프로젝트 구조

```
KGAT_TEST/
├── src/
│   ├── kgat_lightning.py      # PyTorch Lightning KGAT 모델
│   ├── data_module.py         # 데이터 로딩 및 전처리
│   ├── train.py               # 학습 스크립트
│   ├── evaluator.py           # 평가 방법
│   ├── compare_methods.py     # 방법 비교 유틸리티
│   └── evaluate_comparison.py # 학습된 모델 방법 비교
├── configs/
│   ├── config.yaml            # 주요 설정
│   └── config_small.yaml      # 테스트용 작은 설정
├── scripts/
│   └── download_data.py       # 데이터 다운로드 스크립트
├── data/                      # 데이터셋 디렉토리
├── logs/                      # TensorBoard 로그
├── models/                    # 저장된 모델
└── results/                   # 평가 결과
```

## 설정

`configs/config.yaml`의 주요 설정 옵션:

```yaml
data:
  data_dir: data/amazon-book
  batch_size: 1024
  
model:
  embed_dim: 64
  layer_dims: [32, 16]
  aggregator: bi-interaction
  
training:
  max_epochs: 1000
  early_stopping_patience: 20
  lr: 0.001
```

## 데이터 형식

데이터 디렉토리에 필요한 파일:
- `train.txt`: 사용자-아이템 상호작용 (형식: `user_id item_id1 item_id2 ...`)
- `test.txt`: 테스트 상호작용 (동일한 형식)
- `kg_final.txt`: 지식 그래프 트리플 (형식: `head_entity relation_id tail_entity`)

## 학습 모니터링

### TensorBoard 설정
```bash
# TensorBoard 설치 (이미 requirements.txt에 포함)
pip install tensorboard

# TensorBoard 시작
tensorboard --logdir logs/

# 원격 서버 접속용
tensorboard --logdir logs/ --bind_all

# http://localhost:6006에서 확인
```

자세한 설정 및 사용법은 [TensorBoard 가이드](docs/TensorBoard_Guide.md)를 참조하세요.

## 학습 전략 비교

### DDP vs DeepSpeed
```bash
# 다양한 분산 학습 전략 비교
python scripts/compare_strategies.py \
    --data-dir data/amazon-book \
    --devices 4 \
    --batch-size 2048 \
    --max-epochs 10
```

자세한 비교는 [DDP vs DeepSpeed 가이드](docs/DDP_vs_DeepSpeed_KR.md)를 참조하세요.

## 결과

비교 스크립트는 다음을 생성합니다:
1. **메트릭 비교** (Recall@K, Precision@K, NDCG@K)
2. **시각화** (막대 차트, 개선 히트맵)
3. **사용자 수준 분석** (샘플 추천 비교)

예시 출력:
```
표준 방법 (사용자-아이템 유사도만):
  Recall@20: 0.1234
  Precision@20: 0.0456
  NDCG@20: 0.0789

향상된 방법 (사용자+관계-아이템 유사도):
  Recall@20: 0.1456 (+18.0%)
  Precision@20: 0.0523 (+14.7%)
  NDCG@20: 0.0891 (+12.9%)
```

## 멀티 GPU 학습 가이드

자세한 멀티 GPU 학습 지침은 [멀티 GPU 가이드](README_MULTI_GPU.md)를 참조하세요.

## 인용

이 코드를 사용하시면 다음을 인용해주세요:
```bibtex
@inproceedings{KGAT2019,
  author = {Wang, Xiang and He, Xiangnan and Cao, Yixin and Liu, Meng and Chua, Tat-Seng},
  title = {KGAT: Knowledge Graph Attention Network for Recommendation},
  booktitle = {KDD},
  year = {2019}
}
```