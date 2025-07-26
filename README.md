# KGAT PyTorch Lightning 구현

추천 시스템을 위한 Knowledge Graph Attention Network (KGAT)의 PyTorch Lightning 기반 구현입니다. 표준 추천 방식과 관계 기반 향상 추천 방식을 비교하는 평가 방법을 포함합니다.

## 주요 기능

- ✅ PyTorch Lightning 기반 구현
- ✅ DataModule을 활용한 모듈식 아키텍처
- ✅ Hydra 설정 관리
- ✅ TensorBoard 로깅
- ✅ 모델 체크포인트 및 조기 종료
- ✅ 다양한 Aggregator 지원 (bi-interaction, GCN, GraphSAGE)
- ✅ 개선된 KGAT 모델 옵션
- ✅ 두 가지 평가 방법 비교:
  - 표준: 사용자-아이템 유사도만 사용
  - 향상: 사용자+관계-아이템 유사도 사용
- ✅ 멀티 GPU 학습 (DDP, DeepSpeed Stage 1/2/3)

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

#### 기본 모델 학습
```bash
# 기본 설정으로 학습
python src/train.py

# 사용자 정의 설정으로 학습
python src/train.py data.batch_size=512 model.embed_dim=128

# 작은 설정으로 학습 (테스트용)
python src/train.py --config-name config_small
```

#### 개선된 모델 학습
```bash
# 개선된 KGAT 모델 사용 (권장)
python src/train_improved.py use_improved_model=true

# 다양한 Aggregator 사용
python src/train_improved.py use_improved_model=true model.aggregator=gcn
python src/train_improved.py use_improved_model=true model.aggregator=graphsage
python src/train_improved.py use_improved_model=true model.aggregator=bi-interaction

# 기본 모델과 개선된 모델 비교
python src/compare_models.py
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
│   ├── kgat_improved.py       # 개선된 KGAT 모델
│   ├── data_module.py         # 데이터 로딩 및 전처리
│   ├── train.py               # 학습 스크립트
│   ├── train_improved.py      # 개선된 모델 학습 스크립트
│   ├── evaluator.py           # 평가 방법
│   ├── compare_methods.py     # 방법 비교 유틸리티
│   ├── evaluate_comparison.py # 학습된 모델 방법 비교
│   └── compare_models.py      # 기본/개선 모델 비교
├── configs/
│   ├── config.yaml            # 주요 설정
│   ├── config_small.yaml      # 테스트용 작은 설정
│   ├── config_improved.yaml   # 개선된 모델 설정
│   └── config_multi_gpu.yaml  # 멀티 GPU 설정
├── scripts/
│   ├── download_data.py       # 데이터 다운로드 스크립트
│   └── compare_strategies.py  # 학습 전략 비교
├── docs/
│   ├── DDP_vs_DeepSpeed_KR.md    # DDP vs DeepSpeed 가이드
│   ├── KGAT_Improvements.md       # 모델 개선사항 문서
│   └── TensorBoard_Guide.md       # TensorBoard 사용 가이드
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
  aggregator: bi-interaction  # 옵션: bi-interaction, gcn, graphsage
  dropout: 0.1
  
training:
  max_epochs: 1000
  early_stopping_patience: 20
  lr: 0.001
  
# 개선된 모델 사용 여부
use_improved_model: false  # true로 설정하면 개선된 모델 사용
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

## 개선사항

개선된 KGAT 모델의 주요 특징:
- **다양한 Aggregator**: bi-interaction, GCN, GraphSAGE 지원
- **향상된 어텐션 메커니즘**: 더 나은 gradient flow
- **엣지 정규화**: GCN 스타일 메시지 전달
- **최종 변환 레이어**: 효율적인 차원 축소

자세한 내용은 [모델 개선사항 문서](docs/KGAT_Improvements.md)를 참조하세요.

## 멀티 GPU 학습 가이드

자세한 멀티 GPU 학습 지침은 [멀티 GPU 가이드](README_MULTI_GPU.md)를 참조하세요.

## 성능 비교

개선된 모델의 예상 성능 향상:
- Recall@20: 5-10% 향상
- NDCG@20: 5-8% 향상
- 더 안정적인 학습 과정
- 다양한 데이터셋에 대한 적응성 향상

## 사용되지 않는 스크립트 정리

다음 스크립트들은 프로젝트에서 더 이상 사용되지 않으며 정리 대상입니다:
- 디버그 스크립트: `debug_*.py` 파일들
- 백업 파일: `*_backup.py`
- 대체된 구현: `kgat_model.py`, `kgat_original.py`, `kgat_lightning_alt.py`
- 일회성 도구: `create_proper_data.py`, `validate_fix.py`

자세한 내용은 [사용되지 않는 스크립트 분석](unused_scripts_analysis.md)을 참조하세요.

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