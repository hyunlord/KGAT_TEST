# Hydra 설정 가이드

## 주요 변경사항 (Hydra 충돌 해결)

### 문제
```bash
Could not append to config. An item is already at 'model.aggregator'.
```

### 해결 방법

1. **기존 설정 덮어쓰기**: 이미 존재하는 키는 `+` 없이 사용
   ```bash
   model.aggregator=bi-interaction  # 기존 값 덮어쓰기
   ```

2. **새로운 설정 추가**: 존재하지 않는 키는 `++` 사용
   ```bash
   ++model.type=kgat_fixed  # 새로운 키 추가
   ```

3. **키 이름 확인**: 설정 파일과 일치하는 키 사용
   - `model.weight_decay` → `model.reg_weight`
   - `model.aggregator=bi` → `model.aggregator=bi-interaction`

## 올바른 사용 예시

### Fixed KGAT 학습
```bash
python src/train_improved.py \
    ++model.type=kgat_fixed \
    ++model.embedding_size=64 \
    ++model.layer_sizes=[64,32,16] \
    model.aggregator=bi-interaction \
    model.reg_weight=1e-5 \
    training.devices=4 \
    training.strategy=ddp
```

### 개선된 모델 학습
```bash
python src/train_improved.py \
    use_improved_model=true \
    model.aggregator=gcn \
    training.devices=-1
```

## 설정 테스트

설정이 올바른지 미리 확인:
```bash
# 설정만 출력 (학습 X)
python src/train_improved.py [옵션들] --cfg job

# 자동 테스트 스크립트
python scripts/test_hydra_config.py
```

## 주의사항

1. **aggregator 값**:
   - `bi-interaction` (O)
   - `bi` (X - 잘못된 값)

2. **배열 값**:
   ```bash
   ++model.layer_sizes=[64,32,16]  # 올바른 형식
   ++model.layer_sizes="[64,32,16]"  # 문자열로 처리될 수 있음
   ```

3. **디버깅**:
   ```bash
   # 전체 스택 트레이스 보기
   HYDRA_FULL_ERROR=1 python src/train_improved.py ...
   ```

## 자주 발생하는 오류

1. **"Could not append to config"**
   - 원인: 이미 존재하는 키에 `+` 사용
   - 해결: `+` 제거 또는 `++` 사용

2. **"Invalid value for config key"**
   - 원인: 잘못된 값 형식
   - 해결: 설정 파일에서 허용된 값 확인

3. **"Missing mandatory value"**
   - 원인: 필수 설정 누락
   - 해결: `???`로 표시된 필수 값 제공