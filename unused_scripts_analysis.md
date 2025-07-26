# 사용되지 않는 스크립트 분석

## 삭제 가능한 스크립트들

### 1. 백업 및 중복 파일
- `src/train_original_multi_gpu_backup.py` - train_original_multi_gpu.py의 백업 파일

### 2. 디버그 및 테스트 파일 (문제 해결 후 불필요)
- `src/debug_data_consistency.py` - 데이터 일관성 디버그용
- `src/debug_evaluation.py` - 평가 디버그용
- `src/debug_graph_structure.py` - 그래프 구조 디버그용
- `src/debug_metrics.py` - 메트릭 디버그용
- `src/test_fixed_data.py` - 수정된 데이터 테스트용
- `src/validate_fix.py` - 수정 검증용
- `src/verify_data_quality.py` - 데이터 품질 검증용

### 3. 대체된 구현체
- `src/kgat_model.py` - PyTorch Lightning 버전으로 대체됨
- `src/kgat_original.py` - 원본 KGAT 구현 (참고용으로만 필요)
- `src/kgat_lightning_alt.py` - 대체 구현 (사용되지 않음)
- `src/data_loader_original.py` - data_module.py로 대체됨
- `src/train_simple.py` - train.py와 train_improved.py로 대체됨
- `src/evaluate_original.py` - evaluator.py로 대체됨

### 4. 데이터 처리 도구 (일회성)
- `src/create_proper_data.py` - 데이터 생성 후 불필요
- `scripts/create_proper_split.py` - 데이터 분할 후 불필요

## 유지해야 할 주요 스크립트들

### 핵심 모델 파일
- `src/kgat_lightning.py` - 기본 KGAT 모델
- `src/kgat_improved.py` - 개선된 KGAT 모델
- `src/kgat_lightning_fixed.py` - 수정된 KGAT 모델

### 학습 스크립트
- `src/train.py` - 기본 학습 스크립트
- `src/train_improved.py` - 개선된/수정된 모델 학습
- `src/train_original.py` - 원본 구현 학습 (비교용)
- `src/train_original_multi_gpu.py` - 원본 멀티 GPU 학습

### 평가 및 비교
- `src/evaluator.py` - 평가 메트릭
- `src/evaluate_comparison.py` - 평가 방법 비교
- `src/compare_methods.py` - 방법 비교 유틸리티
- `src/compare_models.py` - 모델 비교

### 데이터 처리
- `src/data_module.py` - PyTorch Lightning 데이터 모듈
- `src/data_loader.py` - 데이터 로더

### 셸 스크립트
- `scripts/train_multi_gpu.sh` - 멀티 GPU 학습
- `scripts/train_improved_multi_gpu.sh` - 개선된 모델 멀티 GPU
- `scripts/train_fixed_multi_gpu.sh` - 수정된 모델 멀티 GPU (새로 생성)
- `scripts/train_fixed_model.sh` - 수정된 모델 단일 GPU
- 기타 필요한 학습 스크립트들

## 권장사항
1. 디버그 파일들은 별도의 `debug/` 폴더로 이동하거나 삭제
2. 백업 파일은 Git 히스토리로 관리하고 삭제
3. 대체된 구현체는 `legacy/` 폴더로 이동하거나 문서화 후 삭제