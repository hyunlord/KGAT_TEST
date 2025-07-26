# Legacy 및 사용되지 않는 파일들

이 디렉토리에는 프로젝트에서 더 이상 사용되지 않지만 참고용으로 보관된 파일들이 있습니다.

## 디렉토리 구조

### debug/
- 문제 해결을 위해 사용된 디버그 스크립트들
- 현재는 문제가 해결되어 사용되지 않음

### 백업 파일
- `train_original_multi_gpu_backup.py` - 멀티 GPU 학습 스크립트 백업

### 대체된 구현체
- `kgat_model.py` - PyTorch Lightning 버전으로 대체
- `kgat_original.py` - 원본 KGAT 구현 (참고용)
- `kgat_lightning_alt.py` - 대체 구현
- `data_loader_original.py` - data_module.py로 대체
- `train_simple.py` - train.py와 train_improved.py로 대체
- `evaluate_original.py` - evaluator.py로 대체

### 일회성 유틸리티
- `create_proper_data.py` - 데이터 생성 도구
- `test_fixed_data.py` - 수정된 데이터 테스트
- `validate_fix.py` - 수정 검증
- `verify_data_quality.py` - 데이터 품질 검증
- `create_proper_split.py` - 데이터 분할 도구

## 참고사항
이 파일들은 프로젝트 히스토리와 디버깅 과정을 이해하는 데 도움이 될 수 있습니다.