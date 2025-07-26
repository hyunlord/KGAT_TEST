#!/bin/bash

# 모든 KGAT 모델 구현체의 성능을 비교하는 스크립트
# Original, Lightning, Fixed 버전의 결과를 시각화하여 비교

echo "=========================================="
echo "KGAT 모델 구현체 비교 스크립트"
echo "=========================================="

# 기본 체크포인트 경로 설정
ORIGINAL_CKPT=""
LIGHTNING_CKPT=""
FIXED_CKPT=""
IMPROVED_CKPT=""

# 명령줄 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --original)
            ORIGINAL_CKPT="$2"
            shift 2
            ;;
        --lightning)
            LIGHTNING_CKPT="$2"
            shift 2
            ;;
        --fixed)
            FIXED_CKPT="$2"
            shift 2
            ;;
        --improved)
            IMPROVED_CKPT="$2"
            shift 2
            ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --original <경로>   Original KGAT 체크포인트 경로"
            echo "  --lightning <경로>  Lightning KGAT 체크포인트 경로"
            echo "  --fixed <경로>      Fixed KGAT 체크포인트 경로"
            echo "  --improved <경로>   Improved KGAT 체크포인트 경로"
            echo ""
            echo "예시:"
            echo "  # 모든 모델 비교"
            echo "  $0 --original models/original.pth --lightning models/lightning.ckpt --fixed models/fixed.ckpt"
            echo ""
            echo "  # Original과 Fixed만 비교"
            echo "  $0 --original models/original.pth --fixed models/fixed.ckpt"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "--help를 사용하여 도움말을 확인하세요."
            exit 1
            ;;
    esac
done

# 체크포인트 자동 찾기 (지정되지 않은 경우)
if [ -z "$ORIGINAL_CKPT" ] && [ -d "models" ]; then
    ORIGINAL_CKPT=$(find models -name "*original*.pth" -type f | head -1)
    [ -n "$ORIGINAL_CKPT" ] && echo "Original 체크포인트 자동 감지: $ORIGINAL_CKPT"
fi

if [ -z "$LIGHTNING_CKPT" ] && [ -d "models" ]; then
    LIGHTNING_CKPT=$(find models -name "*kgat_*epoch*.ckpt" -type f | grep -v "fixed\|improved" | head -1)
    [ -n "$LIGHTNING_CKPT" ] && echo "Lightning 체크포인트 자동 감지: $LIGHTNING_CKPT"
fi

if [ -z "$FIXED_CKPT" ] && [ -d "models" ]; then
    FIXED_CKPT=$(find models -name "*fixed*.ckpt" -type f | head -1)
    [ -n "$FIXED_CKPT" ] && echo "Fixed 체크포인트 자동 감지: $FIXED_CKPT"
fi

if [ -z "$IMPROVED_CKPT" ] && [ -d "models" ]; then
    IMPROVED_CKPT=$(find models -name "*improved*.ckpt" -type f | head -1)
    [ -n "$IMPROVED_CKPT" ] && echo "Improved 체크포인트 자동 감지: $IMPROVED_CKPT"
fi

# 비교할 모델이 있는지 확인
if [ -z "$ORIGINAL_CKPT" ] && [ -z "$LIGHTNING_CKPT" ] && [ -z "$FIXED_CKPT" ] && [ -z "$IMPROVED_CKPT" ]; then
    echo "Error: 비교할 모델 체크포인트가 없습니다."
    echo "--help를 사용하여 도움말을 확인하세요."
    exit 1
fi

# 비교 명령어 구성
CMD="python src/evaluate_model_comparison.py"

[ -n "$ORIGINAL_CKPT" ] && CMD="$CMD --original-checkpoint $ORIGINAL_CKPT"
[ -n "$LIGHTNING_CKPT" ] && CMD="$CMD --lightning-checkpoint $LIGHTNING_CKPT"
[ -n "$FIXED_CKPT" ] && CMD="$CMD --fixed-checkpoint $FIXED_CKPT"
[ -n "$IMPROVED_CKPT" ] && CMD="$CMD --improved-checkpoint $IMPROVED_CKPT"

# 실행
echo ""
echo "비교 실행 중..."
echo "명령어: $CMD"
echo ""

$CMD

echo ""
echo "비교 완료!"
echo "결과는 results/model_comparison/ 디렉토리에 저장되었습니다."