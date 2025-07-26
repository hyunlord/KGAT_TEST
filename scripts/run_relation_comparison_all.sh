#!/bin/bash

# KGAT 표준 vs 관계 강화 추천 비교 (모든 모델 지원)
# 사용법: ./run_relation_comparison_all.sh [체크포인트_경로] [사용자수] [모델타입]

echo "======================================"
echo "KGAT 표준 vs 관계 강화 추천 비교"
echo "모든 모델 타입 지원 (Original/Fixed)"
echo "======================================"

# 기본 설정
CHECKPOINT=${1:-"models/amazon-book_kgat.pth"}
N_USERS=${2:-"1000"}
MODEL_TYPE=${3:-"auto"}  # auto, original, fixed

# 설정 출력
echo "설정:"
echo "  체크포인트: $CHECKPOINT"
echo "  비교할 사용자 수: $N_USERS"
echo "  모델 타입: $MODEL_TYPE"
echo ""

# 체크포인트 존재 확인
if [ ! -f "$CHECKPOINT" ]; then
    echo "에러: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT"
    echo "학습된 모델 파일을 지정해주세요."
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p results/relation_comparison

# 비교 실행
echo "비교 분석 시작..."
echo "======================================"

python src/evaluate_relation_comparison_all.py \
    --checkpoint "$CHECKPOINT" \
    --model-type "$MODEL_TYPE" \
    --n-users "$N_USERS" \
    --output-dir results/relation_comparison

echo ""
echo "비교 완료!"
echo "결과는 results/relation_comparison/ 디렉토리에서 확인할 수 있습니다."