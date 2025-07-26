#!/bin/bash

# 단일 관계 기반 추천 평가 스크립트
# 사용법: ./run_single_relation.sh [체크포인트_경로] [사용자수] [특정관계ID]

echo "======================================"
echo "단일 관계 기반 KGAT 추천 평가"
echo "======================================"

# 기본 설정
CHECKPOINT=${1:-"models/amazon-book_kgat_bi_multi_gpu.pth"}
N_USERS=${2:-"1000"}
TARGET_RELATION=${3:-""}  # 비어있으면 모든 관계 평가

# 설정 출력
echo "설정:"
echo "  체크포인트: $CHECKPOINT"
echo "  평가할 사용자 수: $N_USERS"
if [ -z "$TARGET_RELATION" ]; then
    echo "  평가 대상: 모든 관계"
else
    echo "  평가 대상: 관계 $TARGET_RELATION"
fi
echo ""

# 체크포인트 존재 확인
if [ ! -f "$CHECKPOINT" ]; then
    echo "에러: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT"
    echo "먼저 original KGAT 모델을 학습시켜주세요."
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p results/single_relation

# 평가 실행
echo "평가 시작..."
echo "======================================"

if [ -z "$TARGET_RELATION" ]; then
    # 모든 관계 평가
    python src/evaluate_single_relation.py \
        --checkpoint "$CHECKPOINT" \
        --n-users "$N_USERS" \
        --output-dir results/single_relation
else
    # 특정 관계만 평가
    python src/evaluate_single_relation.py \
        --checkpoint "$CHECKPOINT" \
        --n-users "$N_USERS" \
        --target-relation "$TARGET_RELATION" \
        --output-dir results/single_relation
fi

echo ""
echo "평가 완료!"
echo "결과는 results/single_relation/ 디렉토리에서 확인할 수 있습니다."