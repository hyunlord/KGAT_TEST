#!/bin/bash

# TransR 방식 비교 스크립트
# 단순 더하기 vs TransR 선형 변환 비교

echo "======================================"
echo "KGAT TransR 방식 비교"
echo "단순 더하기 vs TransR 변환"
echo "======================================"

# 기본 설정
CHECKPOINT=${1:-"models/amazon-book_kgat_bi_multi_gpu.pth"}
N_USERS=${2:-"1000"}

# 설정 출력
echo "설정:"
echo "  체크포인트: $CHECKPOINT"
echo "  비교할 사용자 수: $N_USERS"
echo ""

# 체크포인트 존재 확인
if [ ! -f "$CHECKPOINT" ]; then
    echo "에러: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT"
    echo "먼저 original KGAT 모델을 학습시켜주세요."
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p results/transr_comparison

# 비교 실행
echo "비교 분석 시작..."
echo "======================================"

python src/evaluate_relation_comparison_transr.py \
    --checkpoint "$CHECKPOINT" \
    --n-users "$N_USERS" \
    --output-dir results/transr_comparison

echo ""
echo "비교 완료!"
echo "결과는 results/transr_comparison/ 디렉토리에서 확인할 수 있습니다."