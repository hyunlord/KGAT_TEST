#!/bin/bash

# 구매 관계 기반 직접 추천 평가
# user + purchase_relation → item 방식

echo "======================================"
echo "구매 관계 기반 직접 추천"
echo "표준 KGAT vs 구매관계 직접 사용"
echo "======================================"

# 기본 설정
CHECKPOINT=${1:-"models/amazon-book_kgat_bi_multi_gpu.pth"}
N_USERS=${2:-"1000"}

# 설정 출력
echo "설정:"
echo "  체크포인트: $CHECKPOINT"
echo "  평가할 사용자 수: $N_USERS"
echo ""

# 체크포인트 존재 확인
if [ ! -f "$CHECKPOINT" ]; then
    echo "에러: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT"
    echo "먼저 original KGAT 모델을 학습시켜주세요."
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p results/purchase_relation

# 평가 실행
echo "평가 시작..."
echo "======================================"
echo ""
echo "비교 대상:"
echo "1. 표준 KGAT (전체 임베딩 사용)"
echo "2. 구매관계 + 단순 더하기 (user + relation[0])"
echo "3. 구매관계 + TransR (변환 후 더하기)"
echo ""

python src/evaluate_purchase_relation.py \
    --checkpoint "$CHECKPOINT" \
    --n-users "$N_USERS" \
    --output-dir results/purchase_relation

echo ""
echo "평가 완료!"
echo "결과는 results/purchase_relation/ 디렉토리에서 확인할 수 있습니다."