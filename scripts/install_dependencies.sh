#!/bin/bash
# 원본 KGAT를 위한 추가 의존성 설치

echo "Installing additional dependencies for original KGAT..."

# 기본 방법
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 또는 conda를 사용하는 경우
# conda install pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg

echo "Installation complete!"