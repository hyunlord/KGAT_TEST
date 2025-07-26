#!/bin/bash
# torch-scatter 설치 스크립트

echo "Installing torch-scatter..."

# PyTorch 버전 확인
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# torch-scatter 설치 (PyTorch 버전에 맞게)
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace('.', ''))" ).html

echo "Installation complete!"