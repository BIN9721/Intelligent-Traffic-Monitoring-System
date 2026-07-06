#!/usr/bin/env bash
# Chạy TRÊN pod RunPod (sau khi đã giải nén pod_bundle.tar vào /workspace).
# Không cài lại torch: template RunPod PyTorch đã có sẵn torch build đúng
# CUDA/driver cho GPU của pod (vd RTX 5090 = Blackwell, cần torch build mới đủ
# hỗ trợ sm_120). Cài đè torch từ PyPI có thể phá bản build đó.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Kiểm tra torch có sẵn ==="
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== Cài các gói còn thiếu (không đụng torch) ==="
pip install --no-deps ultralytics
pip install -r requirements-cloud.txt

echo "=== Verify môi trường ==="
python verify_env.py

echo "=== Sẵn sàng train. Ví dụ: ==="
echo "  python train.py --imgsz 1280 --batch 32 --epochs 100 --name yolo11x_full_5090"
