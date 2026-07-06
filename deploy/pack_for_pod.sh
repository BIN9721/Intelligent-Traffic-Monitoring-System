#!/usr/bin/env bash
# Chạy trên máy cá nhân: đóng gói code + dataset + weights thành 1 file tar
# để chuyển sang RunPod bằng `runpodctl send`.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$PROJECT_ROOT/pod_bundle.tar}"

echo "Đóng gói từ: $PROJECT_ROOT"
echo "File output:  $OUT"

tar -cf "$OUT" -C "$PROJECT_ROOT" \
    combined_dataset \
    yolo11x.pt \
    train.py \
    verify_env.py \
    requirements-cloud.txt

echo "Xong. Kích thước:"
du -sh "$OUT"
echo
echo "Bước tiếp theo trên máy cá nhân:"
echo "  runpodctl send \"$OUT\""
echo "-> sẽ in ra 1 mã code, ví dụ: runpodctl receive 1234-abcd-..."
echo
echo "Trên pod (sau khi SSH vào), chạy đúng lệnh receive đó, rồi:"
echo "  tar -xf pod_bundle.tar -C /workspace/"
