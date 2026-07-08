"""Huấn luyện YOLO11x trên combined_dataset.

Ví dụ chạy thử (smoke test) trên GPU 4GB local:
    python train.py --imgsz 640 --batch 4 --epochs 2 --fraction 0.05

Ví dụ chạy full trên GPU cloud (24GB+):
    python train.py --imgsz 1280 --batch 16 --epochs 100
"""
import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_data_yaml(data_path: str) -> str:
    """Ghi đè key 'path' bằng thư mục tuyệt đối của yaml trên máy hiện tại.

    combined_data.yaml không lưu 'path' tuyệt đối vì file này được chuyển
    qua nhiều máy (local <-> pod cloud); mỗi máy sẽ tự tính lại đường dẫn đúng.
    """
    data_path = Path(data_path).resolve()
    with open(data_path) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(data_path.parent)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(PROJECT_ROOT / "combined_dataset" / "combined_data.yaml"))
    p.add_argument("--weights", default=str(PROJECT_ROOT / "yolo11x.pt"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--fraction", type=float, default=1.0, help="Tỷ lệ tập train dùng để huấn luyện (dùng <1.0 để chạy thử nhanh)")
    p.add_argument("--device", default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="yolo11x_traffic")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--noval", action="store_true", help="Bo qua validate giua cac epoch, chi validate epoch cuoi (tiet kiem thoi gian khi bi gioi han deadline)")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)
    model.train(
        data=resolve_data_yaml(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        fraction=args.fraction,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        resume=args.resume,
        val=not args.noval,
        amp=True,
        cache=False,
        # Ultralytics 8.x không còn hỗ trợ fl_gamma (focal loss gamma) như YOLOv5.
        # cls là hệ số nhân chung cho toàn bộ classification loss (không phân biệt từng lớp),
        # tăng lên để mô hình chú trọng phân loại hơn, giảm nhầm lẫn bus/truck thiểu số.
        cls=2.0,
        project=str(PROJECT_ROOT / "runs" / "detect"),
    )

    if args.noval:
        # val=False bo qua validate ca luc cuoi training, nen chay rieng 1 lan
        # tren weights cuoi cung (last.pt) de van co so lieu mAP cho luan van.
        model.val(data=resolve_data_yaml(args.data), imgsz=args.imgsz, batch=args.batch)


if __name__ == "__main__":
    main()
