# Intelligent Traffic Monitoring System

Pipeline xử lý và chuẩn hóa dữ liệu giao thông cho huấn luyện mô hình **YOLO11** — gán nhãn tự động, làm sạch nhãn và hợp nhất 4 bộ dataset.

## Tổng quan

Dự án chuẩn hóa dữ liệu từ 4 nguồn (Vietnam, UA-DETRAC, AIC21, CityFlowV2) về định dạng YOLO với 4 lớp:

| Class ID | Tên lớp |
|----------|---------|
| 0 | car |
| 1 | motorcycle |
| 2 | bus |
| 3 | truck |

### Kết quả sau xử lý

| Chỉ số | Giá trị |
|--------|---------|
| Tổng frames | 63,988 |
| Tổng bounding boxes | 1,443,175 |
| Nhãn sai lớp đã sửa | 9,338 |
| False positives đã xóa | 137,808 |

Chi tiết thống kê từng dataset: [`thống kê dataset.md`](thống%20kê%20dataset.md)

## Quy trình xử lý

```
Bước 1 — Gán nhãn tự động (YOLO11x)
  auto_label.py          → vietnam_dataset/
  uadetrac_autolabel.py  → uadetrac_dataset/
  aic21_autolabel.py     → aic21_dataset/
  cityflow_convert.py    → cityflow_dataset/

Bước 2 — Làm sạch nhãn
  clean_labels.py              → Dọn dẹp tự động (3 quy tắc hình học)
  tracking_assisted_clean.py   → Làm sạch có hỗ trợ ByteTrack
  extract_suspicious.py        → Trích xuất frame điều kiện khó (đêm/mưa)
  interactive_cleaner.py       → Chỉnh sửa nhãn thủ công
  apply_refined_labels.py      → Áp dụng nhãn đã chỉnh về dataset gốc

Bước 3 — Hợp nhất
  merge_datasets.py        → combined_dataset/
```

## Cài đặt

```bash
git clone https://github.com/BIN9721/Intelligent-Traffic-Monitoring-System.git
cd Intelligent-Traffic-Monitoring-System
pip install -r requirements.txt
```

**Yêu cầu:**
- Python 3.10+
- GPU NVIDIA (khuyến nghị, cho bước gán nhãn)
- File model `yolo11x.pt` đặt tại thư mục gốc project

## Chạy pipeline

> Các script dùng đường dẫn tuyệt đối trong `PROJECT_ROOT`. Cần chỉnh biến `PROJECT_ROOT` ở đầu mỗi file cho phù hợp máy của bạn.

```bash
# 1. Gán nhãn từng dataset
python auto_label.py
python uadetrac_autolabel.py
python aic21_autolabel.py
python cityflow_convert.py

# 2. Làm sạch nhãn
python clean_labels.py

# 3. Hợp nhất thành dataset cuối
python merge_datasets.py
```

Output cuối cùng:

```
combined_dataset/
  train/images/   train/labels/
  val/images/     val/labels/
  test/images/    test/labels/
  combined_data.yaml
```

## Cấu trúc repo

| File | Mô tả |
|------|-------|
| `auto_label.py` | Gán nhãn 196 video giao thông Việt Nam |
| `uadetrac_autolabel.py` | Gán nhãn UA-DETRAC (100 sequences) |
| `aic21_autolabel.py` | Gán nhãn AIC21 Vehicle Counting (31 videos) |
| `cityflow_convert.py` | Trích frame + gán nhãn CityFlowV2 |
| `clean_labels.py` | Làm sạch nhãn tự động toàn bộ dataset |
| `tracking_assisted_clean.py` | Làm sạch nhãn có tracking |
| `extract_suspicious.py` | Trích xuất frame nghi ngờ (val/test) |
| `interactive_cleaner.py` | Công cụ chỉnh nhãn tương tác |
| `apply_refined_labels.py` | Ghi nhãn đã chỉnh về dataset gốc |
| `merge_datasets.py` | Hợp nhất 4 dataset thành `combined_dataset/` |

## Lưu ý về dữ liệu

Dataset (~78 GB) và model weights **không** được đưa lên GitHub do giới hạn kích thước. Cần chuẩn bị local:

| Thư mục / File | Nội dung |
|----------------|----------|
| `Datasets/` | Video/ảnh gốc từ 4 nguồn |
| `yolo11x.pt` | Pretrained YOLO11x (~110 MB) |
| `*_dataset/` | Output sau gán nhãn |
| `combined_dataset/` | Dataset hợp nhất cuối cùng |

## Tác giả

**HoangThang** — [BIN9721](https://github.com/BIN9721)
