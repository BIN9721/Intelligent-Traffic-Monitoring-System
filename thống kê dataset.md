# Báo cáo kết quả hoàn thành: Tác vụ 1 - Chuẩn hóa & Hợp nhất Dữ liệu Giao thông (Đã làm sạch nhãn)

Tài liệu này tổng hợp số liệu chính thức và trực quan mẫu của 4 bộ dữ liệu thành phần (Vietnam, UA-DETRAC, AIC21, CityFlowV2) sau khi chạy thuật toán dọn dẹp nhãn tự động [clean_labels.py](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/clean_labels.py) và gộp lại thành tập dữ liệu hợp nhất chính thức phục vụ huấn luyện mô hình YOLO11.

---

## 1. Kết quả Dọn dẹp Nhãn tự động (Label Sanitization)

Mã nguồn [clean_labels.py](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/clean_labels.py) đã thực hiện dọn dẹp toàn bộ dữ liệu tự động gán nhãn của 4 bộ dữ liệu thông qua 3 quy tắc:
1.  **Loại bỏ hộp bao gộp sai (Containment Check)**: Xóa bỏ các hộp bao lớn `truck` / `bus` bao quanh cụm xe con đỗ sát nhau.
2.  **Sửa lỗi hình học xe máy (Aspect Ratio Filter)**: Loại bỏ các hộp bao xe máy bị dẹt ngang (chiều rộng > chiều cao).
3.  **Sửa lỗi hình học xe tải/xe buýt (Size & Aspect Ratio)**: Chuyển các hộp bao `truck`/`bus` nhỏ ở xa về `car` và loại bỏ các hộp đứng dọc không hợp lý.

### Thống kê dọn dẹp chi tiết:
*   **Số lượng nhãn bị gán sai lớp được tự động sửa đổi**: **9,338 nhãn**
*   **Số lượng nhãn phát hiện ảo (false positives) bị xóa bỏ**: **137,808 nhãn**
*   **Tổng số nhãn sạch được giữ lại**: **1,443,175 nhãn**

---

## PHẦN 1: Bộ dữ liệu Việt Nam (196 Video)

Mã nguồn [auto_label.py](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/auto_label.py) — YOLO11x @1280 + NMS (iou=0.45, agnostic_nms=True) + Lọc nhãn bằng `clean_labels.py`.

### Bảng số liệu chính thức — Vietnam Dataset (Đã làm sạch):

| Split | Frames | car | motorcycle | bus | truck | **Tổng Boxes** |
|---|---|---|---|---|---|---|
| **TRAIN (70%)** | 7,316 | 96,873 | 114,821 | 4,636 | 7,824 | **224,154** |
| **VAL (15%)** | 1,603 | 21,004 | 25,213 | 1,225 | 2,165 | **49,607** |
| **TEST (15%)** | 1,661 | 19,933 | 25,980 | 652 | 1,606 | **48,171** |
| **Tổng cộng** | **10,580** | **137,810** | **166,014** | **6,513** | **11,595** | **321,932** |

*Dữ liệu lưu tại: [vietnam_dataset/](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/vietnam_dataset)*

### Mẫu ảnh gán nhãn — Vietnam Dataset:

```carousel
![Vietnam - 20221003-102556_f54](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_results/visualized_20221003-102556_f54.jpg)
<!-- slide -->
![Vietnam - 7h30.17.9.22_f168](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_results/visualized_7h30.17.9.22_f168.jpg)
<!-- slide -->
![Vietnam - 17h30.27.9.22_f18](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_results/visualized_17h30.27.9.22_f18.jpg)
<!-- slide -->
![Vietnam - 7h30.17.9.22_f180](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_results/visualized_7h30.17.9.22_f180.jpg)
<!-- slide -->
![Vietnam - 7h30.17.9.22_f60](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_results/visualized_7h30.17.9.22_f60.jpg)
```

---

## PHẦN 2: Bộ dữ liệu UA-DETRAC (100 Sequences)

Mã nguồn [uadetrac_autolabel.py](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/uadetrac_autolabel.py) — YOLO11x @1280 Auto-Labeler + Lọc nhãn bằng `clean_labels.py`.

### Bảng số liệu chính thức — UA-DETRAC Dataset (Đã làm sạch):

| Split | Sequences | Frames | car | motorcycle | bus | truck | **Tổng Boxes** |
|---|---|---|---|---|---|---|---|
| **TRAIN** | 70 seqs | 19,943 | 347,895 | 7,072 | 26,353 | 12,789 | **394,109** |
| **VAL** | 15 seqs | 4,029 | 81,914 | 2,951 | 7,266 | 3,287 | **95,418** |
| **TEST** | 15 seqs | 3,675 | 78,820 | 1,589 | 4,476 | 2,800 | **87,685** |
| **Tổng cộng** | **100 seqs** | **27,646** | **508,629** | **11,612** | **38,095** | **18,876** | **577,212** |

*Dữ liệu lưu tại: [uadetrac_dataset/](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/uadetrac_dataset)*

### Mẫu ảnh gán nhãn — UA-DETRAC Dataset:

```carousel
![UA-DETRAC - MVI_39851_f00800](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_uadetrac/visualized_MVI_39851_f00800.jpg)
<!-- slide -->
![UA-DETRAC - MVI_39211_f01310](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_uadetrac/visualized_MVI_39211_f01310.jpg)
<!-- slide -->
![UA-DETRAC - MVI_40243_f00791](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_uadetrac/visualized_MVI_40243_f00791.jpg)
<!-- slide -->
![UA-DETRAC - MVI_39801_f00406](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_uadetrac/visualized_MVI_39801_f00406.jpg)
<!-- slide -->
![UA-DETRAC - MVI_40981_f01306](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_uadetrac/visualized_MVI_40981_f01306.jpg)
```

---

## PHẦN 3: Bộ dữ liệu AIC21 Vehicle Counting (31 Videos)

Mã nguồn [aic21_autolabel.py](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/aic21_autolabel.py) — YOLO11x @1280 + NMS (iou=0.45, agnostic_nms=True) + Lọc nhãn bằng `clean_labels.py`.

### Bảng số liệu — AIC21 Dataset (Đã làm sạch):

| Split | Videos | Frames | car | motorcycle | bus | truck | **Tổng Boxes** |
|---|---|---|---|---|---|---|---|
| **TRAIN (70%)** | 21 | 12,862 | 259,148 | 195 | 665 | 30,106 | **290,114** |
| **VAL (15%)** | 5 | 2,447 | 67,212 | 8 | 131 | 4,577 | **71,928** |
| **TEST (15%)** | 5 | 1,471 | 25,868 | 1 | 43 | 3,155 | **29,067** |
| **Tổng cộng** | **31** | **16,780** | **352,228** | **204** | **839** | **37,838** | **391,109** |

*Dữ liệu lưu tại: [aic21_dataset/](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/aic21_dataset)*

### Mẫu ảnh gán nhãn — AIC21 Dataset:

```carousel
![AIC21 - cam_3_f007260](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_aic21/visualized_cam_3_f007260.jpg)
<!-- slide -->
![AIC21 - cam_3_f008740](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_aic21/visualized_cam_3_f008740.jpg)
<!-- slide -->
![AIC21 - cam_7_f009264](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_aic21/visualized_cam_7_f009264.jpg)
<!-- slide -->
![AIC21 - cam_9_f001920](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_aic21/visualized_cam_9_f001920.jpg)
<!-- slide -->
![AIC21 - cam_20_f000460](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_aic21/visualized_cam_20_f000460.jpg)
```

---

## PHẦN 4: Bộ dữ liệu CityFlowV2 — AIC22 (59 Cameras)

Mã nguồn [cityflow_convert.py](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/cityflow_convert.py) — YOLO11x @1280 Auto-Labeler + Lọc nhãn bằng `clean_labels.py`.

### Bảng số liệu — CityFlowV2 Dataset (Đã làm sạch):

| Split | Cameras | Frames | car | motorcycle | bus | truck | **Tổng Boxes** |
|---|---|---|---|---|---|---|---|
| **TRAIN** | 36 | 2,379 | 29,527 | 35 | 112 | 2,394 | **32,068** |
| **VAL** | 23 | 6,602 | 114,238 | 11 | 332 | 6,273 | **120,854** |
| **Tổng cộng** | **59** | **8,981** | **143,765** | **46** | **444** | **8,667** | **152,922** |

*Dữ liệu lưu tại: [cityflow_dataset/](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/cityflow_dataset)*

### Mẫu ảnh gán nhãn — CityFlowV2 Dataset:

```carousel
![CityFlow - S01_c003_f001430](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_cityflow/visualized_S01_c003_f001430.jpg)
<!-- slide -->
![CityFlow - S03_c010_f001210](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_cityflow/visualized_S03_c010_f001210.jpg)
<!-- slide -->
![CityFlow - S03_c013_f000740](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_cityflow/visualized_S03_c013_f000740.jpg)
<!-- slide -->
![CityFlow - S04_c017_f000000](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_cityflow/visualized_S04_c017_f000000.jpg)
<!-- slide -->
![CityFlow - S04_c039_f000010](/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/visual_cityflow/visualized_S04_c039_f000010.jpg)
```

---

## PHẦN 5: Tổng hợp 4 Dataset — combined_dataset/

Mã nguồn [merge_datasets.py](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/merge_datasets.py) — gộp cả 4 bộ dữ liệu với tiền tố filename riêng để tránh xung đột tên tệp tin:

| Nguồn | Prefix | Ví dụ tên file ảnh / nhãn |
|---|---|---|
| Vietnam | `vn_` | `vn_7h30.17.9.22_f168.jpg` |
| UA-DETRAC | `ua_` | `ua_MVI_20065_f01011.jpg` |
| AIC21 | `aic_` | `aic_cam_5_f000600.jpg` |
| CityFlowV2 | `cf_` | `cf_S01_c001_f000055.jpg` |

### Bảng số liệu tổng hợp cuối cùng của Hệ thống sau khi làm sạch nhãn (Dùng điền vào Luận văn):

| Nguồn Dataset | Split | Frames | car | motorcycle | bus | truck | **Tổng Bboxes** |
|---|---|---|---|---|---|---|---|
| **Vietnam** | train+val+test | 10,580 | 137,810 | 166,014 | 6,513 | 11,595 | **321,932** |
| **UA-DETRAC** | train+val+test | 27,647 | 508,629 | 11,612 | 38,095 | 18,876 | **577,212** |
| **AIC21** | train+val+test | 16,780 | 352,228 | 204 | 839 | 37,838 | **391,109** |
| **CityFlowV2** | train+val | 8,981 | 143,765 | 46 | 444 | 8,667 | **152,922** |
| **COMBINED** | **train** | **42,500** | **733,443** | **122,123** | **31,766** | **53,113** | **940,445** |
| **COMBINED** | **val** | **14,681** | **284,368** | **28,183** | **8,954** | **16,302** | **337,807** |
| **COMBINED** | **test** | **6,807** | **124,621** | **27,570** | **5,171** | **7,561** | **164,923** |
| **TỔNG CỘNG** | **Đầy đủ** | **63,988** | **1,142,432** | **177,876** | **45,891** | **76,976** | **1,443,175** |

*Dữ liệu hợp nhất được lưu tại: [combined_dataset/](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/combined_dataset)*
*Tệp cấu hình YOLO hợp nhất: [combined_data.yaml](file:///home/hoangthang/Intelligent%20Traffic%20Monitoring%20System/combined_dataset/combined_data.yaml)*

---

## Tổng hợp tiến độ chuẩn hóa Dataset

| Dataset | Trạng thái | Script thực thi | Số khung hình | Tổng số nhãn |
|---|---|---|---|---|
| **Vietnam (196 videos)** | ✅ Hoàn thành | `auto_label.py` | 10,580 | 321,932 |
| **UA-DETRAC (100 seqs)** | ✅ Hoàn thành | `uadetrac_autolabel.py` | 27,647 | 577,212 |
| **AIC21 (31 videos)** | ✅ Hoàn thành | `aic21_autolabel.py` | 16,780 | 391,109 |
| **CityFlowV2 (59 cameras)** | ✅ Hoàn thành | `cityflow_convert.py` | 8,981 | 152,922 |
| **Tổng hợp 4 Dataset** | ✅ Hoàn thành | `merge_datasets.py` | **63,988** | **1,443,175** |
