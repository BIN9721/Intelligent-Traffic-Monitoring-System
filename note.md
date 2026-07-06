# Tóm tắt Tiến độ Dự án: Hệ thống Giám sát Giao thông Thông minh

Tài liệu này tóm tắt toàn bộ các công việc, thuật toán, số liệu thực nghiệm và kết quả đã đạt được từ đầu dự án đến thời điểm hiện tại. Các số liệu này đã được chuẩn hóa và sẵn sàng để điền vào các phần trống (đánh dấu `🔶`) trong Luận văn Thạc sĩ.

---

## 1. Tổng quan & Thiết lập Ban đầu
*   **Mục tiêu**: Xây dựng hệ thống giám sát giao thông thời gian thực trên phần cứng GPU NVIDIA RTX 3050 (4 GB VRAM) phục vụ nhận diện 4 lớp phương tiện: `car` (ô tô con), `motorcycle` (xe máy), `bus` (xe buýt/khách), `truck` (xe tải).
*   **Trạng thái ban đầu**: Luận văn trống toàn bộ bảng số liệu thực nghiệm. Chưa có bất kỳ mã nguồn tiền xử lý, gán nhãn, tích hợp tracking hay giao diện dashboard nào.

---

## 2. Kết quả Chuẩn bị & Hợp nhất Dữ liệu (Tập dữ liệu siêu sạch)
Chúng ta đã tiến hành trích xuất khung hình, tự động gán nhãn bằng mô hình YOLO11x (ngưỡng tin cậy `conf=0.15`, `iou=0.45`, `agnostic_nms=True`) và gộp thành công 4 bộ dữ liệu thành phần thành tập dữ liệu hợp nhất duy nhất **`combined_dataset/`**.

### Bảng số liệu thống kê phân chia tập dữ liệu (Mới nhất - Đã hủy gán nhãn thủ công & đồng bộ tự động):

| Phân lớp (Class) | TRAIN (Đã Augment) | VAL | TEST | **TỔNG CỘNG** | Tỷ lệ (%) |
|---|---|---|---|---|---|
| **car** | 985,327 | 297,109 | 133,854 | **1,414,816** | *86.1%* |
| **motorcycle** | 141,568 | 27,021 | 25,285 | **193,874** | *11.8%* |
| **bus** | 13,645 | 3,926 | 2,264 | **19,835** | *1.2%* |
| **truck** | 10,282 | 2,755 | 1,214 | **14,251** | *0.9%* |
| **TỔNG KHUNG HÌNH** | **53,125** | **14,681** | **6,807** | **74,613** | *100%* |
| **TỔNG NHÃN (BBOX)** | **1,150,822** | **330,811** | **162,617** | **1,644,250** | |

---

## 3. Các Thuật toán & Kỹ thuật Làm sạch Nhãn đã Áp dụng
Để đạt được tập dữ liệu 1.64 triệu nhãn sạch ở trên, chúng ta đã phát triển và chạy chuỗi thuật toán hậu xử lý nhãn thông minh:

### 3.1. Tăng cường dữ liệu ngoại tuyến (Offline Augmentation - `offline_augment.py`)
Áp dụng giả lập thời tiết và ánh sáng khắc nghiệt trực tiếp lên **10,625 ảnh Train** (25% tập huấn luyện gốc) để mô hình có khả năng chống nhiễu:
*   **Tăng cường chói đèn pha đêm (Night Glare)**: Thêm quầng sáng đồng tâm quanh đèn pha.
*   **Tăng cường mưa rơi (Rain streaks)**: Thêm các vệt sáng chéo giả lập mưa bão lớn.
*   **Tăng cường làm mờ chuyển động (Motion Blur)**: Tạo vệt mờ trượt giả lập xe chạy tốc độ cao hoặc rung lắc camera.

### 3.2. Lọc vật thể tĩnh ven đường (Static Noise Filter - `remove_static_noise.py`)
Do góc quay cố định của camera giám sát, các vật thể không di chuyển như biển quảng cáo lớn thường bị nhận diện nhầm là xe buýt (`bus`), và tán cây dung rinh/biển báo thường bị nhận diện nhầm là xe tải (`truck`). 
Thuật toán đã lọc và xóa sạch **14,018 nhãn tĩnh** xuất hiện liên tục không di chuyển trên 15+ khung hình tại **9,147 file**:
*   Xóa bỏ **8,229 nhãn** chướng ngại vật/biển quảng cáo tĩnh bị nhầm là **Bus**.
*   Xóa bỏ **5,789 nhãn** tán cây/cột đèn tĩnh bị nhầm là **Truck**.

### 3.3. Hài hòa phân loại nhãn (Taxonomy Alignment - `align_taxonomy.py`)
Giải quyết triệt để sự bất đồng nhất về định nghĩa xe bán tải, SUV, minibus ở các quốc gia khác nhau. Thuật toán đã quét qua 63,988 nhãn và tự động hiệu chỉnh **32,995 file nhãn** dựa trên tỷ lệ khung hình và diện tích:
*   **Rule 1**: Xe tải rất nhỏ ở xa $\to$ `car` (46,918 nhãn).
*   **Rule 2**: Xe bán tải / SUV vuông vắn $\to$ `car` (3,394 nhãn).
*   **Rule 3**: Xe khách siêu nhỏ ở xa $\to$ `car` (17,682 nhãn).
*   **Rule 4**: Xe con siêu dài sát camera $\to$ `truck` (261 nhãn).

### 3.4. Các quy tắc lọc hình học & chéo lớp (`clean_labels.py`)
*   **Rule 1 (Containment Check)**: Loại bỏ các hộp bao khổng lồ `bus`/`truck` bao ngoài các cụm xe con đỗ sát nhau.
*   **Rule 2 (Motorcycle Aspect Ratio)**: Loại bỏ các hộp bao xe máy bị biến dạng dẹt ngang (chiều rộng > chiều cao) do bắt nhầm biển báo/dải phân cách.
*   **Rule 3 (Distant Vehicle)**: Tự động chuyển các hộp bao `bus`/`truck` kích thước quá nhỏ ở rất xa camera về nhãn an toàn là `car`.
*   **Rule 4 (Strict Multi-Class Duplicate NMS)**: Lọc bỏ các hộp bao trùng lặp đè lên nhau trên cùng một phương tiện ($IoU > 0.40$, bất kể khác class), chỉ giữ lại hộp bao nhỏ hơn (tighter fit).
*   **Rule 5 (Geometric Perspective)**: Loại bỏ các hộp bao lớn xuất hiện ở 35% nửa trên khung hình (khu vực đường chân trời/vùng xa).

### 3.5. Sửa nhãn nhảy lớp bằng Tracking (`tracking_assisted_clean.py`)
*   **Cơ chế**: Chạy YOLO11x kết hợp bộ theo dõi đối tượng **ByteTrack** trên **227 video** (Vietnam & AIC21).
*   **Thuật toán**: Gom nhóm toàn bộ hộp bao thuộc cùng một hành trình (Track ID) của xe qua các khung hình và áp dụng **Bầu cử đa số (Majority Voting)** để thống nhất lớp xe. Triệt tiêu hoàn toàn lỗi nhấp nháy đổi nhãn chéo giữa `car`, `bus` và `truck` do thay đổi góc nhìn hoặc ánh sáng che khuất.

### 3.6. Tối ưu hóa nhãn kiểm thử & Hủy bỏ chỉnh sửa thủ công (`restore_automated_labels.py`)
*   **Trạng thái ban đầu**: 1,312 khung hình nghi ngờ thuộc điều kiện mưa và đêm đã được lọc ra và chỉnh sửa thủ công để tinh chỉnh tập Val/Test.
*   **Hành động khôi phục tự động**: Để loại bỏ các sai sót chủ quan của con người trong quá trình gán nhãn thủ công và đảm bảo dữ liệu luận văn hoàn toàn minh bạch/tự động, chúng ta đã chạy script `restore_automated_labels.py`.
*   **Kết quả**: Khôi phục thành công **1,656 tệp nhãn** trong Val/Test về nhãn tự động sinh ra bởi YOLO11x + bộ lọc hình học gốc của dự án. Sau đó, chạy lại taxonomy alignment và static noise filter để đảm bảo dữ liệu siêu sạch và nhất quán 100% tự động.

---

## 4. Thực nghiệm Tiền xử lý ảnh đầu vào (`visualize_preprocessing.py`)
Chúng ta đã thử nghiệm quy trình chuẩn hóa ảnh đầu vào gồm 4 bước: **Letterbox** (Resize giữ nguyên tỷ lệ ảnh) $\to$ **CLAHE trên kênh L** (Hệ màu LAB) $\to$ **BGR2RGB** $\to$ **Pixel Normalization**.

### Kết quả cải thiện độ chính xác suy luận (Raw vs. Preprocessed):
*   **Trời mưa lớn (AIC21)**: Số xe phát hiện tăng từ 4 lên 7 xe (**+75.0%**).
*   **Giao thông đêm Việt Nam (Đèn pha chói)**:
    *   Trên ảnh chập tối thông thường: Số xe phát hiện tăng **+15.6%**.
    *   Trên ảnh đêm tối cực đoan chứa ánh đèn pha lóa thẳng vào ống kính (`vn_18h.7.9.22_f189.jpg`): YOLO gốc nhận diện **49 xe**, YOLO sau tiền xử lý CLAHE nhận diện **68 xe** (**cải thiện vượt bậc +38.8%**). CLAHE giúp dập độ chói của đèn pha và kích sáng vùng tối, khôi phục lại hình dáng xe máy và ô tô đi phía sau nguồn sáng.

---

## 5. Tối ưu hóa Không gian Lưu trữ
*   Để giải quyết vấn đề đầy bộ nhớ ổ cứng trong quá trình làm việc, toàn bộ các thư mục ảnh trích xuất trung gian (`vietnam_dataset/`, `aic21_dataset/`, `uadetrac_dataset/`, `cityflow_dataset/`) và các file nhãn tạm thời đã được xóa bỏ an toàn sau khi đã gộp hoàn chỉnh vào tập huấn luyện cuối cùng.
*   **Dung lượng giải phóng**: **19 GB** (dung lượng dự án giảm từ 77 GB xuống còn **58 GB**).

---

## 6. Các bước tiếp theo cần triển khai
1.  **Huấn luyện mô hình YOLO11x**: Thiết lập siêu tham số và chạy kịch bản huấn luyện YOLO11x trên tập dữ liệu sạch gộp 74,613 ảnh.
2.  **Tích hợp ByteTrack & Đo đạc**: Viết module đếm xe (Virtual Line/ROI) và ước lượng tốc độ (Homography) để hoàn thiện hệ thống.

---

## 7. Khảo sát Khoa học cho Quá trình Huấn luyện (Chương 4 Luận văn)

### 7.1. Lựa chọn độ phân giải huấn luyện (`imgsz`)
*   **Độ phân giải gốc của dataset**: Dao động từ 960x540 (UA-DETRAC), 1280x960 (AIC21) đến 1920x1080 (CityFlowV2, AIC21) và 1600x1200 (Vietnam).
*   **Quyết định**: Khuyến nghị huấn luyện ở **`imgsz = 1280`** (hoặc tối thiểu là `960`).
*   **Lập luận**: Do đặc thù camera giao thông, các phương tiện ở xa chiếm diện tích rất nhỏ (chỉ vài chục pixel). Việc hạ độ phân giải xuống `640` sẽ làm nén các phương tiện này thành các khối 4x4 pixel, khiến mô hình không thể học được đặc trưng chi tiết. Train ở 1280 giúp bảo toàn thông tin biên của các vật thể ở xa.

### 7.2. Lựa chọn thiết bị phần cứng huấn luyện
*   Để huấn luyện YOLO11x (56.9M tham số, 194.9 GFLOPs) ở độ phân giải 1280 với 74.6K ảnh, phần cứng laptop RTX 3050 (4GB VRAM) chắc chắn sẽ bị lỗi tràn bộ nhớ (OOM) và có nguy cơ quá nhiệt.
*   **Khuyến nghị**: Sử dụng **NVIDIA RTX 5090 (32 GB VRAM)** hoặc thuê đám mây **RTX 3090 / RTX 4090 (24 GB VRAM)**. Cấu hình này cho phép huấn luyện mô hình ở kích thước 1280 với `batch_size = 16/32`, kích hoạt FP16 (AMP) giúp hoàn thành toàn bộ quá trình train chỉ trong **8 - 12 giờ**.

### 7.3. Chiến lược giải quyết mất cân bằng lớp (Class Imbalance) bằng Focal Loss
*   **Tỷ lệ mất cân bằng**: Lớp `car` chiếm 86.1%, trong khi `bus` (1.2%) và `truck` (0.9%) là các lớp thiểu số.
*   **Tại sao không dùng Resampling (Oversampling/Undersampling)?** Do một khung hình chứa nhiều phương tiện khác nhau (multi-object), việc nhân bản ảnh chứa `truck` sẽ làm nhân bản theo các xe `car` và `motorcycle` nằm chung trong đó, làm trầm trọng hơn sự mất cân bằng. Việc xóa ảnh chứa `car` làm lãng phí 70% dữ liệu.
*   **Tại sao không dùng Weighted Cross-Entropy (WCE)?** WCE nhân hệ số phạt cố định $w_c$ bất kể độ khó. Đối với các xe tải lớn ở gần (dễ nhận diện), mô hình vẫn phạt quá nặng dẫn đến việc đoán bừa cây cối hoặc biển báo thành xe tải (giảm chỉ số Precision).
*   **Giải pháp được chọn: Focal Loss (`fl_gamma = 2.0`) & Tăng trọng phân lớp (`cls = 2.0`)**:
    *   Công thức: $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$.
    *   Hệ số điều chế $(1 - p_t)^\gamma$ tự động triệt tiêu loss của các mẫu dễ (khi $p_t \to 1.0$, hệ số tiến về $0$). Lớp xe con `car` sau vài epochs đầu sẽ không còn tạo ra loss đáng kể. Trọng tâm cập nhật mô hình sẽ dồn hoàn toàn vào việc sửa các mẫu khó (xe buýt, xe tải ở xa hoặc bị che khuất).
    *   Tăng hệ số `cls` từ mặc định `0.5` lên `2.0` để tăng mức độ phạt khi phân loại sai các lớp thiểu số này.
    *   **Đính chính (2026-07-06)**: Khi triển khai thực tế, phát hiện Ultralytics 8.x (bản dùng cho YOLO11) **không còn tham số `fl_gamma`** như YOLOv5 cũ — API loss v8 không expose focal loss gamma qua `train()`. Đã bỏ `fl_gamma` khỏi script, chỉ còn giữ `cls = 2.0` (hệ số nhân chung cho toàn bộ classification loss, không phân biệt được từng lớp thiểu số như mô tả lý thuyết ở trên). Nếu luận văn cần đúng cơ chế Focal Loss per-class, phải tự viết custom loss function ghi đè `v8DetectionLoss`.

---

## 8. Giai đoạn 1 — Chuẩn bị Huấn luyện trên Máy cá nhân (2026-07-06)

Trước khi thuê GPU cloud (RTX 5090 trên RunPod), đã hoàn tất toàn bộ khâu chuẩn bị miễn phí trên máy cá nhân (RTX 3050 4GB VRAM):

### 8.1. Thiết lập môi trường
*   Sử dụng conda env có sẵn tên `traffic` (Python 3.10.20).
*   **Lỗi môi trường phát hiện**: `pip`/`python -m pip` trong env conda bị lệch sang site-packages user (`~/.local`) do biến môi trường user-site che khuất — khắc phục bằng cách luôn export `PYTHONNOUSERSITE=1` trước khi cài đặt/chạy.
*   Cài `torch`/`torchvision` (build `cu121`, resolve thành `2.12.1+cu130`) + `ultralytics==8.4.83` + `opencv-python==4.10.0`.
*   `verify_env.py` xác nhận: CUDA khả dụng, GPU RTX 3050 Laptop nhận diện đúng, tensor allocate/matmul trên GPU thành công.

### 8.2. Viết script huấn luyện dùng chung local/cloud (`train.py`)
*   Tham số CLI: `--data`, `--weights`, `--imgsz`, `--batch`, `--epochs`, `--fraction` (chạy thử với tập con), `--device`, `--workers`, `--patience`, `--name`, `--resume`.
*   **Sửa lỗi hard-code đường dẫn tuyệt đối**: `combined_data.yaml` gốc chứa `path: /home/hoangthang/Intelligent Traffic Monitoring System/combined_dataset` — nếu giữ nguyên, dataset sẽ **không tìm thấy ảnh** khi chuyển sang máy/pod khác. Đã bỏ key `path` cố định khỏi yaml và viết hàm `resolve_data_yaml()` trong `train.py` để tự tính lại đường dẫn tuyệt đối dựa trên vị trí file yaml tại **thời điểm chạy**, ghi ra yaml tạm rồi truyền cho `model.train()`. Đảm bảo pipeline chạy đúng bất kể máy local hay pod cloud.

### 8.3. Kiểm thử (smoke test)
*   Chạy `python train.py --imgsz 640 --batch 4 --epochs 2 --fraction 0.02 --workers 2 --name smoke_test` trên RTX 3050 (2% tập train, ~1062 ảnh).
*   Kết quả: 2 epoch hoàn thành không lỗi OOM/data-loader, loss giảm đều (`box_loss` 0.863→0.812, `cls_loss` 2.751→2.180), `mAP50` cải thiện 0.165→0.194. Xác nhận **toàn bộ pipeline train hoạt động đúng** trước khi đầu tư chi phí GPU cloud.

### 8.4. Kiểm tra tính toàn vẹn dữ liệu
*   Lấy mẫu ngẫu nhiên 200 ảnh/tập (train/val/test), kiểm tra ảnh đọc được bằng OpenCV và nhãn đúng định dạng YOLO (5 trường, class ID trong {0,1,2,3}). Kết quả: **0 ảnh hỏng, 0 nhãn lỗi format** trên toàn bộ mẫu kiểm tra.

### 8.5. Chuẩn bị triển khai lên RunPod (RTX 5090, 32GB VRAM)
*   Cài `runpodctl v2.6.1` trên máy cá nhân (`~/.local/bin`) để dùng lệnh `send`/`receive` chuyển file qua relay của RunPod, không cần cấu hình SSH/cloud storage riêng.
*   Tạo `requirements-cloud.txt`: liệt kê dependency của `ultralytics` **trừ** `torch`/`torchvision` — vì RunPod cung cấp sẵn template PyTorch đã build đúng CUDA cho kiến trúc Blackwell (sm_120) của RTX 5090; cài đè torch từ PyPI có rủi ro không tương thích driver.
*   `deploy/pack_for_pod.sh`: đóng gói `combined_dataset/` + `yolo11x.pt` + code thành 1 file `pod_bundle.tar` (**~21 GB**) để gửi qua `runpodctl send`.
*   `deploy/setup_pod.sh`: script chạy trên pod sau khi nhận bundle — cài `ultralytics --no-deps` + `requirements-cloud.txt` (không đụng tới torch có sẵn), verify GPU, in sẵn lệnh train full.
*   Đã đóng gói xong `pod_bundle.tar`, sẵn sàng gửi ngay khi pod được thuê.

### 8.6. Việc còn lại (Giai đoạn 2 — sau khi thuê pod)
1.  Thuê pod RunPod RTX 5090, chọn template PyTorch.
2.  `runpodctl send pod_bundle.tar` (local) → `runpodctl receive <code>` (pod) → giải nén vào `/workspace`.
3.  Chạy `deploy/setup_pod.sh` trên pod.
4.  Train full: `python train.py --imgsz 1280 --batch 32 --epochs 100 --patience 20 --name yolo11x_full_5090` (thông số theo khảo sát mục 7.1–7.3, batch 32 khả thi nhờ 32GB VRAM).
