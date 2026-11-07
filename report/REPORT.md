# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: 
- `class_id`=468 ;
- `class_name`="cab" ;
- `rank`=1 ; `score`=0.510915 ;
- `taxonomy_name`="ImageNet-1K"

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record này nói về cả bức ảnh, không phải một vật thể riêng lẻ.

Lý do:
- task: "image_classification" cho biết nhiệm vụ là phân loại ảnh.
- Không có thông tin bounding box như x, y, width, height hoặc bbox.
- Model phân tích toàn bộ hình ảnh và dự đoán lớp phù hợp nhất.

Vì vậy, "cab" là lớp dự đoán đại diện cho nội dung tổng thể của ảnh.


3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Danh sách class không phải do model tự nghĩ ra, mà được xác định bởi taxonomy và cấu hình đi kèm checkpoint.

Trong trường hợp này:
- taxonomy_name: ImageNet-1K.
- model_file: yolo11n-cls.pt.
- Model được huấn luyện để phân loại theo một tập lớp đã xác định trước.
- Mỗi lớp có một class_id và class_name tương ứng.

Model chỉ dự đoán xác suất cho các lớp có sẵn trong danh sách đó, không tự tạo ra lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

Vì 3 trường này có những chức năng khác nhau
- class_id (Mã định danh số của lớp): Giúp hệ thống xử lý dữ liệu bằng mã số ổn định trong phạm vi taxonomy.
- class_name(Tên dễ đọc của lớp): Giúp con người đọc và hiểu kết quả.
- taxonomy_name(Hệ thống phân loại mà lớp đó thuộc về): Giúp xác định chính xác hệ thống phân loại đang sử dụng.

Ví dụ: class_id = 468 trong ImageNet-1K tương ứng với class_name = "cab"

Việc lưu cả ba giúp:
- Tránh nhầm lẫn khi các taxonomy khác nhau có thể sử dụng cùng một id.
- Dễ dàng đối chiếu và kiểm tra dữ liệu.
- Đảm bảo tính nhất quán khi tích hợp nhiều model.
- Hỗ trợ quản lý và tái sử dụng dataset.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

Guideline cần quy định rõ nguyên tắc chọn lớp đại diện cho toàn bộ bức ảnh. Một guideline tốt cần làm rõ:
- Đối tượng chính: Chọn chủ thể chiếm vai trò trung tâm hoặc quan trọng nhất trong ảnh.
- Mức độ nổi bật: Xem xét kích thước, vị trí, độ rõ nét và mức độ thu hút sự chú ý dựa trên mức độ quan trọng trong ngữ cảnh ảnh.
- Ngữ cảnh: Xem xét mục đích hoặc nội dung tổng thể của bức ảnh.
- Quy tắc ưu tiên: Nếu có nhiều đối tượng, cần xác định tiêu chí ưu tiên để tất cả annotator đưa ra kết quả nhất quán.
- Trường hợp không xác định được: Quy định khi nào sử dụng nhãn uncertain, other hoặc chuyển sang bước review (nếu taxonomy hỗ trợ).

Điều quan trọng nhất là hai người gán nhãn độc lập phải có khả năng chọn cùng một lớp dựa trên cùng một bộ quy tắc.

6. Vì sao không được lấy `score` của model làm ground truth?

Vì score chỉ thể hiện mức độ tin tưởng hoặc điểm dự đoán của model đối với một lớp, không chứng minh lớp đó là nhãn đúng thực tế.

Theo output Model dự đoán cab đứng hạng 1:
- cab: 0.510915 (51,09%)
- minibus: 0.164284 (16,43%)
- police_van: 0.085848 (8,58%)

Tuy nhiên, điều này không đồng nghĩa với việc ảnh chắc chắn thuộc lớp cab.

Có thể model dự đoán sai do:
- Model có thể dự đoán sai: Do góc chụp, ánh sáng, chất lượng ảnh hoặc hình ảnh chứa nhiều chủ thể.
- Score không phải ground truth: Đây là kết quả suy luận của model, không phải nhãn được xác minh bởi con người.
- Score có thể không được hiệu chuẩn: Giá trị 0.51 không nhất thiết có nghĩa model đúng 51% trong mọi trường hợp.

Ground truth cần được xác định dựa trên guideline gán nhãn, taxonomy và quá trình kiểm tra chất lượng dữ liệu bởi con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: 
- `class_name`="person" ;
- `score`=0.912624 ;
- `bbox_xyxy`=[ 385.33, 69.24, 498.92, 348.92] ;
- `bbox_width`=113.58 ;
- `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- Hộp bắt đầu tại vị trí (385.33, 69.24).
- Hộp kết thúc tại vị trí (498.92, 348.92).
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

-0.2: Giữ lại 17 vật thể
-0.6: Chỉ giữ lại 6 vật thể có confidence cao hơn.
Khi tăng threshold từ 0.20 lên 0.60, số prediction giảm từ 17 xuống 6, tức giảm 11 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Khi hạ threshold từ 0.60 xuống 0.20, số prediction tăng từ 6 lên 17.
Reviewer có thể gặp:
- Hộp bao quanh vật thể không chính xác.
- Một vật thể bị phát hiện nhiều lần (duplicate box).
- Prediction sai class (nhận diện nhầm vật thể).
- Hộp chỉ bao quanh một phần vật thể.
- Vật thể nhỏ hoặc bị che khuất khó xác định.
Reviewer cần:
- Kiểm tra class có đúng không.
- Kiểm tra bounding box có bao phủ đúng vật thể không.
- Kiểm tra bounding box có bao quanh đúng vật thể không.
- Phát hiện các box trùng lặp hoặc box không hợp lệ.
- Kiểm tra những vật thể nhỏ, khó nhận diện để tránh bỏ sót.
- Đối chiếu với guideline trước khi duyệt nhãn.
Ví dụ: Ở threshold 0.20, model phát hiện thêm spoon, potted plant, dining table và bottle. Reviewer cần kiểm tra xem những vật thể này có thực sự xuất hiện trong ảnh và box có chính xác không.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Quy tắc:
Người gán nhãn phải vẽ bounding box sát với đường biên của vật thể có thể quan sát được, bao gồm toàn bộ phần nhìn thấy của vật thể, không bao gồm nền hoặc các vật thể khác. Tọa độ góc trên bên trái và góc dưới bên phải phải được xác định nhất quán theo định dạng xyxy. Đối với vật thể bị che khuất hoặc cắt khỏi mép ảnh, chỉ đánh dấu phần có thể nhìn thấy theo quy định của guideline.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

- Vật thể bowl có score=0.46 bị che khuất 1 phần
- Tôi chọn escalation lên Lab Coach vì chưa có quy định rõ ràng về việc gán nhãn những vật thể bị che khuất nhiều. Tôi cần xác nhận ngưỡng visibility tối thiểu, cách xác định bounding box và điều kiện loại bỏ vật thể bị che khuất. Sau khi được xác nhận, quy tắc này nên được bổ sung vào guideline để đảm bảo tính nhất quán giữa các annotator.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: 

- `instance_id`="kitchen-009" ; 
- `class_name`="person" ; 
- `score`=0.431437 ;
- số điểm polygon=`polygon_point_count`=65 ;
- 3–5 điểm đầu của `polygon_xy`= [ 1.0, 264.0], [1.0, 309.0], [ 5.0, 309.0], [6.0, 308.0], [15.0, 308.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Bounding box chỉ mô tả vùng hình chữ nhật bao quanh vật thể bằng tọa độ x_min, y_min, x_max, y_max., trong khi polygon mô tả đường biên và hình dạng thực tế của vật thể.

Ví dụ với đối tượng person:
- Bounding box: Cho biết vùng chứa người, bao gồm cả những khoảng nền nằm bên trong hình chữ nhật.
- Polygon: Mô tả đường viền của phần người nhìn thấy.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

instance_id là mã định danh duy nhất cho từng đối tượng cụ thể trong một ảnh hoặc tập dữ liệu.

Ví dụ: 2 cái bát ở 2 vị trí khác nhau có instance_id lần lượt là "kitchen-002" và "kitchen-003", nhưng đều cùng class_id=45, class name=bowl

Vì vậy, instance_id không phải class_id vì nhiều đối tượng khác nhau có thể cùng một class. Nó cũng không phải tracking_id vì nó không mặc định liên kết đối tượng qua nhiều frame video.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Người gán nhãn phải tạo polygon bám sát đường biên thực tế của vật thể, bao phủ toàn bộ vùng pixel nhìn thấy thuộc về đối tượng. Không đưa pixel nền hoặc pixel thuộc instance khác vào mask. Đối với vật thể bị che khuất, chỉ đánh dấu phần có thể quan sát được, không tự suy đoán phần bị che. Khi hai vật thể tiếp xúc hoặc có đường biên không rõ ràng, cần áp dụng quy tắc phân tách đã được dự án phê duyệt; nếu không có quy định, cần escalation lên Lab Coach.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Vị trí: Người ở phía dưới bên trái ảnh kitchen, tại khu vực tiếp giáp với chiếc bàn ăn (dining table).

Tôi chọn: Guideline vì người này bị chiếc bàn che khuất một phần nên không thể quan sát toàn bộ đường biên của cơ thể. Tôi sẽ áp dụng guideline về vật thể bị che khuất, chỉ đánh dấu phần nhìn thấy được và không tự vẽ thêm phần bị che.

Nếu đường biên không thể xác định theo quy tắc hiện có thì mới cần escalation để người phụ trách xem xét.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn class cho toàn bộ ảnh. Đơn vị nhãn: một ảnh tương ứng với một hoặc nhiều class tùy guideline. Ví dụ: cab. | JSON traffic có 5 prediction theo thứ hạng. cab đứng Top-1 với score 0.510915 (51.09%), nhưng chưa đủ chắc chắn để kết luận ảnh chắc chắn là cab. Các class khác như minibus (16.43%) và police_van (8.58%) cho thấy mô hình có thể đang nhầm lẫn giữa các loại phương tiện |Quan sát toàn bộ ảnh PNG, xác định phương tiện chính theo guideline và taxonomy ImageNet-1K. Chọn đúng class dựa trên đặc điểm nhìn thấy, không chọn nhãn chỉ vì model có confidence cao nhất.  | Đối chiếu ảnh PNG với Ground Truth; kiểm tra class_name, class_id, thứ hạng prediction và confidence score. Đặc biệt xác minh cab có thực sự phù hợp hơn minibus hay không. |
| Phát hiện vật thể |Mỗi vật thể có một Bounding Box (tọa độ x_min, y_min, x_max, y_max) và class label theo COCO-80. Đơn vị nhãn: từng object.  | JSON kitchen có 11 prediction tại threshold 0.35, gồm 2 person, 5 bowl, 2 oven và 2 cup. Một số bowl bị che khuất một phần, ví dụ bowl có score 0.464904. Có vật thể nằm sát mép ảnh như oven với x_min = 0.11, person với x_min = 0.12. Khó quyết định ngưỡng visibility tối thiểu và cách vẽ Bounding Box cho vật thể bị che khuất nhiều. |Quan sát PNG, xác định từng vật thể thực tế, vẽ Bounding Box theo ranh giới phần vật thể nhìn thấy và gán class phù hợp. Khi chưa có quy định rõ về vật thể bị che khuất, escalation lên Lab Coach thay vì tự đặt quy tắc.  | Kiểm tra class, số lượng vật thể, tọa độ Bounding Box, vật thể bị bỏ sót, phát hiện trùng và các trường hợp bị che khuất. Đối chiếu PNG với guideline, đặc biệt xác minh quy tắc visibility và cách xử lý vật thể ở mép ảnh. |
| Instance segmentation | Mỗi đối tượng có một nhãn riêng gồm instance_id, class_id, class_name và polygon mask bao quanh đường biên vật thể. | Người ở góc dưới bên trái bị bàn che khuất. JSON có score = 0.431437, polygon có 65 điểm. Biên vật thể khó xác định chính xác do bị che. | Vẽ polygon theo đường biên phần nhìn thấy được, không tự suy đoán phần bị che. Gán đúng class và instance riêng biệt. | Kiểm tra class, polygon có sát biên vật thể không, có lấn sang nền hoặc vật thể khác không và các đối tượng có bị bỏ sót không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Tôi chỉ sử dụng dữ liệu và hình ảnh được cung cấp cho mục đích thực hiện bài lab, không tự ý chia sẻ, sao chép hoặc tải dữ liệu lên các nền tảng bên ngoài khi chưa được cho phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Tôi sẽ dừng xử lý, không tự ý chỉnh sửa hoặc chia sẻ dữ liệu và báo ngay cho giảng viên hoặc người phụ trách lab để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
