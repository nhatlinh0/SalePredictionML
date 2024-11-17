# Dự đoán doanh số bán hàng
Dự đoán doanh số bán hàng bằng kỹ thuật học máy
<br/>
<br/>
# 1. Kaggle Thư mục
Thư mục Kaggle chứa tập dữ liệu đầu vào
<br/>
# 2. Thư mục dữ liệu dự đoán
Chứa ***submission.csv*** có tất cả dữ liệu được dự đoán sau khi đào tạo mô hình
<br/>
# 3. savedModels.json
Tệp JSON chứa liên kết của Mô hình hồi quy rừng ngẫu nhiên mà chúng tôi đã đào tạo và lưu. Mô hình có dung lượng 1,22 GB và do đó đã được liên kết thay vì được tải trực tiếp lên GitHub.
<br/>

## Mô tả tập dữ liệu

### stores.csv
Tệp này chứa thông tin ẩn danh về 45 cửa hàng, cho biết loại và quy mô của cửa hàng.
### train.csv
Đây là dữ liệu đào tạo lịch sử, bao gồm từ ngày 05/02/2010 đến ngày 01/11/2012. Trong tập tin này, bạn sẽ tìm thấy các trường sau:
* Store - số lượng cửa hàng
* Dept - số lượng bộ phận
* Date - tuần
* Weekly_Sales -  doanh số bán hàng cho bộ phận nhất định trong cửa hàng nhất định
* IsHoliday - liệu tuần đó có phải là tuần nghỉ lễ đặc biệt không
### test.csv
Tệp này giống hệt với train.csv, ngoại trừ việc chúng tôi đã giữ lại doanh số bán hàng hàng tuần. Bạn phải dự đoán doanh số bán hàng cho từng bộ ba cửa hàng, bộ phận và ngày trong tệp này.
### features.csv
Tệp này chứa dữ liệu bổ sung liên quan đến hoạt động của cửa hàng, bộ phận và khu vực trong những ngày nhất định. Nó chứa các trường sau:
Store - số lượng cửa hàng
* **Date** - tuần
* **Temperature** - nhiệt độ trung bình khu vực
* **Fuel_Price** - giá nhiên liệu trong khu vực
* **MarkDown1-5** - chương trình giảm giá khuyến mại
* **CPI** - chỉ số giá tiêu dùng
* **Unemployment** - tỉ lệ thất nghiệp
* **IsHoliday** - liệu tuần đó có phải là tuần nghỉ lễ đặc biệt không

Để thuận tiện, bốn ngày nghỉ lễ rơi vào các tuần tiếp theo trong tập dữ liệu (không phải tất cả các ngày nghỉ lễ đều có trong dữ liệu):
* **Super Bowl**: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
* **Labor Day**: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
* **Thanksgiving**: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
* **Christmas**: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

## Weighted Mean Absolute Error
Các mô hình sẽ được đánh giá dựa trên sai số tuyệt đối trung bình có trọng số (WMAE):\
<img src="https://latex.codecogs.com/gif.latex?\huge&space;WMAE&space;=&space;\frac{1}{\sum&space;w_{i}}\sum_{i=1}^{n}w_{i}\left&space;|&space;y_{i}&space;-&space;\widehat{y_{i}}&space;\right&space;|" title="\huge WMAE = \frac{1}{\sum w_{i}}\sum_{i=1}^{n}w_{i}\left | y_{i} - \widehat{y_{i}} \right |" />
