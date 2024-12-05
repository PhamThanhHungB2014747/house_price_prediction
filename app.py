from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Tạo từ điển để lưu trữ dữ liệu cho từng khu vực
datasets = {
    'Cần Thơ': 'cantho.csv',
    'Hà Nội': 'hanoi.csv',
    'Đà Nẵng': 'danang.csv',
    'TP Hồ Chí Minh': 'hochiminh.csv'
}

# Tạo từ điển để lưu trữ đường dẫn hình ảnh cho từng khu vực
images = {
    'Cần Thơ': 'static/bieu_do_gia_nha_can_tho.png',  # Thay đổi đường dẫn cho hình ảnh của Cần Thơ
    'Hà Nội': 'static/bieu_do_gia_nha_ha_noi.png',
    'Đà Nẵng': 'static/bieu_do_gia_nha_da_nang.png',
    'TP Hồ Chí Minh': 'static/bieu_do_gia_nha_ho_chi_minh.png'
}

chart = {
    'Biểu đồ giá theo diện tích': 'static/bieu_do_gia_theo_dien_tich.png',
    'Biểu đồ giá theo loại hình': 'static/bieu_do_gia_theo_loại_hinh.png',
    'Biểu đồ giá theo số phòng': 'static/bieu_do_gia_theo_so_phong.png',
    'Biểu đồ giá nhà Hà Nội': 'static/bieu_do_gia_nha_ha_noi.png',
    'Biểu đồ giá nhà Đà Nẵng': 'static/bieu_do_gia_nha_da_nang.png',
    'Biểu đồ giá nhà Hồ Chí Minh': 'static/bieu_do_gia_nha_ho_chi_minh.png',
    'Biểu đồ giá nhà Cần Thơ': 'static/bieu_do_gia_nha_can_tho.png',
    'Biểu đồ dự đoán mô hình XGBoost': 'static/bieu_do_gia_tri_du_doan_mo_hinh_xgboost.png',
    'Độ quan trọng của nhân trong mô hình XGBoost': 'static/do_quan_trong_cua_nhan_trong_mo_hinh_xgboost.png'
}

# Hàm đọc dữ liệu
def load_data(region):
    data = pd.read_csv(datasets[region])
    data.columns = data.columns.str.strip().str.lower()
    data['giá'] = data['giá'].apply(clean_price)
    return data

def clean_price(price_str):
    price_str = price_str.replace(' ', '').replace(',', '.')
    try:
        return float(price_str)
    except ValueError:
        return np.nan

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    no_price_message = None
    data = None  # Biến để lưu trữ dữ liệu theo khu vực
    image_path = None  # Biến để lưu đường dẫn hình ảnh

    if request.method == 'POST':
        region = request.form.get('region')
        property_type = request.form.get('property_type')
        district = request.form.get('district')
        num_rooms = request.form.get('num_rooms')
        area = request.form.get('area')

        if region:
            data = load_data(region)  # Tải dữ liệu theo khu vực đã chọn

            # Lấy đường dẫn hình ảnh tương ứng với khu vực
            image_path = images.get(region)

            if property_type and district and num_rooms and area:
                num_rooms = int(num_rooms)
                area = float(area)

                filtered_data = data[data['vị trí'] == district]

                if property_type == "Nhà ở":
                    avg_price_per_area = (filtered_data[filtered_data['loại hình'] == 'nhà ở']['giá'] / 
                                          filtered_data[filtered_data['loại hình'] == 'nhà ở']['diện tích']).mean()
                elif property_type == "Chung cư":
                    avg_price_per_area = (filtered_data[filtered_data['loại hình'] == 'chung cư']['giá'] / 
                                          filtered_data[filtered_data['loại hình'] == 'chung cư']['diện tích']).mean()

                room_filtered_data = filtered_data[filtered_data['số phòng'] == num_rooms]
                if not room_filtered_data.empty:
                    avg_price_per_area = (room_filtered_data['giá'] / room_filtered_data['diện tích']).mean()
                    predicted_price = avg_price_per_area * area
                else:
                    no_price_message = "Không có giá dự đoán."

    return render_template('index.html', predicted_price=predicted_price, no_price_message=no_price_message, image_path=image_path)

@app.route('/charts', methods=['GET'])
def charts():
    return render_template('charts.html', charts=chart)

if __name__ == '__main__':
    app.run(debug=True)