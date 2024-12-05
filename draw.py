1# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Đọc dữ liệu
# # data = pd.read_csv('hanoi.csv')
# data = pd.read_csv('hochiminh.csv')
# # data = pd.read_csv('danang.csv')
# # data = pd.read_csv('cantho.csv')

# # Kiểm tra tên cột và xóa khoảng trắng thừa
# data.columns = data.columns.str.strip()
# # print("Tên cột:", data.columns.tolist())

# # Xóa cột 'Tiêu đề' nếu nó tồn tại
# data = data.drop(columns=['Tiêu đề'], errors='ignore')

# # Chuyển đổi cột 'Giá'
# def convert_price(price_str):
#     price_str = price_str.replace(' triệu', 'e6').replace(',', '').strip()
#     try:
#         return float(price_str)
#     except ValueError:
#         return np.nan

# data['Giá'] = data['Giá'].apply(convert_price)

# # Kiểm tra và chuyển đổi cột 'Số phòng' nếu cần
# if data['Số phòng'].dtype != 'int64' and data['Số phòng'].dtype != 'float64':
#     data['Số phòng'] = pd.to_numeric(data['Số phòng'], errors='coerce')

# # Loại bỏ các dòng có giá trị NaN trong các cột quan trọng
# data = data.dropna(subset=['Số phòng', 'Giá', 'Diện tích'])

# # # Kiểm tra thống kê cho các cột
# # print(data[['Giá', 'Diện tích']].describe())
# # print("Các giá trị duy nhất trong 'Vị trí':", data['Vị trí'].unique())

# # Lọc theo giá và diện tích
# data_filtered = data[(data['Giá'] <= 100) & (data['Diện tích'] <= 500 )]

# # # Kiểm tra số dòng trong data_filtered
# # print("Số dòng trong data_filtered:", data_filtered.shape[0])

# # Tạo cột chú thích cho vị trí
# data_filtered['Chú thích'] = data_filtered['Vị trí'] + ' (Số phòng: ' + data_filtered['Số phòng'].astype(str) + ')'

# # Vẽ biểu đồ điểm
# plt.figure(figsize=(12, 8))
# scatter = sns.scatterplot(data=data_filtered, x='Diện tích', y='Giá', size='Số phòng', hue='Vị trí', sizes=(20, 200), legend='full')

# # Tùy chỉnh chú thích
# handles, labels = scatter.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))

# plt.legend(by_label.values(), by_label.keys(), title='Chú thích', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# plt.title('Biểu đồ thể hiện sự phân bố giá nhà cửa theo khu vực ở Hồ Chí Minh')
# plt.xlabel('Diện tích (m²)')
# plt.ylabel('Giá (tỷ đồng)')
# plt.grid(True)
# plt.tight_layout()

# # Lưu biểu đồ
# plt.savefig('bieu_do_gia_nha_ho_chi_minh.png', dpi=300, bbox_inches='tight')  # Lưu với độ phân giải 300 dpi

# plt.show()

2# import pandas as pd
# import xgboost as xgb
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split

# # Hàm để đọc và xử lý dữ liệu
# def load_and_prepare_data(file_name):
#     data = pd.read_csv(file_name)
#     data.columns = data.columns.str.strip()

#     # Hàm để chuyển đổi giá
#     def convert_price(price):
#         try:
#             if 'triệu' in price:
#                 return float(price.replace(' triệu', '').replace('.', '').replace(',', '.')) * 1e6
#             elif 't' in price:
#                 return float(price.replace('t', '').replace('.', '').replace(',', '.')) * 1e6
#             else:
#                 return float(price.replace('.', '').replace(',', '.'))
#         except ValueError:
#             return None

#     # Chuyển đổi giá
#     data['Giá'] = data['Giá'].apply(convert_price)
#     data.dropna(subset=['Giá', 'Số phòng', 'Vị trí', 'Diện tích', 'Loại hình'], inplace=True)

#     # Chuyển đổi các nhãn thành số (encode)
#     data['Vị trí'] = data['Vị trí'].astype('category').cat.codes
#     data['Loại hình'] = data['Loại hình'].astype('category').cat.codes

#     # Xóa các cột không cần thiết
#     data.drop(columns=['Tiêu đề', 'Unnamed: 7'], errors='ignore', inplace=True)

#     # Chuyển đổi cột 'Giá / Diện tích' nếu cần
#     if 'Giá / Diện tích' in data.columns:
#         data['Giá / Diện tích'] = pd.to_numeric(data['Giá / Diện tích'], errors='coerce')

#     return data

# # Đọc và chuẩn bị dữ liệu từ các file
# files = ['danang.csv', 'hochiminh.csv', 'hanoi.csv', 'cantho.csv']
# dataframes = [load_and_prepare_data(file) for file in files]
# data = pd.concat(dataframes, ignore_index=True)

# # Chia dữ liệu thành các biến đầu vào (X) và đầu ra (y)
# X = data.drop('Giá', axis=1)  # Các cột đặc trưng
# y = data['Giá']  # Cột mục tiêu

# # Chia dữ liệu thành tập huấn luyện và kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Xây dựng mô hình XGBoost với enable_categorical
# model = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True)
# model.fit(X_train, y_train)

# # Tính độ quan trọng của các đặc trưng
# importance = model.feature_importances_

# # Tạo DataFrame cho độ quan trọng
# importance_df = pd.DataFrame({'Đặc trưng': X.columns, 'Độ quan trọng': importance})

# # Sắp xếp theo độ quan trọng
# importance_df = importance_df.sort_values(by='Độ quan trọng', ascending=False)

# # # In độ quan trọng
# # print("Độ quan trọng của các đặc trưng:")
# # print(importance_df)

# # Vẽ biểu đồ độ quan trọng
# plt.figure(figsize=(10, 6))
# barplot = sns.barplot(x='Độ quan trọng', y='Đặc trưng', data=importance_df, palette='viridis')
# plt.title('Độ quan trọng của các đặc trưng trong mô hình XGBoost')
# plt.xlabel('Độ quan trọng')
# plt.ylabel('Đặc trưng')

# # Thêm giá trị số vào mỗi ô
# for p in barplot.patches:
#     barplot.annotate(f'{p.get_width():.3f}', 
#                      (p.get_width(), p.get_y() + p.get_height() / 2), 
#                      ha='left', va='center')

# plt.tight_layout()

# # Lưu biểu đồ
# plt.savefig('do_quan_trong_cua_nhan_trong_mo_hinh_xgboost.png', dpi=300, bbox_inches='tight')  # Lưu với độ phân giải 300 dpi

# plt.show()

3# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor

# # Đường dẫn tới các file CSV
# files = ['cantho.csv', 'hanoi.csv', 'danang.csv', 'hochiminh.csv']

# # Danh sách để lưu trữ dữ liệu
# data_frames = []

# # Đọc dữ liệu từ từng file
# for file in files:
#     df = pd.read_csv(file)
#     data_frames.append(df)

# # Kết hợp các DataFrame
# combined_data = pd.concat(data_frames, ignore_index=True)

# # Hàm để chuyển đổi giá
# def convert_price(price):
#     if isinstance(price, str):
#         price = price.replace(' triệu', '000000').replace(' tỷ', '000000000').replace(',', '')
#         return float(price)
#     return price

# # Áp dụng hàm convert_price cho cột 'Giá'
# combined_data['Giá'] = combined_data['Giá'].apply(convert_price)

# # Kiểm tra và loại bỏ các giá trị NaN hoặc không hợp lệ
# combined_data['Giá'] = combined_data['Giá'].replace([np.inf, -np.inf], np.nan)  # Thay thế vô cực bằng NaN
# combined_data = combined_data.dropna(subset=['Giá', 'Diện tích'])  # Xóa các hàng có giá trị NaN trong cột 'Giá' và 'Diện tích'

# # Chọn các đặc trưng và nhãn (target)
# X = combined_data[['Số phòng', 'Diện tích']]  # Thay đổi theo các cột trong dữ liệu
# y = combined_data['Giá']

# # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Khởi tạo mô hình XGBoost
# model = XGBRegressor(objective='reg:squarederror', n_estimators=100)

# # Huấn luyện mô hình
# model.fit(X_train, y_train)

# # Dự đoán trên tập kiểm tra
# y_pred = model.predict(X_test)

# # Vẽ biểu đồ phân bố giá dự đoán
# plt.figure(figsize=(10, 6))
# sns.histplot(y_pred, bins=30, kde=False, color='blue', stat='density', alpha=1)
# plt.title('Phân bố giá dự đoán từ mô hình XGBoost')
# plt.xlabel('Giá dự đoán (triệu đồng)')
# plt.ylabel('Mật độ')
# plt.xlim(0, 4000)  # Giới hạn trục x từ 0 đến 4000
# plt.ylim(0, 0.0025)  # Giới hạn trục y nếu cần thiết
# plt.grid(True)

# # Hiển thị biểu đồ
# plt.savefig('bieu_do_gia_tri_du_daon_mo_hinh_xgboost.png', dpi=300, bbox_inches='tight')  # Lưu với độ phân giải 300 dpi

# plt.show()

4# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Đường dẫn tới các file CSV
# files = ['cantho.csv', 'hanoi.csv', 'danang.csv', 'hochiminh.csv']

# # Danh sách để lưu trữ dữ liệu
# data_frames = []

# # Đọc dữ liệu từ từng file
# for file in files:
#     df = pd.read_csv(file)
#     data_frames.append(df)

# # Kết hợp các DataFrame
# combined_data = pd.concat(data_frames, ignore_index=True)

# # Hàm để chuyển đổi giá
# def convert_price(price):
#     if isinstance(price, str):
#         price = price.replace(' triệu', '000000').replace(' tỷ', '000000000').replace(',', '')
#         return float(price)
#     return price

# # Áp dụng hàm convert_price cho cột 'Giá'
# combined_data['Giá'] = combined_data['Giá'].apply(convert_price)

# # Kiểm tra và loại bỏ các giá trị NaN hoặc không hợp lệ
# combined_data['Giá'] = combined_data['Giá'].replace([np.inf, -np.inf], np.nan)  # Thay thế vô cực bằng NaN
# combined_data = combined_data.dropna(subset=['Giá', 'Diện tích'])  # Xóa các hàng có giá trị NaN trong cột 'Giá' và 'Diện tích'

# # Tính giá điều chỉnh theo diện tích
# def adjust_price(row):
#     if row['Diện tích'] == 20:
#         return row['Giá'] / 1000000
#     else:
#         return row['Giá'] 

# combined_data['Giá điều chỉnh'] = combined_data.apply(adjust_price, axis=1)

# # Tính giá trung bình theo diện tích
# average_price_by_area = combined_data.groupby('Diện tích')['Giá điều chỉnh'].mean().reset_index()

# # Làm tròn giá chỉ lấy 3 chữ số thập phân
# average_price_by_area['Giá điều chỉnh'] = average_price_by_area['Giá điều chỉnh'].round(3)
# # print(average_price_by_area)
# # Vẽ biểu đồ cột
# plt.figure(figsize=(10, 6))
# sns.barplot(data=average_price_by_area, x='Giá điều chỉnh', y='Diện tích', color='green')
# plt.title('Giá theo diện tích')
# plt.xlabel('Gía')
# plt.ylabel('Diện tích (m²)')

# # Giới hạn trục y với một khoảng cách thích hợp
# plt.xlim(0, 10)
# plt.savefig('bieu_do_gia_theo_dien_tich.png', dpi=300, bbox_inches='tight')  # Lưu với độ phân giải 300 dpi

# plt.show()