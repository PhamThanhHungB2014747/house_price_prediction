from random import sample
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Đọc dữ liệu
data = pd.read_csv('danang.csv')
data = pd.read_csv('hanoi.csv')
data = pd.read_csv('cantho.csv')
data = pd.read_csv('hochiminh.csv')
data.columns = data.columns.str.strip()

# Chuyển đổi cột giá
def convert_price(price_str):
    price_str = price_str.replace(' triệu', 'e6').replace(',', '').strip()
    return float(price_str) if price_str else np.nan

if 'Giá' in data.columns:
    data['Giá'] = data['Giá'].apply(convert_price)
else:
    print("Cột 'Giá' không tồn tại trong dữ liệu.")
    exit()

# Tách dữ liệu thành đặc trưng và nhãn
X = data.drop(columns=['Giá'])
y = data['Giá'].dropna()

# Xử lý các cột không phải số
X = pd.get_dummies(X, drop_first=True)
X = X.loc[y.index]  # Giữ lại các hàng tương ứng với y

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xử lý giá trị NaN và chuẩn hóa dữ liệu
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Khởi tạo các mô hình
models = {
    'SVM': SVR(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'MLP': MLPRegressor(),
    'Linear Regression': LinearRegression()
}

# Huấn luyện và đánh giá từng mô hình
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_scaled.shape[1] - 1)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Adjusted R²': adj_r2,
    }

# Hiển thị kết quả
results_df = pd.DataFrame(results).T
formatted_results = results_df.copy()
for column in formatted_results.columns:
    formatted_results[column] = formatted_results[column].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else x)

print(formatted_results.to_string(index=False))