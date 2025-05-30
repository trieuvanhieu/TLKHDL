# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

print("Đang đọc dữ liệu...")
data = pd.read_csv("mnist_train.csv")
X = data.drop("label", axis=1)
y = data["label"]

print("Huấn luyện mô hình...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Lưu mô hình...")
dump(model, "mnist_model.pkl")
print("✅ Xong.")
