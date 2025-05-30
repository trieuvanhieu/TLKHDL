import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from PIL import Image
from joblib import load
from sklearn.metrics import confusion_matrix, classification_report

def load_model():
    return load("mnist_model.pkl")

def load_test_data():
    test = pd.read_csv("mnist_test.csv")
    X_test = test.drop("label", axis=1)
    y_test = test["label"]
    return X_test, y_test

def predict_digit(model, digit, X_test):
    return model.predict(X_test[X_test.index == digit].values)

def predict_from_image(model, image_path):
    img = Image.open(image_path).convert("L")    # chuyển sang ảnh xám
    img = img.resize((28, 28))                   # resize về 28x28
    img_array = np.array(img)
    img_array = 255 - img_array                  # đảo màu: chữ trắng nền đen
    img_vector = img_array.flatten() / 255.0     # chuẩn hóa về 0-1
    return model.predict([img_vector])[0]

# Xử lý ảnh tải lên từ máy tính
def process_uploaded_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))  # grayscale
    img_array = np.array(img)
    img_array = 255 - img_array               # 🔁 Đảo màu: chữ trắng trên nền đen
    img_vector = img_array.flatten() / 255.0  # chuẩn hóa
    return img_vector

def draw_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png")

# ✅ Hàm tính độ chính xác theo từng lớp (chữ số)
def get_per_class_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    accuracy_per_class = {str(i): correct[i] / total[i] if total[i] != 0 else 0.0 for i in range(10)}
    return accuracy_per_class

# ✅ Hàm xuất classification_report thành bảng HTML
def get_classification_report_html(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    html_report = "<pre style='font-size: 14px;'>" + report + "</pre>"
    return html_report
