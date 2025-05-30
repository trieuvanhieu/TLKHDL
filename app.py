from flask import Flask, render_template, request
from utils import (
    load_model, load_test_data, predict_digit, draw_confusion_matrix,
    process_uploaded_image, get_per_class_accuracy, get_classification_report_html
)
import os
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model và dữ liệu test
model = load_model()
X_test, y_test = load_test_data()
accuracy = accuracy_score(y_test, model.predict(X_test))  # ✅ Độ chính xác tổng thể
per_class_accuracy = get_per_class_accuracy(model, X_test, y_test)  # ✅ Độ chính xác từng chữ số
classification_report_html = get_classification_report_html(model, X_test, y_test)  # ✅ Bảng phân loại chi tiết

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    show_cm = False  # ✅ Cờ để hiển thị biểu đồ confusion matrix

    if request.method == "POST":
        if "draw_cm" in request.form:
            draw_confusion_matrix(model, X_test, y_test)
            show_cm = True

        elif "upload" in request.form:
            file = request.files["image"]
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                img_vector = process_uploaded_image(filepath)
                prediction = model.predict([img_vector])[0]
                image_url = filepath.replace("\\", "/")

    return render_template("index.html",
                           prediction=prediction,
                           image_url=image_url,
                           accuracy=round(accuracy * 100, 2),
                           per_class_accuracy=per_class_accuracy,
                           classification_report=classification_report_html,
                           show_cm=show_cm)

if __name__ == "__main__":
    app.run(debug=True)
