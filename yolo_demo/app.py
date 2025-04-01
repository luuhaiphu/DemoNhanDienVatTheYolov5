import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

# Đảm bảo thư mục 'uploads' tồn tại trong 'static'
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Kiểm tra định dạng file ảnh
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Tải mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Đọc ảnh và chạy nhận diện vật thể
        img = cv2.imread(file_path)
        results = model(img)  # Chạy YOLOv5 nhận diện vật thể

        # Lưu ảnh có đánh dấu nhận diện
        result_image = results.render()[0]  # Đánh dấu kết quả lên ảnh
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        cv2.imwrite(result_image_path, result_image)

        # Trả về tên file ảnh kết quả để hiển thị
        return render_template('index.html', filename='result_' + filename)

if __name__ == '__main__':
    app.run(debug=True)
