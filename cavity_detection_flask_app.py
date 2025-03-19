from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
model = load_model('cavity_detection_model.keras')
image_size = (150, 150)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cavity_detection', methods=['GET', 'POST'])
def cavity_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = predict_cavity(file_path)
            return render_template('cavity_detection.html', prediction=prediction, image_path=file_path)
    return render_template('cavity_detection.html')


def predict_cavity(img_path):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return 'Cavity detected' if prediction[0][0] > 0.5 else 'No cavity detected'


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)