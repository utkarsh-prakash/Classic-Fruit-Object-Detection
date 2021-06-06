import cv2
import sys
import io
# sys.path.append("D:/Educational/TensorFlow/models/research/")
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import os
from flask import Flask, render_template, request, jsonify, json, redirect, url_for, flash, Response
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import numpy as np
import time

# Activate GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#     # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#         print(e)

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
camera = cv2.VideoCapture(0)

category_index = {
    1: {'id': 1, 'name': 'apple'},
    2: {'id': 2, 'name': 'banana'},
    3: {'id': 3, 'name': 'orange'},
}

def predict(detect_fn, image_path):
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)
    return image_np_with_detections

def predict_from_frame(frame):
    detect_fn = model
    input_tensor = np.expand_dims(frame, 0)
    detections = detect_fn(input_tensor)
    label_id_offset = 1
    image_np_with_detections = frame.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)
    return image_np_with_detections

def gen_frames():
    while True:
        success, frame = camera.read()
        frame = predict_from_frame(frame)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def load_model(model_path):
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(model_path)
    return detect_fn

def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

UPLOAD_FOLDER = 'inference'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app = Flask(__name__, static_folder=os.path.join("webapp", "static"), template_folder=os.path.join("webapp", "templates"))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# model2 is ssd-mobilenet-v2, a light weight model
model_path = 'exported-models/model2/saved_model/'
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main():
    predict_path=request.args.get('path')
    if predict_path == None:
        predict_path = "static/images/predict.png"
    return render_template("web.html", predict_path=predict_path)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')\

@app.route('/video_feed_page', methods = ['POST', 'GET'])
def video_feed_page():
    predict_path=request.args.get('path')
    if predict_path == None:
        predict_path = "static/images/predict.png"
    return render_template("videofeed.html", predict_path=predict_path)

@app.route('/post_data', methods = ['POST', 'GET'])
def post_data():
    if 'image' not in request.files:
        flash('No image uploaded')
        return redirect(url_for('main'))
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = 'upload.jpg'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        flash('No image uploaded')
        return redirect(url_for('main'))
    image_np_with_detections = predict(model, os.path.join(app.config['UPLOAD_FOLDER'], filename))
    for name in os.listdir('webapp/static/images'):
        if name.startswith('inference_'):
            os.remove('webapp/static/images/' + name)
    new_name = "static/images/inference_" + str(time.time()) + ".jpg"
    plt.axis('off')
    plt.imsave("webapp/"+new_name, image_np_with_detections)
    return redirect(url_for('main', path=new_name))

if __name__ == '__main__':
    app.run(debug = False, threaded=False)