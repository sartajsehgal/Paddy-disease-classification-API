import io
import numpy as np
import tensorflow as tf
#import cv2
from PIL import Image
from flask import Flask, jsonify, request 

model=tf.keras.models.load_model('paddy_classification.h5')

img_size=128

def prepare_img(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((img_size, img_size))
    img = np.array(img)
    img = img.astype('float32')
    img = img.reshape(-1,img_size,img_size,3)
    img /= 255
    return img

num_label={
    0:'tungro',
    1:'hispa',
    2:'downy_mildew',
    3:'bacterial_leaf_streak', 
    4:'bacterial_leaf_blight',
    5:'brown_spot',
    6:'blast',
    7:'normal',
    8:'dead_heart',
    9:'bacterial_panicle_blight'
}

def get_name(x):
    return num_label[x]

def predict_result(img):
    y_pred = model.predict(img)
    num = np.argmax(y_pred)
    return get_name(num)

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def infer_image():
    print(request.files.get('file'))
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    file = request.files.get('file')

    if not file:
        return
    
    img_bytes = file.read()
    img = prepare_img(img_bytes)
    return jsonify(prediction=predict_result(img))

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0')