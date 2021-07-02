import os, io, uuid
import requests
from flask import Flask, request, jsonify, render_template, send_file

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from PIL import Image, ImageDraw, ImageFont

import numpy as np
import cv2


CLASS_NAMES = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'HP',
               'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite',
               'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']


app = Flask(__name__)
app.config['SECRET_KEY'] = 'some-secret-for-logo-recognition'
app.model = None



# @app.route("/")
# def check():
#     return jsonify({'Message': "Server Running!"})


@app.route("/")
def upload_photo():
    return render_template('upload.html')


@app.route("/init")
def init_model():
    init_model()
    return "Init done"


@app.route("/file/<file_name>")
def display_file(file_name):
    print ("in get file")


    file_path = get_file_path(file_name)
    return send_file(file_path)


@app.route("/predict", methods=['POST'])
def precict_logo():
    result = []

    files = request.files.getlist("images")
    for file in files:

        pil_img = Image.open(file)
        numpy_image = np.array(pil_img)

        # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
        # the color is converted from RGB to BGR format
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        predictor = get_app_model()

        outputs = predictor(img)
        outputs = outputs['instances'].get_fields()

        pred_boxes = outputs['pred_boxes'].tensor.numpy()
        scores = outputs['scores'].numpy()
        pred_classes = outputs['pred_classes']

        class_names = []
        for cls in pred_classes:
            class_names.append(CLASS_NAMES[int(cls)])

        source_img = Image.open(file).convert("RGBA")
        draw = ImageDraw.Draw(source_img)

        for i in range(0, len(pred_boxes)):
            shape = pred_boxes[i]
            score = scores[i]
            class_name = class_names[i]

            draw.rectangle(shape, outline="white", width=2)

            text = class_name + " " + str(round(score, 2))
            draw.text((shape[0], shape[3] + 10), text)

        img_name = str(uuid.uuid4()) + ".jpg"
        img_full_name = get_file_path(img_name)

        source_img = source_img.convert('RGB')
        source_img.save(img_full_name)


        result.append({
                        'file': "/file/"+ img_name,
                        'num_detections': len(pred_boxes),
                        'pred_boxes': str(pred_boxes),
                        'scores': str(scores),
                        'class_names': class_names,
                    })


        print ("result ", result)


    return render_template('result.html', result=result)


def get_app_model():
    return init_model()


def init_model():
    if (app.model == None):
        print ("In init model")
        app.model = load_flickr27_model()

    else:
        print ("Model initialized")

    return app.model


def load_flickr27_model(device="cpu"):
    cfg = get_cfg()

    model_file = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = r'model_final.pth'
    cfg.OUTPUT_DIR = "OUTPUT_DIR_RETINA"
    cfg.MODEL.RETINANET.NUM_CLASSES = 27
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4

    # Create a predictor using the trained model
    predictor = DefaultPredictor(cfg)

    return predictor


def get_file_path(file_name):
    return os.path.join("./uploads", file_name)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)

