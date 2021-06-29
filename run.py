from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from flask import request
from os import environ
from flask import jsonify
import os,io
import requests
from PIL import Image
import cv2
device='cpu'

import numpy as np
CLASS_NAMES = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'HP',
               'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite',
               'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']

from PIL import Image
from flask import Flask
#from config import Config
#from flask_migrate import Migrate, MigrateCommand, Manager
#from flask_cors import CORS

# Declaring Constants
FLASK_HOST = environ.get("FLASK_HOST") or '0.0.0.0'
from flask import Flask, request, jsonify

def load_flickr27_model():
    cfg = get_cfg()

    model_file = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.WEIGHTS = r'model_final.pth'
    cfg.OUTPUT_DIR = "OUTPUT_DIR_RETINA"
    cfg.MODEL.RETINANET.NUM_CLASSES =27
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6
    # Create a predictor using the trained model
    predictor = DefaultPredictor(cfg)
    return predictor


app = Flask(__name__)

@app.route("/")
def check():
    return jsonify({'Message': "Server Running!"})

@app.route("/im_labels",methods= ['POST'])
def process_image():
    try:
        file = request.files['image']
        # Read the image via file.stream
        pil_img = Image.open(file.stream)
        numpy_image = np.array(pil_img)
        # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
        # the color is converted from RGB to BGR format
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        predictor = load_flickr27_model()
        outputs = predictor(img)
        outputs = outputs['instances'].get_fields()
        pred_boxes = outputs['pred_boxes'].tensor.to(device).numpy()
        scores = outputs['scores'].to(device).numpy()
        pred_classes = outputs['pred_classes'].to(device).numpy()
        class_names = []
        for cls in pred_classes:
            class_names.append(CLASS_NAMES[int(cls)])

        return jsonify({

            'num_detections' : len(pred_boxes),
            'pred_boxes' : str(pred_boxes),
            'scores' : str(scores),
            'class_names':class_names,

        })

    except:
        return jsonify({  'Message': "API Failed :("})


if __name__ == "__main__":
    app.run(debug=True)


