import os, io
import requests
from flask import Flask, request, jsonify

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from PIL import Image
import numpy as np
import cv2


CLASS_NAMES = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'HP',
               'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite',
               'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']


app = Flask(__name__)
app.config['SECRET_KEY'] = 'some-secret-for-logo-recognition'


@app.route("/")
def check():
    return jsonify({'Message': "Server Running!"})


@app.route("/predict", methods=['POST'])
def precict_logo():

    try:
        file = request.files['image']
        pil_img = Image.open(file)

        numpy_image = np.array(pil_img)

        # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
        # the color is converted from RGB to BGR format
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        predictor = load_flickr27_model()

        outputs = predictor(img)
        outputs = outputs['instances'].get_fields()

        # pred_boxes = outputs['pred_boxes'].tensor.to(device).numpy()
        pred_boxes = outputs['pred_boxes']
        # scores = outputs['scores'].to(device).numpy()
        scores = outputs['scores']
        # pred_classes = outputs['pred_classes'].to(device).numpy()
        pred_classes = outputs['pred_classes']

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
        return jsonify({'Message': "API Failed"})



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



# predictor = load_flickr27_model('cpu')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)

