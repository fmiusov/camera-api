from flask import Flask, request, jsonify
from flask_marshmallow import Marshmallow

import os
import cv2
import base64

HOME = os.path.expanduser("~")
SAMPLE_IMAGE_PATH = os.path.join(HOME, "projects/ssd-dag/data/new_jpeg_images/", "111-1122_IMG.JPG")

# Init Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def get():

    image = cv2.imread(SAMPLE_IMAGE_PATH)
    image_dict = {
        'source': '111-1122_IMG.JPG',
        'bboxes' : [(23,100,200,110,220)],
        'image' : base64.b64encode(image)
    }
    return jsonify(image_dict)

# Run Server
if __name__ == '__main__':
    app.run(debug=True)