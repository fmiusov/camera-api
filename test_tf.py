import os
import sys
import cv2
import numpy as np


# add some paths that will pull in other software
# -- don't add to the path over and over
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)

import tensorflow_util
import label_map_util     # this came from tensorflow/models

# if you made it this far, you python path is good - and you successfully inported

# test the label map util
d = label_map_util.get_label_map_dict('model/mscoco_label_map.pbtxt', 'id')
print (d)

# test getting a model
interpreter = tensorflow_util.get_tflite_interpreter('model/output_tflite_graph.tflite')
model_image_dim, model_input_dim, output_details = tensorflow_util.get_tflite_attributes(interpreter)
print ("Model Image Dim:", model_image_dim)
print ("Model Image Dim:", model_input_dim)

# load an impage & preprocess from a file
image = tensorflow_util.load_image_into_numpy_array('jpeg_images/111-1122_IMG.JPG')
cv2.namedWindow('raw_image', cv2.WINDOW_NORMAL)
cv2.imshow('raw_image', image)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# preprocess the image
preprocessed_image = tensorflow_util.preprocess_image(image, interpreter, model_image_dim, model_input_dim)
print (type(preprocessed_image), preprocessed_image.shape)

# send the preprocessed image to the model
bbox_data, class_data, prob_data = tensorflow_util.send_image_to_model(preprocessed_image, interpreter)