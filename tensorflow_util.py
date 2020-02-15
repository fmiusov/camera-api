import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

# add some paths that will pull in other software
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)


from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
#import tflite_runtime.interpreter as tflite

def get_tflite_interpreter(model_path):
    print ('TF Lite Model loading...')
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()  # failure to do this will give you a core dump
    return interpreter

def get_tflite_attributes(interpreter):
    # using the interpreter - get some of the model attributes
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
        
    model_input_shape = input_details[0]['shape']   # (batch, h, w, channels)
    model_image_dim = (model_input_shape[1], model_input_shape[2])    # model - image dimension
    model_input_dim = (1, model_input_shape[1], model_input_shape[2], 3) # model - batch of images dimensions
    print ("Model Input Dimension: {}".format(model_input_dim))
    return model_image_dim, model_input_dim, output_details

def send_image_to_model(preprocessed_image, interpreter):
    # model inference
    start_time = time.time()
    # input (image) is (1,300,300,3) - shaped like a batch of size 1
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)    # model input is a batch of images
    interpreter.invoke()        # this invokes the model, creating output data

    # model has created an inference from a batch of data
    # - the model creates output like a batch of size = 1
    # - size must be 1, so simplify the shape by taking first row only
    #   [0] at the end effectivtly means bbox (1,10,4) becomes (10,4)
    bbox_data = interpreter.get_tensor(output_details[0]['index'])[0]
    class_data = interpreter.get_tensor(output_details[1]['index'])[0]
    prob_data = interpreter.get_tensor(output_details[2]['index'])[0]

    finish_time = time.time()
    print("time spent: {:.4f}".format(finish_time - start_time))

    return bbox_data, class_data, prob_data

def load_image_into_numpy_array(image_path):
    image = cv2.imread(image_path)
    # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image, interpreter, model_image_dim, model_input_dim):
    resized_image = cv2.resize(image, model_image_dim, interpolation = cv2.INTER_AREA)  # resized to 300x300xRGB
    reshaped_image = np.reshape(resized_image, model_input_dim)                         # reshape for model (1,300,300,3)
    return reshaped_image
