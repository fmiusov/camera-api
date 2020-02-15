# camera-api
RESTful API for Amcrest cameras

## Python 3.7
pip install flask  
pip install opencv-python  

Tensorflow or just the TFLite Interpreter?  Well, you will be using a lot of tf.* utilities  
pip install tensorflow-gpu==1.15 

## get the TensorFlow utils & model => ~/projects
This will also compile the protobufs  
bash ./install_tf_models.sh

you need the label map:  
cp research/object_detection/data/mscoco_label_map.pbtxt ~/projects/camera-api/model/  

you need a tflite model - easiest place to get that is from s3  
you should have created it using the ssd-dag/UnderstandingTensorRT_ConvertGraph notebook  


