from amcrest import AmcrestCamera
from PIL import Image
import io
import numpy
import cv2
import time

camera = AmcrestCamera("192.168.1.109",80,"admin","uwasat0ad").camera
cv2.namedWindow('Amcrest', cv2.WINDOW_NORMAL)

for i in range(100):

    try:
        response = camera.snapshot(0)
        print (i, "success")
        img_bytes = response.read()
        img_array = Image.open(io.BytesIO(img_bytes))
        img_numpy = numpy.array(img_array)
        cv2.imshow('Amcrest', img_numpy)
    except:
        print (i, "error")

    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

