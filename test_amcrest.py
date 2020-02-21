from amcrest import AmcrestCamera
from PIL import Image
import io
import numpy
import cv2
import time
import base64


# testing w/ 2496

# 720p
# bit rate 1536
# fps 1    0.95  0
# fps 5    0.95  0
# fps 10   0.94, 0 errors
# fps 20   0.95  0
# fps 30   0.94  0 

# bit rate 2048  15 fps  0.94
# bit rate 512           0.94

# frame interval 30 (default)
# 15 fps  15 interval  0.94
# 15 fps  30 interval  0.94
# 15 fps 150 interval  0.94


# 4K   15 fps  0.75
# 4K   10 fps
# 4K    2 fps  0.75, slower lag
#      15 fps  4096 bit   0.75

# 1080   15 fps, 0.93


# Bcamera = AmcrestCamera("192.168.1.109",80,"admin","uwasat0ad").camera
camera = AmcrestCamera("192.168.1.114",80,"admin","jg00dman").camera
cv2.namedWindow('Amcrest', cv2.WINDOW_NORMAL)

# works
# http://admin:uwasat0ad@192.168.1.109/cgi-bin/snapshot.cgi?channel=1
# http://admin:uwasat0ad@192.168.1.109/cgi-bin/snapshot.cgi

# http://admin:uwasat0ad@192.168.1.109/cgi-bin/mjpg/video.cga

# http://admin:uwasat0ad@192.168.1.109/cgi-bin/devAudioInput.cga?action=getCollect

# works
# http://admin:uwasat0ad@192.168.1.109/cgi-bin/configManager.cgi?action=getConfig&name=Snap

data = {
    'username' : 'admin',
    'password' : 'uwasat0ad'
}

error_count = 0
success_count = 0
for i in range(5000):
    start_time = time.time()
    print (start_time)

    try:
        response = camera.snapshot(0)
        print (i, "success", time.time() - start_time)
        success_count = success_count + 1
        img_bytes = response.read()
        img_array = Image.open(io.BytesIO(img_bytes))
        img_numpy = numpy.array(img_array)
        img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        print (img_bgr.shape)
        cv2.imshow('Amcrest', img_bgr)
    except:
        print (i, "error")
        error_count = error_count + 1

    # time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print ("Success: {}  Error: {}".format(success_count, error_count))

# https://stackoverflow.com/questions/18139093/base64-authentication-python

api_URL = "http://192.168.1.109/cgi-bin/snapshot.cgi"
usrPass = 'admin:uwasat0ad'
b64Val = base64.b64encode(usrPass)
r=requests.post(api_URL, 
                headers={"Authorization": "Basic %s" % b64Val},
                )

from requests.auth import HTTPBasicAuth
r = requests.post(api_URL, auth=HTTPBasicAuth('admin', 'uwasat0ad')) #, data=payload)
r.request.headers['Authorization']

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

    # time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

