from amcrest import AmcrestCamera
from PIL import Image
import io
import numpy
import cv2
import time
import base64

camera = AmcrestCamera("192.168.1.109",80,"admin","uwasat0ad").camera
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

