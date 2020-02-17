import cv2

def get_camera(ip, port, username, password):
    '''
    get the camera object
    '''
    # construct camera URL
    URL = "http://{}:{}@{}/cgi-bin/mjpg/video.cgi?channel=0&subtype=1".format(username, password, ip)

    camera = cv2.VideoCapture(URL)     # returns a VideoCapture object
    camera.set(cv2.CAP_PROP_FPS, 20)   # set the capture rate - not sure this did anyting
    # return the VideoCapture object
    return camera

def configure_cameras(config):
    # -- configure the cameras --
    camera_list = []   # list of camera objects - from which you will capture
    camera_count = 0

    # loop through the cameras found in the json 
    for camera in config['camera']:
        print (camera['name'])
        # get the VideoCapture object for this camera
        capture = (get_camera(
            camera['ip'],
            camera['port'],
            camera['username'],
            camera['password']
        ))
        # create cv2 named windows - window name = camera name
        cv2.namedWindow(camera['name'], cv2.WINDOW_NORMAL)

        # camera tuple = (name, VideoCature object, flip)
        camera_tuple = (camera['name'], capture, camera['flip'])
        camera_list.append(camera_tuple)
    return camera_list