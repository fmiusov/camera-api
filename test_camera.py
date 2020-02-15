import cv2

import camera_util
import gen_util

# get the app config - including passwords
config = gen_util.read_app_config('app_config.json')

# -- configure the cameras --
camera_list = []   # list of camera objects - from which you will capture
camera_count = 0

# loop through the cameras found in the json 
for camera in config['camera']:
    print (camera['name'])
    # get the VideoCapture object for this camera
    capture = (camera_util.get_camera(
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


while True:

    for name, capture, flip in camera_list:
        #print('About to start the Read command')
        ret, frame = capture.read()

        if flip == "vert":
            frame = cv2.flip(frame, 0)
        
    

        cv2.imshow(name,frame)
        #print('Running..')

        # Use key 'q' to close window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
