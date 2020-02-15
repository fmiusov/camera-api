import os
import sys
import time
import cv2
import numpy as np

import camera_util
import gen_util
import tensorflow_util
import label_map_util 
import display

# add some paths that will pull in other software
# -- don't add to the path over and over
cwd = os.getcwd()
models = os.path.abspath(os.path.join(cwd, '..', 'models/research/'))
slim = os.path.abspath(os.path.join(cwd, '..', 'models/research/slim'))
sys.path.append(models)
sys.path.append(slim)


PROBABILITY_THRESHOLD = 0.6    # only display objects with a 0.6+ probability


def main():
    # get the app config - including passwords
    config = gen_util.read_app_config('app_config.json')

    # set up camerass
    camera_list = camera_util.configure_cameras(config)

    # set up tflite model
    label_dict = label_map_util.get_label_map_dict(config['label_map'], 'id')
    interpreter = tensorflow_util.get_tflite_interpreter('model/output_tflite_graph.tflite')
    model_image_dim, model_input_dim, output_details = tensorflow_util.get_tflite_attributes(interpreter)

    while True:

        # for name, capture, flip in camera_list:
            name, capture, flip = camera_list[0]
            print(name)

            ret, frame = capture.read()

            if flip == "vert":
                frame = cv2.flip(frame, 0)
            if False:
                # pre-process the frame -> a compatible numpy array for the model
                preprocessed_image = tensorflow_util.preprocess_image(frame, interpreter, model_image_dim, model_input_dim)
                bbox_array, class_id_array, prob_array = tensorflow_util.send_image_to_model(preprocessed_image, interpreter)

                inference_image, orig_image_dim, detected_objects = display.inference_to_image( 
                        frame,
                        bbox_array, class_id_array, prob_array, 
                        model_input_dim, label_dict, PROBABILITY_THRESHOLD)

                # enlarged_inference = cv2.resize(inference_image, (1440, 1440), interpolation = cv2.INTER_AREA)
                cv2.imshow(name, inference_image)
            else:
                cv2.imshow(name, frame)

            time.sleep(3)

            # Use key 'q' to close window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
