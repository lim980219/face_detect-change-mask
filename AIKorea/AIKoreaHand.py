###############################################################################
### Simple demo on gesture recognition
### Input : Live video of hand
### Output: 2D display of hand keypoint
###         with gesture classification
### Usage : python 02_gesture.py -m train (to log data)
###       : python 02_gesture.py -m eval  (to perform gesture recognition)
###############################################################################
import cv2
#import argparse

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition

#parser = argparse.ArgumentParser()
#parser.add_argument('-m', '--mode', default='eval', help='train / eval')
#args = parser.parse_args()
#mode = args.mode
mode = 'eval'
imgchoice = ""
# Load mediapipe hand class
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=1)
# Load display class
disp = DisplayHand(max_num_hands=2)
# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0
# Load gesture recognition class
gest = GestureRecognition(mode)
#counter = 0

def AIKoreaHands(img):

    global imgchoice
    imgchoice="none"
    # Flip image for 3rd person view
    img = cv2.flip(img, 1)

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feedforward to extract keypoint
    param = pipe.forward(img)
    if (param[0]['class'] is not None) and (mode == 'eval'):
        param[0]['gesture'] = gest.eval(param[0]['angle'])

        # my custom
        imgchoice = param[0]['gesture']

    img.flags.writeable = False

    return img, imgchoice

if __name__=="__main__":
    AIKoreaHands(cap)
    pipe.pipe.close()
    cap.release()