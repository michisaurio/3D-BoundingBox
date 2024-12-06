"""
Use YOLO to obtain 2D boxes around ship image
Use chosen Pytorch trained model to estimate heading
Input:
1. Path to folder with images
    Images are assumed to show a single object. The same object in each picture
    Image filenames have the structure [integer].[file extension]
    The string [integer]%360 gives the observation angle in degrees
    In other words, the images in the folder show a timelapse, where the camera moves around the boat; or the boat rotates
2. Path to .pkl file that contains the trained model
Output:
1.
2.
"""
import os
from yolo.yolo import cv_Yolo

def main():
    # inputs
    model_folder = ''
    model_file =

    # load yolo
    yolo = cv_Yolo(confidence=0.5, threshold=0.3)

    # load model
    weights_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights')



if __name__ == '__main__':
    main()