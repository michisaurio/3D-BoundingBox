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
from yolo.yolo import cv_Yolo
from torch_lib import Model
from torch_lib.Dataset import DetectedObject, generate_bins

import os
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
from torchvision.models import vgg, VGG19_BN_Weights

def main():
    # inputs
    model_folder = 'bin_02_overlap_00_kitty'
    model_file = 'epoch_100.pkl'
    image_folder = 'single_ship_001'
    #image_folder = 'jesus_car'

    # load yolo
    yolo = cv_Yolo(confidence=0.5, threshold=0.3)

    # load model
    weights_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights', model_folder, model_file)
    checkpoint = torch.load(weights_path, weights_only=True)
    try:
        number_bins = checkpoint['number_bins']
    except:
        print('Number of bins not in checkpoint. Using default value 2')
        number_bins = 2
    angle_bins = generate_bins(number_bins)
    my_vgg = vgg.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
    model = Model.Model(features=my_vgg.features, bins=number_bins).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #load images
    image_set_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'eval', image_folder)
    calibration_file_path = os.path.join(image_set_path, 'calib_cam_to_cam.txt') #TODO: Find out actual calibration matrix
    image_folder_path = os.path.join(image_set_path, 'images')
    image_ids = [x.split('.')[0] for x in sorted(os.listdir(image_folder_path))]
    image_file_type = [x.split('.')[1] for x in sorted(os.listdir(image_folder_path))][0]

    orientation_gt = []
    orientation_est = []

    for image_id in image_ids:
        true_heading_deg = np.asarray(image_id, dtype=float) - 1.0
        print(true_heading_deg)
        image_file_path = os.path.join(image_folder_path, image_id + '.' + image_file_type)
        truth_img = cv2.imread(image_file_path)
        img = np.copy(truth_img) # TODO:necessary?
        yolo_img = np.copy(truth_img) # TODO:necessary?

        detections = yolo.get_detections(yolo_img)

        for detection in detections:
            try:
                detectedObject = DetectedObject(img, 'Car', detection.box_2d, calibration_file_path)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            #alpha = np.pi/2 - alpha

            orientation_gt.append(true_heading_deg)
            orientation_est.append(np.rad2deg(alpha))

    print(orientation_gt)
    print(orientation_est)
    plt.scatter(orientation_gt, orientation_est)
    
    plt.show()

if __name__ == '__main__':
    main()