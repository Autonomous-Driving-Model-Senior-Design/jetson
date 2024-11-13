import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import time
import gc

from wba import WeBACNN
from dataset import dataset
from train_model import train_model
from sklearn.preprocessing import MinMaxScaler 



def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=960,
    display_height=540,
    framerate=5,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    window_title = "CSI Camera"
    device = torch.device("cuda:0")
    model = WeBACNN()
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)

    with torch.no_grad():
        model.eval()
        video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if video_capture.isOpened():
            while True:
                st = time.time()
                _, frame = video_capture.read()
                temp = frame
                frame = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_LINEAR)
                frame = torch.from_numpy(frame) 
                frame = frame.to(device) 

                data = frame / 255
                data = data.to(device)
                data = data.permute((2, 0, 1)).float() 

                data = torch.reshape(data, (1, 3, 160, 160))
                # start_time = time.time() 
                predictions, _, _ = model(data)

                thresh = 0.5  # Post-processing threshold
                pred = predictions[0].permute(1, 2, 0) # still on device 

                pred[pred < thresh] = 0  # Set values below threshold to 0
                pred[pred >= thresh] = 1  # Set values above threshold to 1
                pred = pred * 255
                pred = pred.cpu().detach().numpy()
                pred = pred.astype(np.uint8) 
                #print(f"\nthresh_pred: {thresh_pred}\n")
                #print(f"\nthresh_pred.shape: {thresh_pred.shape}\n")
                cv2.imshow("frame", temp) 
                cv2.imshow("prediction", pred) 
                # print(int(thresh_pred*255))
                # cv2.imshow("frmae", frame)
                cv2.waitKey(1)
                # plt.imshow(thresh_pred)
                print(f"post-processing time: {time.time() - st}\n")

        else:
            print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
