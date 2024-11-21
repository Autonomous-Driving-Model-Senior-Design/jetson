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
import serial

from wba import WeBACNN
from dataset import dataset
from train_model import train_model
from sklearn.preprocessing import MinMaxScaler 
from calc_turn_angle import turn_angle

ser = serial.Serial(
    port='/dev/ttyTHS1',  # UART port
    baudrate=115200,        # Baud rate
    timeout=1             # Read timeout
)

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=160,
    capture_height=160,
    display_width=160,
    display_height=160,
    framerate=10,
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

def send_data(data):
    if ser.isOpen():
        print("Sending data...")
        ser.write(data)  # Convert string to bytes and send
    else:
        print("Serial port not open.")

def show_camera():
    window_title = "CSI Camera"
    device = torch.device("cuda:0")
    model = WeBACNN(device)
    model.load_state_dict(torch.load("new_model_light.pt", map_location=device))
    model.to(device)

    angle_list = np.ones(3) * -1 
    previous_angle = 80 
    idx = 0 

    with torch.no_grad():
        model.eval()
        video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if video_capture.isOpened():
            while True:
                st = time.time()
                _, frame = video_capture.read()
                temp = frame
                frame = torch.from_numpy(frame) 
                frame = frame.to(device) 

                data = frame / 255
                data = data.to(device)
                data = data.permute((2, 0, 1)).float() 

                data = torch.reshape(data, (1, 3, 160, 160))
                predictions = model(data)

                # post processing 
                thresh = 0.5  # Post-processing threshold
                pred = predictions[0].permute(1, 2, 0) # still on device 
                pred[pred < thresh] = 0  # Set values below threshold to 0
                pred[pred >= thresh] = 1  # Set values above threshold to 1
                # print(f"post-processing time: {time.time() - st}\n")
                pred = pred.cpu().detach().numpy()
                # st1 = time.time()

                # moving average  
                angle, speed = turn_angle(pred) 
                if idx < 10: speed = 77 # add initial boost (dan chi) 
                # idx %= 3 
                previous_angle = angle_list[(idx+2)%3] 
                angle_list[idx%3] = angle 
                control_angle_list = np.delete(angle_list, np.where(angle_list == -1)) # remove invalid entries 
                if len(control_angle_list) == 0: 
                    angle = previous_angle
                    if angle < 0: angle = 80 
                else: 
                    angle = np.average(control_angle_list) 
                idx += 1
                if idx == 99: idx = 0 
                print(f"angle: {angle:.5f}")
                print(f"speed: {speed}")
                assert(angle >= 40 and angle <= 120)
                # if(angle < 40 or angle > 120):
                #     print(angle)
                #     return
                # # send data 
                # speed = 75
                # angle = 40
                # # print(f"post-processing time angle: {time.time() - st1}\n")
                x = hex((speed << 8) + int(angle))
                print(x)
                send_data(x.encode()) # UART
                print("--------------------------------\n")

                # # display prediction result 
                pred = pred * 255
                pred = pred.astype(np.uint8) 
                cv2.imshow("frame", temp) 
                cv2.imshow("prediction", pred)  
                cv2.waitKey(1)
                torch.cuda.empty_cache() 
        else:
            print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
