import torch
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics import YOLO
from detect import *
from tracking import Tracking

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


if __name__ == "__main__":
    '''
    Args: 
        imgsz (int): Input of size image. Defaut: 640
        input (str): Path of input data. Defaut: 337.png
        output (str): Path of output data. Defaut: output
        model (str): Path of model. Default: ./Model/Boat-detect-medium.pt
        conf (float): Score confidence. Default: 0.6
        iou_threshold (float): IOU threshold. Default: 0.5
        video (bool): Input is video. Default: False
        detect (bool): Task is detection. Default: False
        tracking (bool): Task is tracking. Default: False
        track_buffer (int): buffer to calculate the time when to remove tracks. Default: 30
        match_thresh (float): Matching threshold for tracking in bytetrack. Default: 0.5
        time_check_state (float): Time to update state of ship (second). Default: 1.5
        train (bool): Task is training. Default: False
        epoch (int): Num of epoch. Default: 50
    '''
    parser = argparse.ArgumentParser(prog='Boat-detect',
                                     epilog='Text at the bottom of help')
    parser.add_argument("-imgsz", type=int,
                        default=640, help="Size img")
    parser.add_argument("-input", type=str,
                        default="337.png", help="Path of input data")
    parser.add_argument("-output", type=str,
                        default="output", help="Path of output data")
    parser.add_argument("-model", type=str,
                        default="./Model/Boat-detect-medium.pt", help="Path of model")
    parser.add_argument("-conf", type=float,
                        default=0.6, help="Score confidence")
    parser.add_argument("-iou_threshold", type=float,
                        default=0.5, help="IOU threshold")
    parser.add_argument("-video", type=bool, action=argparse.BooleanOptionalAction,
                        default=False, help="Confirm input is a Video")
    parser.add_argument("-detect", type=bool, action=argparse.BooleanOptionalAction,
                        default=True, help="Activate task detection")
    parser.add_argument("-tracking", type=bool, action=argparse.BooleanOptionalAction,
                        default=False, help="Activate task tracking")
    parser.add_argument("-track_buffer", type=float,
                        default=30, help="buffer to calculate the time when to remove tracks")
    parser.add_argument("-match_thresh", type=float,
                        default=0.5, help="Matching threshold for tracking in bytetrack")
    parser.add_argument("-time_check_state", type=float,
                        default=1.5, help="Time to update state of ship")
    parser.add_argument("-train", type=bool, action=argparse.BooleanOptionalAction,
                        default=False, help="Task is training model")
    parser.add_argument("-epoch", type=int,
                        default=50, help="Num epochs")
    
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.empty_cache()
    # Read model from file .pt
    model = YOLO(args.model)
    if args.train:
        model.train(data="data.yaml", epochs=args.epoch, imgsz=args.imgsz, single_cls=True)
    else:
        # Detect and tracking on video
        if args.video == True:
            if args.tracking == True:
                Tracking(args, model)
            else:
                detectVideo(args, model)
        # Detect on image
        else:
            img = cv2.imread(args.input)
            detect_img = Draw(model, img)
            cv2.imwrite("./Output/"+args.output+".jpg", detect_img)
            print("The image was successfully detected")
            print("The image was successfully saved")
