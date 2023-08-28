import cv2
import time
from tracker.byte_tracker import BYTETracker
from Ship import Ship_manager
from detect import detectImg
import numpy as np


# Task tracking
def Tracking(args, model):
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
        
    '''
    #Read video
    tracker = BYTETracker(args)
    ArrayShip = Ship_manager(args.track_buffer)
    video = cv2.VideoCapture(args.input)
    FPS = 30
    if (video.isOpened() == False):
        print("Error reading video file")
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    print(size)
    result = cv2.VideoWriter("./Data/Output/"+args.output+".avi",
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             FPS, size)
    since = time.time()
    frame_count = 0
    
    while (True):
        ret, frame = video.read()
        if ret == True:
            # Return bbox and conf from model
            boxes, conf = detectImg(model,frame, args.conf, args.iou_threshold)
            online_targets = tracker.update(conf, boxes, size, size)

            ArrayShip.update(online_targets, frame_count)
            if frame_count % (FPS*1.5) == 0:
                ArrayShip.check_state()
                ArrayShip.update_bbox(online_targets)
            frame_count += 1

            for i in ArrayShip.list_ship:
                if i.is_activate == True:
                    x = int(i.bbox[0]+i.bbox[2])
                    y = int(i.bbox[1]+i.bbox[3])
                    state = "MOVING" if i.is_move == True else "STOP"
                    frame = cv2.rectangle(
                        frame, (int(i.bbox[0]), int(i.bbox[1])), (x, y), (255, 0, 0), 3)
                    cv2.putText(frame, 'conf= '+str(i.score), (int(i.bbox[0]), int(i.bbox[1]-3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'id= '+str(i.track_id), (int(i.bbox[0]), y+3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                    if i.is_move == True:
                        state += ": 0 s"
                    else:
                        elapsed = time.time() - since
                        process_fps = frame_count / elapsed
                        ratio = process_fps/FPS
                        timeend = time.time()
                        state += ": " + \
                            str(np.round_((timeend-i.timestart)
                                * ratio, decimals=2))+" s"
                    cv2.putText(frame, 'time: '+str(state), (x, y+3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            result.write(frame)
            cv2.imshow("a", frame)
            if cv2.waitKey(1) & 0xFF == ord("c"):
                break
        else:
            break
    video.release()
    result.release()
    cv2.destroyAllWindows()
    print("The video was successfully tracked")
    print("The video was successfully saved")