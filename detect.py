import numpy as np
import cv2

# Task detect image

def detectImg(model,img, conf=0.5, iou_thresh=0.45):
    result = model(img, iou=iou_thresh)

    boxes = result[0].boxes  # Boxes object for bbox outputs
    conf_detect = boxes.conf.cpu().numpy()
    box_detect = boxes.xyxy.cpu().numpy()
    idx = np.where(conf_detect > conf)
    return box_detect[idx], np.round_(conf_detect[idx], decimals=3)


# Draw
def Draw(model,img):
    boxes, conf = detectImg(model,img)
    for num, i in enumerate(boxes):
        img = cv2.rectangle(img, (int(i[0]), int(i[1])), 
                        (int(i[2]), int(i[3])), (255, 0, 0), 2)
        cv2.putText(img, str(conf[num]), (int(i[0]), 
                        int(i[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


# Task detect video
def detectVideo(args, model):
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
    video = cv2.VideoCapture(args.input)
    if (video.isOpened() == False):
        print("Error reading video file")
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    print(size)
    result = cv2.VideoWriter("./Data/Output/"+args.output+".avi",
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, size)
    while (True):
        ret, frame = video.read()
        if ret == True:
            frame = Draw(model,frame)
            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    video.release()
    result.release()
    cv2.destroyAllWindows()
    print("The video was successfully detected")
    print("The video was successfully saved")