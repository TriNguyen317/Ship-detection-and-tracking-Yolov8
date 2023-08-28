from fastapi import FastAPI, File, UploadFile, Form, Response
import cv2
from ultralytics import YOLO
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

from main import Draw
app = FastAPI()

model = YOLO("./Model/Boat-detect-medium.pt")

# def detect(model,img, conf=0.3, iou_thresh=0.45):
#     result = model(img, iou=iou_thresh)

#     boxes = result[0].boxes  # Boxes object for bbox outputs
#     conf_detect=boxes.conf.cpu().numpy()
#     box_detect=boxes.xyxy.cpu().numpy()
#     idx=np.where(conf_detect>conf)
#     return box_detect[idx], conf_detect[idx]

# def Draw(model,img):
#     img=np.array(img)
#     boxes, conf =detect(model,img)
#     for num, i in enumerate(boxes):
#         img=cv2.rectangle(img,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255, 0, 0),2)
#         cv2.putText(img, str(conf[num]), (int(i[0]),int(i[1]-3)),cv2.FONT_HERSHEY_SIMPLEX, 0.75 ,(255, 0, 0) ,2,cv2.LINE_AA)
    
#     #cv2.imwrite("68-detect.png",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return img



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), 
                             Path_model: str = Form(default="./Model/Boat-detect-medium.pt"), 
                             imgsz: int = Form(default=640),
                             conf: float = Form(default=0.5),
                             iou: float = Form(default=0.45)):
    
    data=await file.read()
    img= Image.open(io.BytesIO(data)).convert("RGB")
    img=np.array(img)
    detect_img=Draw(model,img)
    detect_img=cv2.cvtColor(detect_img,cv2.COLOR_RGB2BGR)
    _,detect_img=cv2.imencode(".png",detect_img)
        
    response= Response(content=detect_img.tobytes(), media_type="image/png")
    
    return response