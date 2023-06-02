from ultralytics import YOLO
import cv2
import cvzone
import math

cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model=YOLO("C:/Users/ChinmayiRajaram/OneDrive/Desktop/hackotsava/YOLO_Weights/yolov8n.pt")

objNames = ["person", "box", "phone","bottle", "car", "fan", "board", "bag", "chair", "shoe", "laptop", "tennis racket", "book", "pen", "glass", "tie", "zebra","giraffe","motorbike", "aeroplane","plant","clock", "camera","pizza","cake","window","orange","tomato","apple","brinjal","mouse","clip","juicebox","pen"]
while True:
    success, img= cap.read()
    results=model(img, stream=True)
    for r in results:
        boxes =r.boxes
        for box in boxes:
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1),int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            #above opencv
            w=x2-x1
            h= y2-y1
            #x1,y1,w,h = box.xywh[0]
           # bbox = int(x1), int(y1),int(w), int(h)
            cvzone.cornerRect(img,(x1,y1,w,h))
            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            #print(conf)
            
            #above for cvzone
            #ClassName
            cls = int(box.cls[0])
           # print()
            cvzone.putTextRect(img, f'{cls}{conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1) 
            
    
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)