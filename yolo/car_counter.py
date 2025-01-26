from ultralytics import YOLO
import torch
import cv2
import cvzone
import math
from sort import *
cap = cv2.VideoCapture("videos/cars.mp4") # for video
if not cap.isOpened():
    print("Error: Could not open video file.")


model = YOLO("yolo_weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#Check is cuda available
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

mask = cv2.imread("yolo/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = (223, 375, 673, 375)

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Boxs
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 3)

            w, h = x2-x1,y2-y1
            bbox = int(x1),int(y1),int(w),int(h)
            # print(x1,y1,w,h)

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100
            
            # Class Name
            cls = box.cls[0]
            
            currentClass = classNames[int(cls)]
            
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(13,x1+13),max(30,y1-15)), scale=0.8, thickness=1, offset=3)
                # cvzone.cornerRect(img,bbox, l=5, rt=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2) 
        #cvzone.putTextRect(img, f'{currentClass} {Id}', (max(13,x1+13),max(30,y1-15)), scale=0.8, thickness=1, offset=3)
        cvzone.cornerRect(img, bbox, l=5, rt=3, colorR=(255,0,0))
        cvzone.putTextRect(img, f' {int(id)}', (max(13,x1+13),max(30,y1-15)), scale=2, thickness=3, offset=10)
        print(result)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy), 5, (255,0,255), cv2.FILLED)


    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0) # when we turn waitkey to zero we can move by keyboard

