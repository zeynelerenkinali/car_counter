from ultralytics import YOLO
import torch
import cv2
import cvzone
import math

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

mask = cv2.imread("yolo_with_webcam/mask.png")

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 3)

            w, h = x2-x1,y2-y1
            bbox = int(x1),int(y1),int(w),int(h)
            # print(x1,y1,w,h)
            cvzone.cornerRect(imgRegion,bbox, l=5)

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100
            
            # Class Name
            cls = box.cls[0]
            
            currentClass = classNames[int(cls)]
            
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(imgRegion, f'{currentClass} {conf}', (max(13,x1+13),max(30,y1-15)), scale=0.8, thickness=1, offset=3)


    # cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0) # when we turn waitkey to zero we can move by keyboard

