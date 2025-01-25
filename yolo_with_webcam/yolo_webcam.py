from ultralytics import YOLO
import torch
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # for webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("videos/motorbikes-1.mp4") # for video
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
print(torch.cuda.is_available())
print(torch.cuda.device_count())

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 3)

            # w, h = x2-x1,y2-y1
            # bbox = int(x1),int(y1),int(w),int(h)
            # print(x1,y1,w,h)
            # cvzone.cornerRect(img,bbox)

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100
            
            # Class Name
            cls = box.cls[0]

            # Draw Box
            if cls == 0: # detect only person
                cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(13,x1+13),max(30,y1-15)), scale=2, thickness=3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

