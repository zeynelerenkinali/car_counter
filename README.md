# car_counter
Car Counter using YOLOv8

This project implements a car counting system using the YOLOv8 object detection model. The system is capable of detecting and tracking vehicles in video feeds and counting them as they pass through a defined region of interest (ROI).
Table of Contents

    Overview
    Features
    Installation
    Usage
    Results
    Contributing
    License

Overview

This project utilizes YOLOv8, the latest version of the popular YOLO (You Only Look Once) object detection algorithm, to detect and track vehicles in a video stream. The main objective is to count the number of cars passing through a specific region of the frame.
How it works:

    Object Detection: YOLOv8 is used to detect cars in each frame of the video.
    Object Tracking: A tracking algorithm is employed to assign unique IDs to detected vehicles, allowing for continuous tracking between frames.
    Counting Logic: Vehicles that pass through the defined region of interest are counted.

Features

    Real-time car detection and tracking
    Vehicle counting based on region of interest (ROI)
    Customizable detection and tracking parameters
    Easy-to-use interface for video processing

Installation

    Clone the repository:

git clone https://github.com/yourusername/car-counter-yolov8.git
cd car-counter-yolov8

Install the dependencies:

    pip install -r requirements.txt

    Download YOLOv8 weights:

    Download YOLOv8 weights (e.g., yolov8n.pt) by following the instructions in the Ultralytics YOLOv8 GitHub repository.

Usage

    Run the car counter:

    python car_counter.py --input video.mp4 --output result.mp4 --roi 300

        --input: Path to the input video file.
        --output: Path to save the output video with the car count.
        --roi: The region of interest (e.g., a pixel line on the y-axis where the car is counted).

    Output video:

    The program will process the video and save an output video with bounding boxes, tracking IDs, and car counts displayed.

License

This project is licensed under the MIT License.
