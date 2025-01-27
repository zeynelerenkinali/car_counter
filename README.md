# Car Counter using YOLOv8

A simple car counting project that uses YOLOv8 for vehicle detection and tracking. This program counts the number of cars in a video by detecting them in each frame and tracking them as they move across the frame.

## Features

- Detects cars using YOLOv8.
- Tracks cars and assigns unique IDs to each.
- Counts cars as they pass through a defined region of interest (ROI).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/car-counter-yolov8.git
   cd car-counter-yolov8
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 weights:**

   Download YOLOv8 weights (e.g., `yolov8n.pt`) by following the instructions in the [Ultralytics YOLOv8 GitHub repository](https://github.com/ultralytics/ultralytics).

## Usage

1. **Run the car counter:**

   ```bash
   python car_counter.py --input video.mp4 --output result.mp4 --roi 300
   ```

   - `--input`: Path to the input video file.
   - `--output`: Path to save the output video with the car count.
   - `--roi`: The region of interest (e.g., a pixel line on the y-axis where the car is counted).

2. **Output video:**

   The program will process the video and save an output video with bounding boxes, tracking IDs, and car counts displayed.

## License

This project is licensed under the MIT License.
