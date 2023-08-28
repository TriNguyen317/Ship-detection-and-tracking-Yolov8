# Ship Detection Project

This project focuses on ship detection in images and videos using computer vision techniques and the YOLO (You Only Look Once) algorithm implemented with the Ultralytics library. It provides an API for users to upload images and receive the detected ship images as a response.

## Features

- Ship detection in images and videos.
- FastAPI-based API for easy integration and usage.
- Utilizes the YOLO algorithm for accurate ship detection.
- Supports image and video inputs.
- Provides bounding box visualization of detected ships.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [API Documentation](#API-Documentation)
- [Script Documentation](#Script-Documentation)
- [Examples](#examples)


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TriNguyen317/Ship-detection-and-tracking.git

   ```

2. Install the required dependencies:

3. Download the YOLO model weights and place them in the appropriate directory:

   - You can download the model weights file or use existing weights in the `Model` directory.

## API Documentation

1. Start the API server:

   ```bash
   uvicorn API:app --host 0.0.0.0 --port 8000

   ```

2. Access the API at `http://localhost:8000/docs#/default/create_upload_file_uploadfile__post` and click `Try it out` button upload an image file containing ships.

The API will process the input file, perform ship detection, and return the image or video with bounding boxes indicating the detected ships.

The API provides the following endpoint:

### Upload File \[/uploadfile/\]

- Description: Uploads an image or video file for ship detection.
- Parameters:
  - `file` (file): The image or video file to be uploaded.
  - `Path_model` (string, optional): The path to the YOLO model weights file. Default: `./Model/Boat-detect-medium.pt`.
  - `imgsz` (integer, optional): The image size for processing. Default: `640`.
  - `conf` (float, optional): Confidence threshold for ship detection. Default: `0.6`.
  - `iou` (float, optional): IOU (Intersection over Union) threshold for ship detection. Default: `0.45`.
- Response:

## Script Documentation

### Command-line Arguments
The project supports the following command-line arguments:

- `-imgsz`: Size of the image (default: 640)
- `-input`: Path to the input file (default: "170740.mp4")
- `-output`: Path for the output file (default: "track")
- `-model`: Path to the model file (default: "./Model/Boat-detect-medium.pt")
- `-conf`: Score confidence threshold (default: 0.6)
- `-iou_threshold`: IOU threshold (default: 0.5)
- `-video`: Flag indicating if the input is a video (default: False)
- `-detect`: Activate the detection task (default: True)
- `-tracking`: Activate the tracking task (default: False)
- `-track_buffer`: Buffer to calculate when to remove tracks (default: 30)
- `-match_thresh`: Matching threshold for tracking (default: 0.5)
- `-time-check-state`: Time to reset state (default: 1.5)
- `-train`: Task is training (default: False)
- `-epoch`: Num epochs (default: 50)

Explain each argument in detail, including its purpose, default value, and any constraints or limitations.

### Config data
```
- Data
   - train
      - images
      - label
   - valid 
      - images
      - label
   - test
      - images
      - label
```

- Link data: https://drive.google.com/file/d/1c46R47X17maEfEUvCb8T6snlHiNLsGzx/view?usp=sharing

### Config data.yaml file
```
train: Path to train data
val: Path to valid data
test: Path to test data

nc: num of class
names: Array of name each class
```

### Train

```bash
python main.py -train -epoch 50
```

### Detection and Tracking

```bash
python main.py -detect -input 377.png -output output
```

```bash
python main.py -tracking -video -input 10.mp4 -output output
```

## Acknowledgments

- Providing the YOLO implementation.
- Task track and detect ship with YOLOv8
- The FastAPI framework for creating the API server.
- Any other resources or references that have been used in this project.

## Contact

For any inquiries or questions, please contact:

- Project Maintainer: Nguyen Dinh Tri (dinhtrikt11102002@gmail.com)
- Project Homepage: [https://github.com/TriNguyen317/Ship-detection-and-tracking]

Feel free to reach out with any feedback or suggestions!