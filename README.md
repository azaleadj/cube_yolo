#  Rubik’s Cube Detection using YOLOv8

This project trains a custom YOLOv8 model to detect Rubik’s Cubes in real-time using a camera feed.  
It includes both **model training** and **live object detection** code, and can be connected to a robot for further manipulation tasks.

---
##  Project Structure
cube_yolo/
├── data.yaml # Dataset configuration
├── best.pt # Trained YOLOv8 model weights
├── robot_cube_detection.ipynb # Training + real-time detection notebook
├── test_realsense.py # Camera test script (for Intel RealSense)
├── yolo_realsence_detect.py # Real-time detection with YOLO + RealSense

---

##  Useage
### 1. Test camera connection
python test_realsense.py

### 2. Run YOLO detection
python yolo_realsence_detect.py

---
Results

Precision: 0.997

Recall: 0.965

mAP@0.5: 0.994

mAP@0.5:0.95: 0.809

The trained model can accurately detect both solved and unsolved Rubik’s cubes under varied lighting conditions.

---
Author

Jane (Juan Du)
Volunteer researcher at UVic ACIS Lab
Focus: robot vision, human–robot interaction, and visual data analysis
