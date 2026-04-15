# Low-Light Object Detection for Autonomous Vehicles

##  Project Overview
This project focuses on developing a real-time object detection system for autonomous vehicles, specifically designed to perform under low-light and nighttime conditions.

The model is trained on synthetic data generated using the CARLA simulator and evaluated in nighttime environments to analyze its robustness under challenging visibility conditions.

---

##  Workflow
The project follows a structured pipeline:

1. Data Collection – Dataset generation using CARLA in good light conditions 
2. Data Augmentation – Simulating low-light conditions  
3. Model Training – Training using YOLOv5  
4. Deployment – Testing in nighttime simulation in CARLA

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- YOLOv5  
- OpenCV  
- CARLA Simulator  
- TensorBoard  

---

## 📊 Dataset
- Generated using CARLA Simulator  
- Total images: ~2240  
- Classes:
  - Vehicle  
  - Pedestrian  
  - Tree  

### Data Preparation
- 80% training / 20% validation split  
- Low-light simulated using:
  - Brightness reduction  
  - Gaussian blur  

---

##  Data Collection

The dataset was generated using CARLA in a controlled simulation environment.

### Key Steps
- Loaded Town maps  
- Enabled synchronous simulation mode  
- Spawned ego vehicle with autopilot  
- Attached sensors:
  - RGB camera  
  - Semantic segmentation camera  
- Spawned pedestrians,trees and vehicles dynamically  

### Detection Pipeline
- Captured frames from sensors  
- Projected 3D objects to 2D image space  
- Detected:
  - Pedestrians (class 12)  
  - Vehicles (class 13–19)  
  - Trees (class 9 via segmentation)  
- Applied Non-Max Suppression (NMS)  
- Generated:
  - Annotated images  
  - Pascal VOC XML labels  

---

##  Annotation Conversion
- Converted VOC (.xml) annotations to YOLO (.txt) format  
- Normalized bounding box coordinates  
- Class mapping:
  - 0 → Vehicle  
  - 1 → Pedestrian  
  - 2 → Tree  

---

##  Model Training

### Setup
- Cloned YOLOv5 repository  
- Installed dependencies  
- Prepared dataset in YOLO format  

### Configuration
- Image size: 640 × 640  
- Epochs: 25  
- Pretrained weights: YOLOv5s  

### Training Command
Run : python train.py --img 640 --batch 16 --epochs 25 --data data.yaml --weights yolov5s.pt --name yolov5s_carla

---

##  Training Monitoring
- Used TensorBoard to track:
  - Loss  
  - Precision  
  - Recall  
  - mAP  

Run : tensorboard --logdir runs/train


---

##  Results
- Precision: 99.3%
- Recall: 95.4%
- mAP@0.5: 98.5%
- mAP@0.5–0.95: 85.7%

### Per-class Performance
- Pedestrians: 90.4%
- Trees: 88.7%
- Vehicles: 77.9% due to:
  - Fewer training samples  
  - Reduced visibility in low light  

### Observations
- Strong detection for pedestrians and trees  
- Lower performance for vehicles due to:
  - Reduced visibility  
  - Fewer training samples  

---

##  Deployment
- Tested in CARLA nighttime environment  
- RGB camera mounted on ego vehicle  

### Process
- Loaded trained YOLOv5 model best pt file   
- Captured real-time frames  
- Performed object detection  
- Displayed bounding boxes with tracking  

### Observations
- Stable detection for pedestrians and trees  
- Vehicles detected with lower confidence in darker regions  

---

##  Key Learnings
- Low-light conditions significantly impact detection performance  
- Data augmentation improves generalization  
- Dataset imbalance affects model accuracy  
- Simulation environments help controlled experimentation  

---

##  Future Improvements
- Balance dataset (especially vehicles)  
- Add more object classes  
- Use real-world low-light datasets  
- Explore advanced models (YOLOv8)  
- Integrate multi-sensor fusion  

---

##  Team Members
- Keerthana  
- Nithya  

---

##  Note
This repository contains a simplified version of a university research project. Confidential implementation details and datasets are not included.
