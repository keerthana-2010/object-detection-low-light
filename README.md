#########################  Low-Light Object Detection for Autonomous Vehicles  ###########################

##  Project Overview
This project focuses on developing a real-time object detection system for autonomous vehicles, specifically designed to perform under low-light and nighttime conditions.

The model is trained on bright-light synthetic data generated using the CARLA simulator and evaluated in nighttime environments to test its robustness under challenging lighting conditions.

---

##  Workflow
The project follows a structured pipeline:

1. Model Selection – Selected YOLOv5s for real-time detection  
2. Data Collection – Generated dataset using CARLA simulator  
3. Model Training – Applied data augmentation and trained the model  
4. Deployment – Tested performance in low-light simulation  


---

##  Technologies Used
- Python  
- PyTorch  
- YOLOv5  
- OpenCV  
- CARLA Simulator  
- TensorBoard (for monitoring metrics)

---

##  Dataset
- Collected using CARLA Simulator  
- Total images: ~2240   
- Classes:
  - Vehicles  
  - Pedestrians  
  - Trees  

### Data Preparation
- 80% training / 20% validation split  
- Synthetic data generated in bright conditions  
- Low-light simulated using:
  - Brightness reduction  
  - Gaussian blur  

---

##  Methodology

###  Model
- YOLOv5s (single-stage detector)
- Chosen for:
  - High speed and efficiency  
  - Real-time performance  
  - Low computational cost  


---

### Training Configuration
- Image size: 640 × 640  
- Epochs: 25  
- Framework: PyTorch  
- Metrics tracked:
  - Precision  
  - Recall  
  - mAP  
  - Loss curves   

---

######## Results ######
- mAP (0.5–0.95): ~85%  
- Precision: ~99%  
- Recall: ~95%  

### Per-class performance:
- Pedestrians → High accuracy (~90%)  
- Trees → Strong performance (~88%)  
- Vehicles → Lower performance (~77%) due to:
  - Fewer training samples  
  - Reduced visibility in low light  


---

## Deployment
- Tested in CARLA nighttime environment  
- RGB camera mounted on ego vehicle  

### Observations:
- Stable detection for pedestrians and trees  
- Vehicles detected with lower confidence in darker regions  
- Bounding boxes remained consistent across frames  

---

##  Key Learnings
- Low-light conditions significantly affect detection performance  
- Data augmentation improves generalization  
- Dataset imbalance impacts model accuracy  
- Simulation environments like CARLA are useful for controlled testing  

---

##  Future Improvements
- Balance dataset (especially vehicles)  
- Add more object classes (traffic signs, signals)  
- Use real-world low-light datasets  
- Try advanced models like YOLOv8  
- Integrate multi-sensor fusion for better perception  

---

##  Team Members
- Keerthana  
- Nithya  

---

##  Note
This repository contains a simplified version of a university research project. Confidential implementation details and full datasets are not included.
