# Collision Tracker: Real-Time Traffic Accident Detection Using Computer Vision and Temporal Modeling

## Overview

Collision Tracker is a computer vision pipeline designed to detect vehicle collisions and anomalous driving behavior from traffic video footage. The system combines object detection, multi-object tracking, optical flow estimation, and temporal anomaly detection to identify potential accidents and dangerous traffic events.

Unlike traditional frame-by-frame classification approaches, this project models vehicle interactions over time by tracking object motion, estimating scene dynamics, and learning normal traffic behavior. Sudden deviations from learned patterns are flagged as potential collision events.

---

## Motivation

Traffic accident detection is an important problem for intelligent transportation systems, autonomous driving, and roadway safety monitoring. Many existing approaches rely on manual monitoring or frame-level classifiers that struggle to capture complex interactions between vehicles.

This project explores whether accident detection can be improved by combining:

- Object-level vehicle tracking
- Motion estimation through optical flow
- Interaction-based features between vehicles
- Temporal anomaly detection

The goal is to create a system that understands how vehicles move and interact over time rather than treating each frame independently.

---

## Pipeline Architecture

```text
Video Frames
     │
     ▼
YOLO Vehicle Detection
     │
     ▼
DeepSORT / ByteTrack Tracking
     │
     ▼
RAFT Optical Flow Estimation
     │
     ▼
Feature Engineering
     │
     ▼
Sequence Construction
     │
     ▼
LSTM Autoencoder
     │
     ▼
Accident / Anomaly Detection
```

### 1. Vehicle Detection and Tracking

Vehicles are detected using YOLO and assigned persistent identities using DeepSORT/ByteTrack-based tracking.

This stage provides:

- Vehicle bounding boxes
- Unique object IDs
- Consistent tracking across frames

Tracking enables the system to analyze vehicle trajectories and behavior over time.

### 2. Motion Estimation with RAFT

The RAFT optical flow model is used to estimate pixel-level motion between consecutive frames.

For each tracked vehicle, flow information is aggregated within its bounding box to generate motion descriptors such as:

- Average horizontal flow
- Average vertical flow
- Mean flow magnitude
- Flow variance

These features capture vehicle dynamics that may not be apparent from bounding box coordinates alone.

### 3. Feature Engineering

For every tracked vehicle, spatial, kinematic, and interaction-based features are extracted.

#### Spatial Features

- Normalized x position
- Normalized y position
- Bounding box width
- Bounding box height

#### Motion Features

- Velocity
- Acceleration
- Speed
- Optical flow statistics

#### Interaction Features

- Time-to-Collision (TTC)
- Relative velocity
- Relative angle
- Closing speed
- Intersection-over-Union (IoU) dynamics

These interaction features allow the system to reason about potential collision risk between nearby vehicles.

### 4. Sequence Construction

Vehicle features are organized into fixed-length temporal sequences.

Each sequence captures:

- Multiple tracked vehicles
- Vehicle state evolution over time
- Interaction patterns between vehicles

This transforms raw detections into structured spatio-temporal representations suitable for sequence learning.

### 5. Anomaly Detection

An LSTM Autoencoder is trained on normal driving behavior.

The model learns to reconstruct typical traffic patterns. When a collision or dangerous event occurs, reconstruction error increases significantly, signaling an anomaly.

Advantages:

- Does not require large amounts of labeled accident data
- Learns normal behavior directly from video
- Can generalize to unseen accident scenarios

---

## Feature Representation

The system combines several categories of features:

| Category | Features |
|-----------|------------|
| Spatial | x, y, width, height |
| Kinematic | velocity, acceleration, speed |
| Optical Flow | mean flow x, mean flow y, flow magnitude, flow variance |
| Interaction | TTC, relative angle, closing speed, IoU dynamics |

These features provide both object-level and interaction-level understanding of traffic behavior.

---

## Dataset

Frames are extracted from traffic simulation videos and processed sequentially through the pipeline.

Example frame naming convention:

```text
Town03_head-on_clear_22_frame_0000.jpg
```

Generated features are stored in HDF5 format for efficient training and experimentation.

Example feature tensor:

```text
(num_sequences, sequence_length, max_objects, feature_dimension)
```

Example:

```text
(6009, 16, 50, 19)
```

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- HDF5
- YOLO
- DeepSORT / ByteTrack
- RAFT Optical Flow
- LSTM Autoencoders

---

## Key Challenges

### Delayed Accident Detection

A major challenge was that anomaly scores often peaked after vehicles had already collided and stopped moving.

### Multi-Vehicle Scene Representation

Traffic scenes contain varying numbers of vehicles. The system uses fixed-size sequence representations and object buffering strategies to maintain consistent input dimensions for temporal models.

---

## Results

The system successfully:

- Detects and tracks vehicles across traffic scenes
- Extracts motion-aware features using optical flow
- Models vehicle interactions over time
- Identifies anomalous driving behavior associated with collisions

This project demonstrates how combining object tracking, motion estimation, and temporal anomaly detection can improve accident recognition compared to frame-level approaches.

---

## Future Improvements

Potential directions for future work include:

- Graph Neural Networks for vehicle interaction modeling
- Transformer-based temporal architectures
- Collision anticipation before impact
- Real-time deployment optimization
- Multi-camera traffic monitoring
- Integration with autonomous driving safety systems

---

## Repository Structure

```text
Collision-tracker/
│
├── detector_tracker/      # Vehicle detection and tracking
├── optical_flow/          # RAFT motion estimation
├── feature_builder/       # Feature engineering pipeline
├── sequence_builder/      # Temporal sequence generation
├── models/                # LSTM autoencoder
├── data/                  # Processed datasets
├── outputs/               # Predictions and visualizations
└── notebooks/             # Experiments and analysis
```

## Author

**Kanish Reddy Vuyyuru [vuyyu011@umn.edu]**

**Rajit Bhargav Mahesh [Mahes110@purdue.edu]**
