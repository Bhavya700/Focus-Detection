# Focus-Detection
A real-time system to monitor user focus during work or study sessions using your device's camera. Inspired by driver drowsiness detection systems, FocusGuard identifies whether a user is focused, distracted, procrastinating, or has fallen asleep, providing a foundation for building smart productivity tools.

## Overview
In the age of digital distractions, maintaining focus is a significant challenge. This project aims to create a personal productivity assistant that uses computer vision to understand a user's state. By analyzing the video feed from a webcam, the system performs a hierarchical classification:
Primary State: Is the user focused or unfocused?
Secondary State (if unfocused): What is the reason for the lack of focus?
procrastinating: Looking at the screen but on non-work content.
distracted: Looking away from the screen (e.g., at a phone).
asleep: Eyes closed for an extended period, head nodding.

## How It Works (The Pipeline)
The project follows a classic machine learning pipeline:
- Data Collection: A Python script (collect_data.py) uses OpenCV to capture and label short video clips for each user state (e.g., unfocused_asleep). This creates a custom dataset tailored to the user.
- Feature Extraction: The core processing script (process_videos.py) analyzes each video frame-by-frame. It uses Dlib to detect 68 facial landmarks and calculates a set of key features:
  - Eye Aspect Ratio (EAR): To detect blinks and eye closure.
  - Mouth Aspect Ratio (MAR): To detect yawns.
  - Head Pose Estimation: To determine the head's pitch, yaw, and roll, which helps identify when a user is looking down or away.
- Model Training (Upcoming): The extracted numerical features (features.npy) and their corresponding labels (labels.npy) will be used to train two machine learning models:
  - A binary classifier for focused vs. unfocused.
  - A multi-class classifier to determine the sub-state of unfocused.
- Real-time Inference (Upcoming): The trained models will be deployed in a real-time script that analyzes the live webcam feed and outputs the user's current focus state.

## Features
- Hierarchical State Detection: Classifies focus state on two levels for more nuanced understanding.
- Customizable Data Collection: Easily create your own dataset for personalized accuracy.
- Robust Feature Engineering: Uses well-established computer vision metrics (EAR, MAR, Head Pose) for reliable state detection.
- Local & Private: All processing is done locally on your machine, ensuring your data and video feed remain private.

## Requirements
- Python 3.7+
- OpenCV
- Dlib
- NumPy
- Scikit-learn
- TensorFlow/Keras (for model training)