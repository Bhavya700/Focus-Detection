import cv2
import dlib
import numpy as np
import os
from imutils import face_utils
from feature_extractor_utils import calculate_ear, calculate_mar, get_head_pose

# --- Configuration and Constants ---
VIDEO_DATA_PATH = "video_data"
DLIB_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
OUTPUT_FEATURES_FILE = "features.npy"
OUTPUT_LABELS_FILE = "labels.npy"

# --- Main Processing Logic ---

def extract_features():
    """
    Loops through all recorded videos, extracts features from each frame,
    and saves the features and labels to .npy files.
    """
    print("[INFO] Initializing Dlib's face detector and landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_LANDMARK_PREDICTOR)
    
    # These lists will hold all our data
    all_features = []
    all_labels = []

    print("[INFO] Starting feature extraction from videos...")
    # Iterate through main class folders ('focused', 'unfocused')
    for main_class_label in os.listdir(VIDEO_DATA_PATH):
        main_class_path = os.path.join(VIDEO_DATA_PATH, main_class_label)
        if not os.path.isdir(main_class_path):
            continue

        # Iterate through the videos in each folder
        for video_filename in os.listdir(main_class_path):
            if not video_filename.endswith('.mp4'):
                continue
            
            # Parse the filename to get the labels
            parts = video_filename.split('_')
            # main_class is parts[0], sub_class is parts[1]
            label = [parts[0], parts[1]] 
            
            video_path = os.path.join(main_class_path, video_filename)
            print(f"[INFO] Processing video: {video_path} with label: {label}")

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # The landmark predictor works best on grayscale images
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image_dims = frame.shape[:2] # (height, width)
                
                # Detect faces in the grayscale frame
                rects = detector(gray, 0)
                
                # We are assuming one person is in the frame
                if len(rects) > 0:
                    # Get the 68 facial landmarks
                    shape = predictor(gray, rects[0])
                    shape = face_utils.shape_to_np(shape)

                    # --- Feature Calculation ---
                    
                    # 1. Eye Aspect Ratio (EAR)
                    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                    left_eye = shape[lStart:lEnd]
                    right_eye = shape[rStart:rEnd]
                    ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

                    # 2. Mouth Aspect Ratio (MAR)
                    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                    mouth = shape[mStart:mEnd]
                    mar = calculate_mar(mouth)

                    # 3. Head Pose Estimation
                    pitch, yaw, roll = get_head_pose(shape, image_dims)

                    # --- Assemble Feature Vector ---
                    feature_vector = [ear, mar, pitch, yaw, roll]
                    
                    all_features.append(feature_vector)
                    all_labels.append(label)
                    
                    frame_count += 1
            
            cap.release()
            print(f"[INFO] Extracted features from {frame_count} frames.")

    # Convert lists to NumPy arrays for efficient saving
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n[SUCCESS] Feature extraction complete.")
    print(f"Shape of features array (X): {X.shape}") # Should be (total_frames, 5)
    print(f"Shape of labels array (y): {y.shape}")   # Should be (total_frames, 2)

    # Save the arrays to disk
    print(f"[INFO] Saving features to {OUTPUT_FEATURES_FILE}")
    np.save(OUTPUT_FEATURES_FILE, X)
    
    print(f"[INFO] Saving labels to {OUTPUT_LABELS_FILE}")
    np.save(OUTPUT_LABELS_FILE, y)


if __name__ == '__main__':
    extract_features()