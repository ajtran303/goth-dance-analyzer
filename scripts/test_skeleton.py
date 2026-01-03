"""
Test MediaPipe pose detection with live camera.
Use this to verify tracking works in your space before recording.
"""

import cv2
import numpy as np
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Model download URL and path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker.task"

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
]


def download_model():
    """Download the pose landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def draw_landmarks_on_image(frame, detection_result, mirror=False):
    """Draw pose landmarks on an image."""
    if not detection_result.pose_landmarks:
        return frame
    
    height, width = frame.shape[:2]
    
    for pose_landmarks in detection_result.pose_landmarks:
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            
            # Convert normalized coordinates to pixel coordinates
            if mirror:
                start_point = (int((1 - start.x) * width), int(start.y * height))
                end_point = (int((1 - end.x) * width), int(end.y * height))
            else:
                start_point = (int(start.x * width), int(start.y * height))
                end_point = (int(end.x * width), int(end.y * height))
            
            # Draw line
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in pose_landmarks:
            if mirror:
                x = int((1 - landmark.x) * width)
            else:
                x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    return frame


def test_skeleton_tracking():
    """Test MediaPipe pose detection with live camera."""
    
    # Download model if needed
    download_model()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "="*50)
    print("SKELETON TRACKING TEST")
    print("="*50)
    print(f"\nCamera: {width}x{height}")
    print("\nChecklist:")
    print("  [ ] Full body visible (head to feet)")
    print("  [ ] Skeleton tracks smoothly")
    print("  [ ] No major flickering")
    print("  [ ] Works when moving")
    print("\nPress Q to quit\n")
    
    # Create pose landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    frames_processed = 0
    frames_with_pose = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_processed += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect poses
        detection_result = detector.detect(mp_image)
        
        # Mirror frame first (so skeleton draws correctly mirrored)
        frame = cv2.flip(frame, 1)
        
        # Draw skeleton if detected
        if detection_result.pose_landmarks:
            frames_with_pose += 1
            
            # Draw landmarks (coordinates need to be flipped too)
            frame = draw_landmarks_on_image(frame, detection_result, mirror=True)
            
            # Get first pose landmarks
            landmarks = detection_result.pose_landmarks[0]
            
            # Wrist indices: left=15, right=16
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Display wrist coordinates
            cv2.putText(frame, f"L Wrist: ({left_wrist.x:.2f}, {left_wrist.y:.2f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"R Wrist: ({right_wrist.x:.2f}, {right_wrist.y:.2f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Visibility indicator
            avg_visibility = np.mean([lm.visibility for lm in landmarks])
            cv2.putText(frame, f"Visibility: {avg_visibility:.1%}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            status = "TRACKING"
            status_color = (0, 255, 0)
        else:
            status = "NO POSE DETECTED"
            status_color = (0, 0, 255)
        
        # Detection rate
        detection_rate = frames_with_pose / frames_processed if frames_processed > 0 else 0
        
        # Display status
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Detection rate: {detection_rate:.1%}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Skeleton Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Frames processed: {frames_processed}")
    print(f"Frames with pose: {frames_with_pose}")
    if frames_processed > 0:
        detection_rate = frames_with_pose / frames_processed
        print(f"Detection rate: {detection_rate:.1%}")
        
        if detection_rate > 0.9:
            print("\n✓ Excellent tracking! Ready to record.")
        elif detection_rate > 0.7:
            print("\n⚠ Good tracking, but consider improving lighting or camera angle.")
        else:
            print("\n✗ Poor tracking. Check:")
            print("  - Is full body visible?")
            print("  - Is lighting sufficient?")
            print("  - Is background too cluttered?")


if __name__ == "__main__":
    test_skeleton_tracking()
