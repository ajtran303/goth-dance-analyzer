"""
Extract skeleton data from recorded dance videos.
Outputs JSON files with landmark positions for each frame.
"""

import cv2
import json
import os
import sys
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Model download URL and path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker.task"


def download_model():
    """Download the pose landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def extract_skeleton(video_path, output_path=None, output_dir="skeleton_data"):
    """
    Extract skeleton data from a video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output JSON (default: skeleton_data/<name>_skeleton.json)
        output_dir: Base directory for output files
    
    Returns:
        Path to output JSON file
    """
    
    # Download model if needed
    download_model()
    
    # Set default output path in skeleton_data folder
    if output_path is None:
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        # Preserve subdirectory structure (e.g., recordings/alice/ -> skeleton_data/alice/)
        rel_dir = os.path.dirname(video_path)
        if rel_dir.startswith("recordings"):
            rel_dir = rel_dir.replace("recordings", "", 1).lstrip(os.sep)
        output_subdir = os.path.join(output_dir, rel_dir) if rel_dir else output_dir
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, f"{video_name}_skeleton.json")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nProcessing: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.1f} seconds\n")
    
    # Create pose landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Process frames
    skeleton_data = {
        'metadata': {
            'source_video': video_path,
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps
        },
        'frames': []
    }
    
    frame_count = 0
    frames_with_pose = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect poses
        detection_result = detector.detect(mp_image)
        
        # Extract landmark data
        frame_data = {
            'frame': frame_count,
            'timestamp': frame_count / fps,
            'landmarks': None
        }
        
        if detection_result.pose_landmarks:
            frames_with_pose += 1
            landmarks = detection_result.pose_landmarks[0]
            
            frame_data['landmarks'] = [
                {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
                for lm in landmarks
            ]
        
        skeleton_data['frames'].append(frame_data)
        frame_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
    
    cap.release()
    
    # Add summary stats
    skeleton_data['metadata']['frames_with_pose'] = frames_with_pose
    skeleton_data['metadata']['detection_rate'] = frames_with_pose / frame_count if frame_count > 0 else 0
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(skeleton_data, f)
    
    print(f"\nComplete!")
    print(f"Frames processed: {frame_count}")
    print(f"Frames with pose: {frames_with_pose}")
    print(f"Detection rate: {100*frames_with_pose/frame_count:.1f}%")
    print(f"Saved to: {output_path}")
    
    return output_path


def batch_extract(input_dir="recordings", output_dir="skeleton_data"):
    """
    Extract skeleton data from all videos in a directory.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory for output JSON files
    """
    
    video_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files\n")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(video_files)}")
        print(f"{'='*50}")
        
        extract_skeleton(video_path, output_dir=output_dir)
    
    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"{'='*50}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python extract_skeleton.py <video_path>           # Single video")
        print("  python extract_skeleton.py --batch [input_dir]    # All videos in directory")
        print("\nExamples:")
        print("  python extract_skeleton.py recordings/alice/alice_gallowdance.mp4")
        print("  python extract_skeleton.py --batch recordings")
        return
    
    if sys.argv[1] == '--batch':
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "recordings"
        batch_extract(input_dir)
    else:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            return
        extract_skeleton(video_path)


if __name__ == "__main__":
    main()
