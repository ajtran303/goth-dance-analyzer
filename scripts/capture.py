# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false

"""
Dance session capture script.
Records video with proper naming for later skeleton extraction.
"""

import cv2
import os


def get_recording_filename(dancer_name, song_name, output_dir="recordings"):
    """Generate a filename for the recording."""
    # Create output directory if it doesn't exist
    dancer_dir = os.path.join(output_dir, dancer_name.lower().replace(" ", "_"))
    os.makedirs(dancer_dir, exist_ok=True)
    
    # Create filename
    song_clean = song_name.lower().replace(" ", "_").replace("'", "")
    filename = f"{dancer_name.lower().replace(' ', '_')}_{song_clean}.mp4"
    
    return os.path.join(dancer_dir, filename)


def record_dance(dancer_name, song_name, output_dir="recordings"):
    """Record a dance session."""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Try to set resolution (MacBook may override)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"Camera: {width}x{height} @ {fps}fps")
    
    # Prepare video writer (but don't start yet)
    filepath = get_recording_filename(dancer_name, song_name, output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    recording = False
    frame_count = 0
    
    print("\n" + "="*50)
    print(f"Dancer: {dancer_name}")
    print(f"Song: {song_name}")
    print("="*50)
    print("\nControls:")
    print("  [SPACE] - Start/Stop recording")
    print("  [Q]     - Quit without saving")
    print("  [S]     - Save and exit")
    print("\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            break
        
        # Create display frame (flip for mirror effect)
        display = cv2.flip(frame, 1)
        
        # Add status overlay
        if recording:
            status = f"RECORDING - {frame_count} frames"
            color = (0, 0, 255)  # Red
            # Add recording indicator
            cv2.circle(display, (30, 30), 15, (0, 0, 255), -1)
        else:
            status = "READY - Press SPACE to start"
            color = (0, 255, 0)  # Green
        
        # Add text overlay
        cv2.putText(display, status, (60, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display, f"{dancer_name} - {song_name}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Dance Capture', display)
        
        # Write frame if recording (use non-flipped frame)
        if recording:
            out.write(frame)
            frame_count += 1
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - toggle recording
            if not recording:
                # Start recording
                out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                recording = True
                frame_count = 0
                print(f"Recording started: {filepath}")
            else:
                # Stop recording
                recording = False
                print(f"Recording paused at {frame_count} frames")
        
        elif key == ord('s'):  # Save and exit
            if out is not None:
                out.release()
                print(f"\nSaved: {filepath}")
                print(f"Total frames: {frame_count}")
                print(f"Duration: ~{frame_count/fps:.1f} seconds")
            break
        
        elif key == ord('q'):  # Quit without saving
            if out is not None:
                out.release()
                # Delete the file if it exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print("Recording discarded")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return filepath if frame_count > 0 else None


def main():
    """Main entry point with interactive prompts."""
    print("\n" + "="*50)
    print("GOTH DANCE CAPTURE")
    print("="*50 + "\n")
    
    # Get dancer name
    dancer_name = input("Dancer name: ").strip()
    if not dancer_name:
        print("Error: Dancer name required")
        return
    
    # Song list
    songs = [
        "Bela Lugosis Dead",
        "Lucretia My Reflection", 
        "Cities in Dust",
        "Gallowdance",
        "Love Will Tear Us Apart"
    ]
    
    print("\nSongs:")
    for i, song in enumerate(songs, 1):
        print(f"  {i}. {song}")
    print(f"  {len(songs)+1}. Custom")
    
    choice = input("\nSelect song number: ").strip()
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(songs):
            song_name = songs[choice_num - 1]
        elif choice_num == len(songs) + 1:
            song_name = input("Enter song name: ").strip()
        else:
            print("Invalid choice")
            return
    except ValueError:
        print("Invalid input")
        return
    
    print(f"\nReady to record {dancer_name} dancing to {song_name}")
    input("Press Enter to open camera...")
    
    record_dance(dancer_name, song_name)


if __name__ == "__main__":
    main()
