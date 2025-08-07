import cv2
import os
import time

# --- Configuration ---
# Directory to save the videos
SAVE_PATH = "video_data" 

# Video properties
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20 # Frames per second

# --- Main Application ---

def record_video(label, video_index):
    """
    Records a single video clip and saves it with a structured name.
    
    Args:
        label (str): The label for the video (e.g., 'focused_na', 'unfocused_distracted').
        video_index (int): The index number for this video.
    """
    # Create the save directory if it doesn't exist
    label_path = os.path.join(SAVE_PATH, label.split('_')[0]) # Create 'focused' or 'unfocused' folders
    if not os.path.exists(label_path):
        os.makedirs(label_path)
        print(f"Created directory: {label_path}")

    # --- Set up the camera ---
    # 0 is usually the built-in webcam
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # --- Define the video writer ---
    # Use 'mp4v' codec for .mp4 files on macOS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    file_name = f"{label}_{int(time.time())}_{video_index}.mp4"
    # Save the file inside the respective main class folder
    full_path = os.path.join(label_path, file_name) 
    out = cv2.VideoWriter(full_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # Extract the main and sub-class for display
    display_label = label.replace('_', ' -> ').upper()

    print("\n-----------------------------------------")
    print(f"PREPARING TO RECORD: '{display_label}' - Video #{video_index}")
    print("Press 's' to start/stop recording.")
    print("Press 'q' to quit the current session.")
    print("-----------------------------------------")

    is_recording = False
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        # Flip the frame horizontally for a more natural, mirror-like view
        frame = cv2.flip(frame, 1)

        # Display recording status on the window
        if is_recording:
            # Draw a red circle to indicate recording
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1) 
            cv2.putText(frame, 'RECORDING', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame) # Write the frame to the file

        # Display the frame in a window
        cv2.imshow('Data Collection - Press "s" to record, "q" to quit', frame)

        # --- Keyboard Controls ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'): # 's' key to start/stop
            if is_recording:
                print(f"Stopped recording. Video saved to: {full_path}")
                break # Exit the loop to finish the function
            else:
                print(">>> Recording started...")
                is_recording = True
                
        elif key == ord('q'): # 'q' key to quit
            print("Quitting session.")
            # Release resources before breaking the loop
            if is_recording:
                out.release()
            cap.release()
            cv2.destroyAllWindows()
            exit() # Exit the entire script

    # --- Release all resources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # --- Data Collection Workflow ---
    while True:
        # --- MODIFIED MENU AND LOGIC ---
        print("\nWhat state are you recording?")
        print("1: Focused")
        print("\n--- Unfocused States ---")
        print("2: Procrastinating (e.g., watching a video, on social media)")
        print("3: Distracted (Looking at phone / away from screen)")
        print("4: Asleep")
        print("------------------------")
        print("Enter 'q' to exit the program.")
        
        choice = input("Enter your choice (1/2/3/4/q): ").strip().lower()

        if choice == '1':
            # Main class: focused, Sub-class: not applicable
            label_name = 'focused_na'
        elif choice == '2':
            # Main class: unfocused, Sub-class: procrastinating
            label_name = 'unfocused_procrastinating'
        elif choice == '3':
            # Main class: unfocused, Sub-class: distracted
            label_name = 'unfocused_distracted'
        elif choice == '4':
            # Main class: unfocused, Sub-class: asleep
            label_name = 'unfocused_asleep'
        elif choice == 'q':
            print("Exiting data collection.")
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        # --- END OF MODIFICATIONS ---
            
        num_videos = int(input(f"How many videos for '{label_name.replace('_', ' -> ')}' do you want to record in this session? "))
        
        for i in range(1, num_videos + 1):
            record_video(label_name, i)
            # Give a moment to reset between recordings
            time.sleep(1)