import cv2
import os
import pandas as pd
from deepface import DeepFace

# Configuration
KNOWN_FACES_DIR = "known_faces"
MODEL_NAME = "VGG-Face" # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib
DISTANCE_METRIC = "cosine"

def run_recognition():
    print("Initializing DeepFace...")
    
    # Check if known_faces directory exists and is not empty
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created '{KNOWN_FACES_DIR}' folder.")
        print("Please put photos of people to recognize in this folder and restart.")
        return

    if len(os.listdir(KNOWN_FACES_DIR)) == 0:
        print(f"'{KNOWN_FACES_DIR}' is empty. Feature extraction might fail if no db is present.")
        print("Please add at least one image to 'known_faces' folder.")
        # Proceeding anyway, DeepFace might just return empty Pandas dataframe
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    print("Starting Webcam...")
    print("Press 'q' to quit.")
    
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break
            
        frame_count += 1
        
        # DeepFace.find is powerful but can be heavy. 
        # We will run it every 5 frames to keep the video smooth, 
        # or we just iterate continuously if the PC is fast enough.
        # For simplicity in this script, we'll try to run it on every frame but handle exceptions.
        # To make it faster, you typically detect face first, then run find.
        
        # However, DeepFace.stream() is a built-in function that does exactly this efficiently.
        # But to keep control of the window here, let's use a try-except block with find().
        
        try:
            # We save the current frame to a temporary file because some versions of DeepFace prefer paths
            # Or we pass the numpy array directly.
            
            # Run DeepFace.find
            # enforce_detection=False allows it to return usually empty list if no face found, instead of crashing
            dfs = DeepFace.find(
                img_path=frame, 
                db_path=KNOWN_FACES_DIR, 
                model_name=MODEL_NAME, 
                distance_metric=DISTANCE_METRIC,
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace.find returns a list of dataframes. 
            if len(dfs) > 0:
                for df in dfs:
                    if df.shape[0] > 0:
                        # We have matches!
                        # The dataframe contains columns like 'identity', 'source_x', 'source_y', 'source_w', 'source_h'
                        for index, row in df.iterrows():
                            x = int(row['source_x'])
                            y = int(row['source_y'])
                            w = int(row['source_w'])
                            h = int(row['source_h'])
                            
                            # Identity is the full path, e.g., "known_faces/Obama.jpg"
                            identity_path = row['identity']
                            user_name = os.path.basename(identity_path).split('.')[0]
                            
                            # Draw rectangle
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, user_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
        except Exception as e:
            # print(f"Error: {e}")
            pass

        cv2.imshow('Face Recognition (DeepFace)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
