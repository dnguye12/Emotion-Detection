import cv2
import torch
from tensorflow.keras.models import model_from_json
from multiprocessing import Process, Queue
from ultralytics import YOLO
import numpy as np
import time
import gaze
import mediapipe as mp
import socket

#Set up socket with an indicated connection ip and port
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1',26950))

# Ensure PyTorch uses GPU if available
device = 0 if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(device)

# Constants
VIDEO_INPUT = 0

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
GAZE_COLOR = (0, 0, 255)

EMOTION_FRAME_THRESHOLD = 3
LAST_DETECTED_THRESHOLD = 1

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the YOLO model with the specified device
face_detector = YOLO('./yolov8m-face.pt').to(device)

# Load emotion detection model
with open('./emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('./emotion_model.weights.h5')

# Load Mediapipe for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    static_image_mode=False, 
    refine_landmarks=True,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

#Function to read each frame of the video/webcam
def capture_frames(video_path, frame_queue):
    # Open the video file or webcam
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file or webcam opened successfully
    if not cap.isOpened():
        print("Error reading video file")
        return
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        # If the frame is not read successfully, exit the loop
        if not ret:
            break
        
        # If the frame queue is empty, add the frame to the queue
        if frame_queue.empty():
            frame_queue.put(frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Add a None value to the queue to signal the end of the video
    frame_queue.put(None)
    # Release the video capture object
    cap.release()
    
# Function to detect emotions from video frames
def emotion_detection(emotion_frame_queue, emotion_faces_queue, emotion_output_queue):
    while True:
        # Get a frame from the emotion frame queue
        frame = emotion_frame_queue.get()
        
        # If the frame is None, break the loop
        if frame is None:
            break
        
        # Get the detected faces from the emotion faces queue
        faces = emotion_faces_queue.get()
        updated_faces = {}
        
        # Process each detected face for emotion detection
        for tracker_id, tracker in faces.items():
            x,y,w,h  = tracker['box']
            x2, y2 = x + w, y + h
            
            # Prepare the face ROI for emotion detection
            face_resized = cv2.resize(frame[y:y2, x:x2], (48, 48))
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_resized = face_resized.astype("float32") / 255.0
            face_expanded = np.expand_dims(face_resized, axis=0)
            face_expanded = np.expand_dims(face_expanded, axis=-1)  
            
            # Predict the emotion of the face
            predictions = emotion_model.predict(face_expanded)
            max_index = np.argmax(predictions[0])
            current_emotion = emotion_dict[max_index]
            
            # Update the tracker's emotion data
            old_display_emotion = tracker['display_emotion']
            old_emotion = tracker['emotion']
            old_counter = tracker['counter']
            old_changed = tracker['changed']
            old_looking_camera = tracker["looking_camera"]
            
            if old_display_emotion == '':
                # Initialize emotion data if not set
                old_display_emotion = current_emotion
                old_emotion = current_emotion
                old_counter = 0
                old_changed = True
            else:
                # Update emotion counter and change display emotion if needed
                if current_emotion == old_emotion:
                    old_counter += 1
                else:
                    old_emotion = current_emotion
                    old_counter = 0
                
                if old_counter > EMOTION_FRAME_THRESHOLD:
                    if old_emotion != old_display_emotion:
                        old_display_emotion = old_emotion
                        old_changed = True
            
            # Prepare the updated tracker data
            current_tracker = {
                "box": tracker['box'],
                "emotion": old_emotion,
                "display_emotion": old_display_emotion,
                "counter": old_counter,
                "last_detected": tracker['last_detected'],
                "gaze": tracker['gaze'],
                "last_gaze": tracker['last_gaze'],
                "changed": old_changed,
                "looking_camera": old_looking_camera
            }
            updated_faces[tracker_id] = current_tracker
        # Put the updated faces data into the emotion output queue
        emotion_output_queue.put(updated_faces)
        
# Function to track gaze in video frames 
def gaze_tracking(gaze_frame_queue, gaze_faces_queue, gaze_output_queue):
    while True:
        # Get a frame from the gaze frame queue
        frame = gaze_frame_queue.get()
        
        # If the frame is None, break the loop
        if frame is None:
            break           
        
        # Convert the frame from BGR to RGB color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the detected faces from the gaze faces queue
        faces = gaze_faces_queue.get()
        
        updated_faces = {}
        
        # Process each detected face for gaze tracking
        for tracker_id, tracker in faces.items():
            x,y,w,h  = tracker['box']
            x2, y2 = x + w, y + h
            
            # Process the frame to detect face landmarks for gaze tracking
            gazeResults = face_mesh.process(frame)
            current_gaze = None
            
            if gazeResults.multi_face_landmarks:
                for face_landmarks in gazeResults.multi_face_landmarks:
                    # Get the current gaze direction
                    current_gaze = gaze.gaze(frame, face_landmarks)
                    
                    if current_gaze is None:
                        continue
                    
                    # Extract gaze points
                    #is_looking_at_camera = False
                    (p1_left, p2_left, p1_right, p2_right, is_looking_at_camera) = current_gaze
                    current_gaze = (p1_left, p2_left, p1_right, p2_right)
                    
                    # Check if the gaze points are within the face bounding box
                    if p1_left[0] >= x and p1_left[0] <= x2 and p1_left[1] >= y and p1_left[1] <= y2:
                        # Prepare the updated tracker data
                        current_tracker = {
                            "box": tracker['box'],
                            "emotion": tracker['emotion'],
                            "display_emotion": tracker['display_emotion'],
                            "counter": tracker['counter'],
                            "last_detected": tracker['last_detected'],
                            "gaze": current_gaze,
                            "last_gaze": time.time(),
                            "changed": True,
                            "looking_camera": is_looking_at_camera
                        }
                        updated_faces[tracker_id] = current_tracker
        # Put the updated faces data into the gaze output queue
        gaze_output_queue.put(updated_faces)

def main():
    # Initialize queues for inter-process communication
    frame_queue = Queue()
    emotion_frame_queue = Queue()
    emotion_faces_queue = Queue()
    emotion_output_queue = Queue()
    gaze_frame_queue = Queue()
    gaze_faces_queue = Queue()
    gaze_output_queue = Queue()
    
    # Start processes for capturing frames, detecting emotions, and tracking gaze
    frame_process = Process(target=capture_frames, args=(VIDEO_INPUT, frame_queue))
    emotion_process = Process(target=emotion_detection, args=(emotion_frame_queue, emotion_faces_queue, emotion_output_queue))
    gaze_process = Process(target=gaze_tracking, args=(gaze_frame_queue, gaze_faces_queue, gaze_output_queue))
    
    frame_process.start()
    emotion_process.start()
    gaze_process.start()
    
    # Dictionary to keep track of detected faces and their attributes
    drawing_list = {}
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            if frame is None:
                break
            
            # Preprocess the frame for face detection
            (h, w) = frame.shape[:2]
            detections = face_detector.predict(frame, conf=0.7)
            
            for face in detections:
                parameters = face.boxes
                for box in parameters:
                    startX, startY, endX, endY = box.xyxy[0]
                    startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
                    w2 = endX - startX
                    h2 = endY - startY
                    
                    if w2 < 100 or h2 < 100 or startX < 0 or startY < 0 or endX > w or endY > h:
                        continue
                    
                    bbox = (startX, startY, w2, h2)

                    # Check if the new detection overlaps with any existing tracker
                    overlap = False
                    for tracker_id, tracker in drawing_list.items():
                        (tx, ty, tw, th) = tracker['box']
                        if (startX < tx + tw and startX + (w2) > tx and
                            startY < ty + th and startY + (h2) > ty):
                            overlap = True
                            tracker['box'] = bbox
                            tracker['last_detected'] = time.time()
                            break

                    # If no overlap, create a new tracker
                    if not overlap:
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, bbox)
                        tracker_id = id(tracker)
                        drawing_list[tracker_id] = {
                            "box": bbox,
                            "emotion": "",
                            "display_emotion": "",
                            "counter": 0,
                            "last_detected": time.time(),
                            "gaze": None,
                            "last_gaze": time.time(),
                            "changed": True,
                            "looking_camera": False
                        }
                        
            # Remove old trackers that no longer exist in the screen
            updated_trackers = {}
            helper_time = time.time()
            for tracker_id, tracker in drawing_list.items():
                if helper_time - tracker['last_detected'] < LAST_DETECTED_THRESHOLD:
                    updated_trackers[tracker_id] = tracker
            drawing_list.clear()
            drawing_list = updated_trackers
            
            # Pass frame and face data to emotion detection process
            emotion_frame_queue.put(frame)
            emotion_faces_queue.put(drawing_list)
            
            # Retrieve and update emotion data
            if not emotion_output_queue.empty():
                emotions = emotion_output_queue.get()
                
                for tracker_id, tracker in emotions.items():
                    if tracker_id in drawing_list:
                        drawing_list[tracker_id]['emotion'] = tracker['emotion']
                        drawing_list[tracker_id]['display_emotion'] = tracker['display_emotion']
                        drawing_list[tracker_id]['counter'] = tracker['counter']
                        
            # Pass frame and face data to gaze tracking process
            gaze_frame_queue.put(frame)
            gaze_faces_queue.put(drawing_list)
                  
            # Retrieve and update gaze data
            if not gaze_output_queue.empty():
                gazes = gaze_output_queue.get()
                
                for tracker_id, tracker in gazes.items():
                    if tracker_id in drawing_list:
                        drawing_list[tracker_id]['gaze'] = tracker['gaze']  
                        drawing_list[tracker_id]['last_gaze'] = helper_time
                        drawing_list[tracker_id]['looking_camera'] = tracker['looking_camera']
            
            # Draw bounding boxes, emotions, and gaze lines on the frame
            for tracker_id, tracker in drawing_list.items():
                (x, y, w, h) = tracker['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
                display_emotion = tracker['display_emotion']
                cv2.putText(frame, display_emotion, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2) 
        
                if helper_time - tracker['last_gaze'] <= LAST_DETECTED_THRESHOLD * 2:
                    gaze = tracker['gaze']
                    if not gaze is None:
                        (p1_left, p2_left, p1_right, p2_right) = gaze
                        cv2.line(frame, p1_left, p2_left, GAZE_COLOR, 2)
                        cv2.line(frame, p1_right, p2_right, GAZE_COLOR, 2)
                else:
                    drawing_list[tracker_id]['gaze'] = None
                    
                if tracker['changed']:
                    message = f"{tracker_id}-{tracker['box']}-{tracker['display_emotion']}-{tracker['gaze']}".encode('utf-8')
                    client.send(message)


                    
            # Display the frame
            cv2.imshow('Real-time Face Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # Terminate and join processes
    frame_process.terminate()
    frame_process.join()
    emotion_process.terminate()
    emotion_process.join()
    gaze_process.terminate()
    gaze_process.join()
    cv2.destroyAllWindows()
    client.close()
    
if __name__ == "__main__":
    main()
