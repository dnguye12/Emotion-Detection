import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import socket
from deepface import DeepFace

#Set up the socket with an indicated connection ip and port
#client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client.connect(('127.0.0.1',26950))

# Constants
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
EMOTION_FRAME_THRESHOLD = 2

# Initialize video stream
video_stream = cv2.VideoCapture(0)

# Trackers and face emotion storage
face_emotions = {}
trackers = {} 

while True:
    grabbed, frame = video_stream.read()
    
    #If no camera is present, stop the program.
    if not grabbed:
        break
    
    frame = cv2.resize(frame, (1280, 720)) # Resize frame for consistency
    
    try:
        results = DeepFace.analyze(frame, actions=['emotion'])
    except:
        continue
    
    active_trackers = {}
    
    for result in results:
        x,y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
        x2, y2 = x + w, y + h
        
        # Ignore out of bounds or too small boxes
        if x <= 0 or y <= 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or w < 30 or h < 30:
            continue

        tracker_id = id(result)
        active_trackers[tracker_id] = result
        
        current_emotion = result['dominant_emotion']

        if tracker_id not in face_emotions:
            face_emotions[tracker_id] = {'emotions': [], 'display_emotion': ""}

        face_emotions[tracker_id]['emotions'].append(current_emotion)
        if len(face_emotions[tracker_id]['emotions']) > EMOTION_FRAME_THRESHOLD:
            # Confirm emotion if it remains consistent over a threshold of frames
            if all(e == face_emotions[tracker_id]['emotions'][-1] for e in face_emotions[tracker_id]['emotions'][-EMOTION_FRAME_THRESHOLD:]):
                face_emotions[tracker_id]['display_emotion'] = current_emotion
                message = f"{tracker_id}-{current_emotion}".encode('utf-8')
                # client.send(message)
            face_emotions[tracker_id]['emotions'].pop(0)

        display_emotion = face_emotions[tracker_id]['display_emotion']

        if display_emotion:
            cv2.rectangle(frame, (x, y), (x2, y2), BOX_COLOR, 2)
            cv2.putText(frame, display_emotion, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
      
    trackers = active_trackers # Update trackers
    
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()
#client.close()