import cv2
import dlib
from tensorflow.keras.models import model_from_json
import numpy as np

# Constants
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
EMOTION_FRAME_THRESHOLD = 2

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load DNN face detector
modelFile = "./res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load emotion detection model
with open('./emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('./emotion_model.weights.h5')

# Initialize video stream
video_stream = cv2.VideoCapture(0)

# Trackers and face emotion storage
face_emotions = {}
trackers = {} 

while True:
    grabbed, frame = video_stream.read()
    if not grabbed:
        break
    
    frame = cv2.resize(frame, (1280, 720))
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    if not trackers:
        trackers.clear()
        face_emotions.clear()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, x2 - x, y2 - y))
                tracker_id = id(tracker)
                trackers[tracker_id] = tracker
                face_emotions[tracker_id] = {'emotions': [], 'display_emotion': ""}
                
    active_trackers = {}
    for tracker_id, tracker in trackers.items():
        success, bbox = tracker.update(frame)
        if not success:
            continue
        
        x, y, w, h = [int(v) for v in bbox]
        x2, y2 = x + w, y + h
        
        if x <= 0 or y <= 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or w < 30 or h < 30:
            continue  
        
        active_trackers[tracker_id] = tracker
        
        face_resized = cv2.resize(frame[y:y2, x:x2], (48, 48))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_resized = face_resized.astype("float32") / 255.0
        face_expanded = np.expand_dims(face_resized, axis=0)
        face_expanded = np.expand_dims(face_expanded, axis=-1)  
        
        predictions = emotion_model.predict(face_expanded)
        max_index = np.argmax(predictions[0])
        current_emotion = emotion_dict[max_index]
        
        face_emotions[tracker_id]['emotions'].append(current_emotion)
        if len(face_emotions[tracker_id]['emotions']) > EMOTION_FRAME_THRESHOLD:
            if all(e == face_emotions[tracker_id]['emotions'][-1] for e in face_emotions[tracker_id]['emotions'][-EMOTION_FRAME_THRESHOLD:]):
                face_emotions[tracker_id]['display_emotion'] = current_emotion
            face_emotions[tracker_id]['emotions'].pop(0)
            
        display_emotion = face_emotions[tracker_id]['display_emotion']
        
        if display_emotion:
            cv2.rectangle(frame, (x, y), (x2, y2), BOX_COLOR, 2)
            cv2.putText(frame, display_emotion, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2) 
            
    trackers = active_trackers
    
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()
