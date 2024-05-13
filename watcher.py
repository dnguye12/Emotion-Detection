import cv2
import dlib
from tensorflow.keras.models import model_from_json
import numpy as np

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
EMOTION_FRAME_THRESHOLD = 2

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

video_stream = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

json_file = open('./emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('./emotion_model.weights.h5')

last_predicted_time = 0
last_predicted_emotion = ""

face_emotions = {}
trackers = {} 

while True:
    grabbed, frame = video_stream.read()
    
    if grabbed:
        frame = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not trackers:
            trackers.clear()
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                if w < 30 and h<30:
                    continue
                
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x+w, y+h)
                tracker.start_track(frame, rect)
                tracker_id = id(tracker)
                trackers[tracker_id] = tracker
                face_emotions[tracker_id] = {'emotions': [], 'display_emotion': ""}
        active_trackers = {}
        for tracker_id, tracker in trackers.items():
            quality = tracker.update(frame)
            pos = tracker.get_position()
            x, y, x2, y2 = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
            
            if quality < 7 or x <= 0 or y <= 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or (x2 - x) < 30 or (y2 - y) < 30:
                continue  
            
            active_trackers[tracker_id] = tracker
            
            face_resized = cv2.resize(gray[y:y2, x:x2], (48, 48))
            face_resized = face_resized.astype("float32") / 255.0
            face_expanded = np.expand_dims(face_resized, axis=0)
            face_expanded = np.expand_dims(face_expanded, axis=-1)  
            
            predictions = emotion_model.predict(face_expanded)
            max_index = np.argmax(predictions[0])
            current_emotion = list(emotion_dict.values())[max_index]
            
            face_emotions[tracker_id]['emotions'].append(current_emotion)
            if len(face_emotions[tracker_id]['emotions']) > EMOTION_FRAME_THRESHOLD:
                if all(e == face_emotions[tracker_id]['emotions'][-1] for e in face_emotions[tracker_id]['emotions'][-EMOTION_FRAME_THRESHOLD:]):
                    face_emotions[tracker_id]['display_emotion'] = current_emotion
                face_emotions[tracker_id]['emotions'].pop(0)
                
            display_emotion = face_emotions[tracker_id]['display_emotion']
            
            if display_emotion is not None:
                cv2.rectangle(frame, (x, y), (x2, y2), BOX_COLOR, 2)
                cv2.putText(frame, display_emotion, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2) 
        trackers = active_trackers
        
        
         
        cv2.imshow('Emotion Detection', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_stream.release()
cv2.destroyAllWindows()     