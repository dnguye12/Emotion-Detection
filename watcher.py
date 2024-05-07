import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1',26950))

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotion_last = {}
emotion_current = {}

EMOTION_THRESHOLD = 2
frame_counter = {}

json_file = open('./emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights('./emotion_model.weights.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        print(ret)
        continue
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    if len(faces) != len(emotion_last):
        emotion_last = {}
        emotion_current = {}
        frame_counter = {}
    
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray_frame, (48, 48))
        cropped_img = cropped_img.astype("float32") / 255.0 
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
        
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]
        
        if idx not in emotion_last:
            emotion_last[idx] = emotion_label
            emotion_current[idx] = emotion_label
            frame_counter[idx] = 0
        
        if emotion_label != emotion_last[idx]:
            if emotion_label == emotion_current[idx]:
                frame_counter[idx] += 1
            else:
                frame_counter[idx] = 0
                emotion_current[idx] = emotion_label
                
            if frame_counter[idx] > EMOTION_THRESHOLD:
                emotion_last[idx] = emotion_label
                frame_counter[idx] = 0
                
            cv2.putText(frame, emotion_last[idx], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            byt = emotion_current[idx].encode()
            client.send(byt)
        else:
            cv2.putText(frame, emotion_last[idx], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()