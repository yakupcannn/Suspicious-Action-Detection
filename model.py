import cv2 
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

print("Libraries Installed")

## Load YOLO Model
yolo_model = YOLO("yolov8n.pt") # installation for yolov8 nano pretrained
print("YOLOv8 model installed")

## Initiate Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils # to draw landmark on the media (image or video)
print("Pose initiated")

## Open Input Video or Stream
cap = cv2.VideoCapture(0)

## Get video features
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define videowriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
SAVED_VIDEO_PATH = "./saved_annotations"
#out = cv2.VideoWriter(SAVED_VIDEO_PATH) This is optional

def detect_actions(landmarks):
    if not landmarks:
        return False,""
    # get the position of nose, hips, wrist and heels 
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
    
    # Calculate average hips and heels
    hip_y = (left_hip.y + right_hip.y) / 2
    heel_y = (left_heel.y + right_heel.y) / 2

    # Lift Up Motion
    if left_wrist.y < nose.y and right_wrist.y < nose.y:
        return True, "Lifting Up" 
    elif ((right_hip.y - right_wrist.y < 0.02) and (right_wrist.x  - right_hip.x < 0.2) or (left_hip.y - left_wrist.y < 0.02) and (left_wrist.x  - left_hip.x < 0.2)):
        return True,"Digging Down"
    elif (hip_y - heel_y) < -0.1 and (hip_y - heel_y) > -0.5:
        print(hip_y - heel_y)
        return True,"Crouching"

    else:
        return False, "None"


while True:
    ret,frame = cap.read()
    if not ret:
        break
    # Yolo Detection
    results = yolo_model(frame,conf=0.5)
    annotated_frame = results[0].plot()

    # Media Pipe Pose Detection
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    # Analyse For Suspicious Activities
    suspicious, label = False, ""
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark
        suspicious, label = detect_actions(landmarks)

    # Mark suspicious activity
    if suspicious:
        cv2.putText(annotated_frame,f"Suspicious Action Detected: {label}",(30,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)

    # To save annotated frame
    #out.write(annotated_frame)
    cv2.imshow("Motion Detection",annotated_frame)
    # To quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break