import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


POSE_TEMPLATES = {
    "Mountain Pose": np.array([(0.5, 0.1, 0.0), (0.5, 0.2, 0.0), (0.4, 0.3, 0.0), (0.6, 0.3, 0.0)]),
    "Warrior Pose": np.array([(0.5, 0.1, 0.0), (0.4, 0.2, 0.0), (0.6, 0.2, 0.0), (0.3, 0.3, 0.0)]),
}


def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((landmark.x, landmark.y, landmark.z))
    return np.array(keypoints)


def calculate_similarity(user_keypoints, template_keypoints):
    if user_keypoints is None or template_keypoints is None:
        return 0
    distances = [
        distance.euclidean(user, template)
        for user, template in zip(user_keypoints[:len(template_keypoints)], template_keypoints)
    ]
    return 1 - (np.mean(distances) if distances else 0)


st.title("AI Yoga Assistant")
st.sidebar.header("Yoga Assistant Settings")

yoga_pose = st.sidebar.selectbox("Select a Yoga Pose", list(POSE_TEMPLATES.keys()))
difficulty = st.sidebar.radio("Select Difficulty Level", ["Beginner", "Intermediate", "Advanced"])
uploaded_video = st.sidebar.file_uploader("Upload Your Video", type=["mp4", "mov", "avi"])

if uploaded_video:
  
    user_video_path = "user_video.mp4"
    with open(user_video_path, "wb") as f:
        f.write(uploaded_video.read())

    
    template_keypoints = POSE_TEMPLATES[yoga_pose]

    
    cap = cv2.VideoCapture(user_video_path)
    similarity_scores = []
    st.text("Evaluating your pose...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        user_keypoints = extract_keypoints(frame)

        
        score = calculate_similarity(user_keypoints, template_keypoints)
        similarity_scores.append(score)

       
        if user_keypoints is not None:
            mp_drawing.draw_landmarks(frame, pose.process(frame).pose_landmarks, mp_pose.POSE_CONNECTIONS)

       
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    
    st.header("Evaluation Results")
    st.line_chart(similarity_scores)

    st.success(f"Your average alignment score: {np.mean(similarity_scores) * 100:.2f}%")
    st.text("Processing complete! Keep practicing to improve alignment and balance.")
else:
    st.warning("Please upload a video to proceed.")
