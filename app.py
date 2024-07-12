import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import mediapipe as mp

# Initialize MediaPipe Holistic and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load video model architecture from JSON file
video_model_architecture_path = 'video.json'  # Update with your file path
with open(video_model_architecture_path, 'r') as f:
    video_model_json = f.read()

# Reconstruct the video model
video_model = model_from_json(video_model_json)

# Load video model weights from HDF5 file
video_model_weights_path = 'signvideo_Model.h5'  # Update with your file path
video_model.load_weights(video_model_weights_path)

# Load image model architecture from JSON file
image_model_architecture_path = 'image.json'  # Update with your file path
with open(image_model_architecture_path, 'r') as f:
    image_model_json = f.read()

# Reconstruct the image model
image_model = model_from_json(image_model_json)

# Load image model weights from HDF5 file
image_model_weights_path = 'image_Model.h5'  # Update with your file path
image_model.load_weights(image_model_weights_path)

# Labels mapping for video model
video_labels = {0: 'how_are_you', 1: 'what_is_your_name'}

# Labels mapping for image model
image_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
                14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
                26: 'del', 27: 'nothing', 28: 'space'}

# Function to analyze frames using MediaPipe Holistic
def analyze_frames(frames):
    holistic = mp_holistic.Holistic()
    all_keypoints = []
    
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Initialize lists for keypoints
        pose = np.zeros(33 * 4)     # 33 landmarks with 4 coordinates each (x, y, z, visibility)
        face = np.zeros(468 * 4)    # 468 landmarks with 4 coordinates each
        left_hand = np.zeros(21 * 4)  # 21 landmarks with 4 coordinates each
        right_hand = np.zeros(21 * 4) # 21 landmarks with 4 coordinates each
        
        # Extract keypoints if available
        if results.pose_landmarks:
            pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten()
        if results.face_landmarks:
            face = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.face_landmarks.landmark]).flatten()
        if results.left_hand_landmarks:
            left_hand = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            right_hand = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.right_hand_landmarks.landmark]).flatten()
        
        # Concatenate all keypoints into a single array
        keypoints = np.concatenate([pose, face, left_hand, right_hand])
        all_keypoints.append(keypoints)
    
    holistic.close()  # Close the holistic model to release resources
    return np.array(all_keypoints)

# Function to predict sign language from a single image
def predict_from_image(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the required input size for the model
    image_resized = cv2.resize(image_rgb, (28, 28))
    
    # Normalize the image data to 0-1 range
    image_normalized = image_resized / 255.0
    
    # Expand dimensions to match the model's expected input shape
    image_input = np.expand_dims(image_normalized, axis=0)
    
    # Predict the sign language
    prediction = image_model.predict(image_input)
    predicted_label = image_labels[np.argmax(prediction)]
    
    return predicted_label

# Streamlit app
def main():
    st.title('Sign Language Prediction App')

    # Option to choose between image or video upload
    option = st.selectbox('Choose an option', ['Upload an Image', 'Upload a Video'])

    if option == 'Upload an Image':
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Convert the uploaded file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Display the uploaded image
            st.image(image, channels="BGR")
            
            # Predict the sign language
            predicted_label = predict_from_image(image)
            st.write("Predicted label:", predicted_label)
    
    elif option == 'Upload a Video':
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded video to a temporary file
            with open("temp_video.mov", "wb") as f:
                f.write(uploaded_file.read())
            
            # Open the video using OpenCV VideoCapture
            cap = cv2.VideoCapture("temp_video.mov")
            if not cap.isOpened():
                st.error("Error opening video file.")
                return
            
            frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            # Analyze frames and make predictions
            keypoints = analyze_frames(frames)
            keypoints = np.expand_dims(keypoints, axis=0)
            
            prediction = video_model.predict(keypoints)
            predicted_label = video_labels[np.argmax(prediction)]
            
            st.write("Predicted label:", predicted_label)

if __name__ == '__main__':
    main()
