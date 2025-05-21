import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io
import os
import threading
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Constants
IMG_HEIGHT, IMG_WIDTH = 48, 48
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(
    page_title="Facial Emotion Recognition",
    layout="wide"
)

# App title
st.title("Facial Emotion Recognition")

@st.cache_resource
def load_fer_model():
    """Load the FER model once and cache it"""

    if not os.path.exists('fer_final_model.h5'):
        st.error("Model file 'fer_final_model.h5' not found. Please make sure it's in the same directory as the app.")
        st.stop()
        
    try:
        return load_model('fer_final_model.h5', compile=False)
    except ValueError as e:
        if 'batch_shape' in str(e):
            st.error("Model incompatibility detected. Try using a newer TensorFlow version.")
            st.info("Check that your TensorFlow version matches what was used to create the model.")
            st.stop()
        else:
            st.error(f"Error loading model: {e}")
            st.stop()

@st.cache_resource
def load_face_cascade():
    """Load the Haar cascade once and cache it"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        st.error(f"Haar cascade file not found at: {cascade_path}")
        st.info("Make sure OpenCV is properly installed with Haar cascades.")
        st.stop()
    return cv2.CascadeClassifier(cascade_path)

# resource loading with error catching
try:
    model = load_fer_model()
    face_cascade = load_face_cascade()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

def preprocess_face(face_img):
    """Preprocess a face image for the model"""
    # Convert to grayscale
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to model
    face_img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values
    face_img = face_img / 255.0
    
    # Reshape for model input
    face_img = face_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    return face_img

def predict_emotion(face_img):
    """Predict emotion from preprocessed face"""
    prediction = model.predict(face_img)
    emotion_idx = np.argmax(prediction[0])
    emotion = EMOTIONS[emotion_idx]
    confidence = float(prediction[0][emotion_idx])
    
    return emotion, confidence

def process_image(image):
    """Process a single image and detect emotions"""
    if image is None:
        return None, []
    
    # Make a copy to draw on
    result_img = image.copy()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    results = []
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess face
        face_input = preprocess_face(face_roi)
        
        # Predict emotion
        emotion, confidence = predict_emotion(face_input)
        
        # Draw rectangle around face
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text with emotion
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        results.append({
            "emotion": emotion,
            "confidence": f"{confidence:.2f}",
            "coordinates": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
        })
    
    return result_img, results

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Process uploaded image
            img = np.array(Image.open(uploaded_file))
            
            # Convert RGB to BGR for OpenCV processing
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Process image
            result_img, results = process_image(img)
            
            # Convert back to RGB for display
            if len(result_img.shape) == 3 and result_img.shape[2] == 3:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(result_img, caption="Processed Image", use_column_width=True)
            
            with col2:
                st.subheader("Detected Emotions")
                if not results:
                    st.info("No faces detected in the image.")
                else:
                    for i, result in enumerate(results):
                        st.markdown(f"**Face {i+1}**")
                        st.markdown(f"- Emotion: {result['emotion']}")
                        st.markdown(f"- Confidence: {result['confidence']}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

with tab2:
    st.header("Live Webcam")
    st.info("Click 'Start' to begin webcam facial emotion recognition. Make sure to allow camera access when prompted.")
    
    if 'webcam_results' not in st.session_state:
        st.session_state.webcam_results = []

    results_placeholder = st.empty()
    
    # callback for WebRTC
    class VideoProcessor:
        def __init__(self):
            self.results = []
            self.frame_lock = threading.Lock()
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Process the frame
            result_img, face_results = process_image(img)
            
            # Update the results in a thread-safe way
            with self.frame_lock:
                self.results = face_results
                
            # Update session state with latest results
            st.session_state.webcam_results = face_results
            
            # Return the processed frame
            return av.VideoFrame.from_ndarray(result_img, format="bgr24")
    
    # Create WebRTC streamer
    ctx = webrtc_streamer(
        key="fer-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display results
    if ctx.state.playing:
        with results_placeholder.container():
            st.subheader("Detected Emotions")
            results = st.session_state.webcam_results
            if not results:
                st.info("No faces detected yet. Make sure your face is visible to the camera.")
            else:
                for i, result in enumerate(results):
                    st.markdown(f"**Face {i+1}**: {result['emotion']} (Confidence: {result['confidence']})")
    else:
        with results_placeholder.container():
            st.info("Webcam is not active. Click 'Start' to begin face detection.")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a trained model to recognize facial emotions.
    
    Upload an image or use your webcam to see it in action.
    
    Emotions detected:
    - Angry
    - Disgust
    - Fear
    - Happy
    - Neutral
    - Sad
    - Surprise
    """
)