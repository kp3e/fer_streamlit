import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io

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

# Page config
st.set_page_config(
    page_title="Facial Emotion Recognition",
    layout="wide"
)

# App title
st.title("Facial Emotion Recognition")

@st.cache_resource
def load_fer_model():
    """Load the FER model once and cache it"""
    return load_model('fer_final_model.h5')

@st.cache_resource
def load_face_cascade():
    """Load the Haar cascade once and cache it"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load resources
model = load_fer_model()
face_cascade = load_face_cascade()

def preprocess_face(face_img):
    """Preprocess a face image for the model"""
    # Convert to grayscale if needed
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
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
    
    # Convert to grayscale for face detection
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
        # Process uploaded image
        img = np.array(Image.open(uploaded_file))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Process image
        result_img, results = process_image(img)
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

with tab2:
    st.header("Live Webcam")
    st.warning("Note: Webcam access requires permission from your browser.")
    
    # Start/stop webcam button
    run = st.checkbox("Start Webcam")
    
    # Placeholder for webcam feed
    image_place = st.empty()
    
    # Info placeholder for detected emotions
    info_place = st.empty()
    
    if run:
        # Use streamlit-webrtc for webcam
        try:
            cap = cv2.VideoCapture(0)
            
            while run:
                success, frame = cap.read()
                if not success:
                    st.error("Failed to access webcam")
                    break
                
                # Process frame
                result_img, results = process_image(frame)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # Update image display
                image_place.image(result_img, channels="RGB")
                
                # Update info display
                with info_place.container():
                    st.subheader("Detected Emotions")
                    if not results:
                        st.info("No faces detected")
                    else:
                        for i, result in enumerate(results):
                            st.markdown(f"**Face {i+1}**: {result['emotion']} ({result['confidence']})")
                
            # Release resources when stopped
            cap.release()
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")
    
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