import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os
import time

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Vision Explorer 🚀",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for vibrant theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #E91E63, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px #E91E63); }
        to { filter: drop-shadow(0 0 30px #FF9800); }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #2E7D32;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .stApp {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFFFFF 100%);
    }
    
    .upload-box {
        border: 3px dashed #4CAF50;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(76, 175, 80, 0.1);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .upload-box:hover {
        border-color: #FF5722;
        background: rgba(255, 87, 34, 0.1);
        transform: translateY(-2px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #E91E63, #AD1457);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(233, 30, 99, 0.3);
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background: #FFE0B2;
        height: 20px;
        border-radius: 10px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 100%;
        border-radius: 10px;
        animation: fillUp 1.5s ease-out;
    }
    
    @keyframes fillUp {
        from { width: 0%; }
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #FF5722;
        color: #333;
    }
    
    .sample-image {
        border-radius: 10px;
        transition: transform 0.3s ease;
        cursor: pointer;
    }
    
    .sample-image:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(233, 30, 99, 0.3);
    }
    
    .footer {
        background: #E91E63;
        color: white;
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-radius: 15px 15px 0 0;
    }
    
    .class-emoji {
        font-size: 3rem;
        margin: 1rem;
        display: inline-block;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4CAF50;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #E91E63, #FF9800, #4CAF50);
        border-radius: 3px;
        margin: 2rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# CIFAR-10 class names and emojis
CLASS_NAMES = ['✈️ Airplane', '🚗 Automobile', '🐦 Bird', '🐱 Cat', '🦌 Deer', 
               '🐶 Dog', '🐸 Frog', '🐎 Horse', '🚢 Ship', '🚚 Truck']

CLASS_FACTS = {
    0: "Airplanes can fly at speeds over 600 mph! ✈️",
    1: "The first automobile was invented in 1885! 🚗",
    2: "Birds are the only animals with feathers! 🐦",
    3: "Cats sleep 12-16 hours per day! 🐱",
    4: "Deer can jump up to 10 feet high! 🦌",
    5: "Dogs have been human companions for over 15,000 years! 🐶",
    6: "Frogs can breathe through their skin! 🐸",
    7: "Horses can sleep both lying down and standing up! 🐎",
    8: "Ships have been used for transportation for over 10,000 years! 🚢",
    9: "The first truck was built in 1896! 🚚"
}

@st.cache_resource
def load_cifar_model():
    """Load the trained CIFAR-10 model"""
    try:
        model_path = os.path.join("model", "v2_best_cifar10_model.h5")
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = "v2_best_cifar10_model.h5"
        
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for CIFAR-10 model"""
    # Resize to 32x32 (CIFAR-10 input size)
    image = image.resize((32, 32))
    # Convert to array
    image_array = np.array(image)
    # Normalize
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    return predicted_class, confidence

# Header Section
st.markdown('<h1 class="main-header">CIFAR-10 Vision Explorer 🚀</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI-powered portal into the world of object recognition</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Model Introduction Section
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 About This AI Vision System</h3>
            <p>Welcome to the CIFAR-10 Vision Explorer! This intelligent system can identify 10 different types of objects in your images. 
            Simply upload a photo and watch as our neural network analyzes it with incredible precision.</p>
            <p><strong>Supported Objects:</strong> Airplanes, Automobiles, Birds, Cats, Deer, Dogs, Frogs, Horses, Ships, and Trucks!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; background: white; padding: 1rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <h4 style="color: #333; margin-bottom: 1rem;">Recognizable Classes</h4>
        """, unsafe_allow_html=True)
        
        # Display class emojis in a grid
        emoji_cols = st.columns(5)
        emojis = ['✈️', '🚗', '🐦', '🐱', '🦌']
        for i, emoji in enumerate(emojis):
            with emoji_cols[i]:
                st.markdown(f'<div class="class-emoji">{emoji}</div>', unsafe_allow_html=True)
        
        emoji_cols2 = st.columns(5)
        emojis2 = ['🐶', '🐸', '🐎', '🚢', '🚚']
        for i, emoji in enumerate(emojis2):
            with emoji_cols2[i]:
                st.markdown(f'<div class="class-emoji">{emoji}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Load model
model = load_cifar_model()

if model is not None:
    # Image Upload Section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📸 Upload Your Image")
    
    uploaded_file = st.file_uploader(
        "",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing objects like animals, vehicles, or aircraft"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.image(image, caption="Your uploaded image", use_column_width=True)
            
            # Prediction button
            if st.button("🎯 Analyze Image", type="primary"):
                with st.spinner(""):
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <div class="loading-spinner"></div>
                        <p>Analyzing your image... please hold tight! 🎯</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a small delay for dramatic effect
                    time.sleep(1)
                    
                    # Make prediction
                    predicted_class, confidence = predict_image(model, image)
                    
                    # Display results
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🎉 Prediction Result</h2>
                        <h1 style="font-size: 3rem; margin: 1rem 0;">{CLASS_NAMES[predicted_class]}</h1>
                        <p style="font-size: 1.2rem; margin-bottom: 1rem;">Confidence Level</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100:.1f}%;"></div>
                        </div>
                        <p style="font-size: 1.5rem; font-weight: bold;">{confidence*100:.1f}%</p>
                        <p style="margin-top: 1rem; font-style: italic;">💡 Fun Fact: {CLASS_FACTS[predicted_class]}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Sample Images Gallery
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## 🖼️ Try These Sample Images")
    
    # Create sample images section
    sample_cols = st.columns(5)
    sample_descriptions = [
        "Sample Airplane", "Sample Car", "Sample Bird", "Sample Cat", "Sample Dog"
    ]
    
    for i, col in enumerate(sample_cols):
        with col:
            # Create a placeholder for sample images
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 4rem; margin: 1rem 0;">{['✈️', '🚗', '🐦', '🐱', '🐶'][i]}</div>
                <p style="font-size: 0.9rem; color: #E91E63; font-weight: 600;">{sample_descriptions[i]}</p>
                <p style="font-size: 0.8rem; color: #666;">Click to upload your own!</p>
            </div>
            """, unsafe_allow_html=True)
    
else:
    st.error("❌ Model could not be loaded. Please ensure the model file is in the correct location.")
    st.info("📁 Expected location: ./model/v2_best_cifar10_model.h5")

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <h3>Made with ❤️ by Asjad Raza</h3>
    <p>BSCS @ FAST NUCES – Passionate about AI 🤖, Code 💻, and Creativity 🎨</p>
    <p>Trained on CIFAR-10 dataset • Powered by TensorFlow & Streamlit</p>
    <div style="margin-top: 1rem;">
        <span style="font-size: 1.5rem;">🚀 🎯 🌟</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 🔧 Model Info")
    st.info("""
    **Model Architecture:** Convolutional Neural Network
    
    **Dataset:** CIFAR-10
    
    **Input Size:** 32x32 RGB images
    
    **Classes:** 10 object categories
    
    **Framework:** TensorFlow/Keras
    """)
    
    st.markdown("## 📊 Performance")
    st.success("Training Accuracy: ~82%")
    st.success("Validation Accuracy: ~86%")
    
    st.markdown("## 💡 Tips")
    st.markdown("""
    - Use clear, well-lit images
    - Objects should be the main subject
    - Try different angles and backgrounds
    - JPEG, PNG formats work best
    """)