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
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .confidence-bar {
        background: #FFE0B2;
        height: 25px;
        border-radius: 15px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 100%;
        border-radius: 15px;
        animation: fillUp 1.5s ease-out;
        position: relative;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes fillUp {
        from { width: 0%; }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 6px solid #FF5722;
        color: #333;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .footer {
        background: linear-gradient(135deg, #E91E63, #AD1457);
        color: white;
        text-align: center;
        padding: 3rem 2rem;
        margin-top: 4rem;
        border-radius: 20px 20px 0 0;
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .class-emoji {
        font-size: 4rem;
        margin: 0.5rem;
        display: inline-block;
        animation: bounce 2s infinite;
        transition: transform 0.3s ease;
        cursor: pointer;
    }
    
    .class-emoji:hover {
        transform: scale(1.2);
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .loading-container {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #4CAF50;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #E91E63, #FF9800, #4CAF50, #2196F3);
        border-radius: 4px;
        margin: 3rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .stats-card {
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        animation: fadeIn 1s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .feature-card {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(33, 150, 243, 0.4);
    }
    
    /* Fix for Streamlit headings */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #333 !important;
        font-weight: 600 !important;
    }
    
    /* Fix for image caption */
    .stImage > div > div {
        color: #333 !important;
        font-weight: 500 !important;
    }
    
    /* Remove Streamlit info boxes */
    .stAlert[data-baseweb="notification"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# CIFAR-10 class names and emojis
CLASS_NAMES = ['✈️ Airplane', '🚗 Automobile', '🐦 Bird', '🐱 Cat', '🦌 Deer', 
               '🐶 Dog', '🐸 Frog', '🐎 Horse', '🚢 Ship', '🚚 Truck']

CLASS_FACTS = {
    0: "Did you know? The Wright brothers' first flight lasted only 12 seconds! ✈️",
    1: "Fun fact: The average car has over 30,000 parts! 🚗",
    2: "Amazing: Birds are the only animals with feathers in the entire world! 🐦",
    3: "Incredible: Cats spend 70% of their lives sleeping! 🐱",
    4: "Wow: Deer can jump up to 10 feet high and run 30 mph! 🦌",
    5: "Awesome: Dogs have been human companions for over 15,000 years! 🐶",
    6: "Cool: Frogs can breathe through their skin underwater! 🐸",
    7: "Fascinating: Horses can sleep both lying down and standing up! 🐎",
    8: "Historical: The first ships were built over 10,000 years ago! 🚢",
    9: "Interesting: The first truck was invented in 1896 in Germany! 🚚"
}

@st.cache_resource
def load_cifar_model():
    """Load the trained CIFAR-10 model silently"""
    possible_paths = [
        os.path.join("model", "object-detection-model.h5"),
        os.path.join("model", "object-detection-model"),
        os.path.join("model", "v2_best_cifar10_model.h5"),
        "object-detection-model.h5",
        "object-detection-model",
        "v2_best_cifar10_model.h5"
    ]
    
    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                model = load_model(model_path)
                return model
        except Exception:
            continue
    
    return None

def preprocess_image(image):
    """Preprocess image for CIFAR-10 model"""
    image = image.resize((32, 32))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    return predicted_class, confidence

# Header Section
st.markdown('<h1 class="main-header">CIFAR-10 Vision Explorer 🚀</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI-powered portal into the world of object recognition</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Model Introduction Section
with st.container():
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2>🎯 Welcome to AI Vision Explorer!</h2>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                Step into the future of computer vision! Our advanced neural network can instantly recognize and classify objects in your images. 
                Using state-of-the-art deep learning techniques, this AI system has been trained to identify 10 different categories of objects 
                with remarkable accuracy.
            </p>
            <p style="font-size: 1rem; margin-top: 1rem; color: #666;">
                <strong>🚀 How it works:</strong> Simply upload any image, and watch as our AI analyzes every pixel to determine what object it sees. 
                The system provides both the prediction and a confidence score!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: #333; margin-bottom: 2rem; font-size: 1.5rem;">🎨 Recognizable Objects</h3>
        """, unsafe_allow_html=True)
        
        # Display class emojis in a grid
        emoji_rows = [['✈️', '🚗', '🐦', '🐱', '🦌'], ['🐶', '🐸', '🐎', '🚢', '🚚']]
        for row in emoji_rows:
            emoji_cols = st.columns(5)
            for i, emoji in enumerate(row):
                with emoji_cols[i]:
                    st.markdown(f'<div class="class-emoji" title="{CLASS_NAMES[emoji_rows[0].index(emoji) if emoji in emoji_rows[0] else emoji_rows[1].index(emoji) + 5]}">{emoji}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Load model
model = load_cifar_model()

if model is not None:
    # Features Section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## ⭐ Key Features")
    
    feature_cols = st.columns(4)
    features = [
        ("🎯", "High Accuracy", "86% validation accuracy"),
        ("⚡", "Lightning Fast", "Results in milliseconds"),
        ("🔍", "Deep Learning", "Advanced CNN architecture"),
        ("🎨", "Easy to Use", "Just drag and drop!")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with feature_cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <h4 style="margin: 0.5rem 0; font-size: 1.1rem;">{title}</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Image Upload Section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📸 Upload & Analyze Your Image")
    
    uploaded_file = st.file_uploader(
        "",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing objects like animals, vehicles, or aircraft for instant AI analysis!"
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.image(image, caption="📷 Your uploaded image ready for analysis", use_container_width=True)
            
            # Prediction button
            if st.button("🎯 Analyze Image with AI", type="primary", use_container_width=True):
                # Show loading animation
                loading_placeholder = st.empty()
                result_placeholder = st.empty()
                
                with loading_placeholder.container():
                    st.markdown("""
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                        <h3 style="color: #333; margin: 1rem 0;">🔍 AI is analyzing your image...</h3>
                        <p style="color: #666;">Our neural network is processing thousands of features!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Process image
                time.sleep(1.5)  # Dramatic pause
                predicted_class, confidence = predict_image(model, image)
                
                # Clear loading and show results
                loading_placeholder.empty()
                
                with result_placeholder.container():
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="margin-bottom: 1rem;">🎉 AI Analysis Complete!</h2>
                        <div style="font-size: 5rem; margin: 1rem 0;">{CLASS_NAMES[predicted_class].split()[0]}</div>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">{CLASS_NAMES[predicted_class]}</h1>
                        <p style="font-size: 1.3rem; margin-bottom: 1rem; opacity: 0.9;">Confidence Level</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100:.1f}%;"></div>
                        </div>
                        <p style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">{confidence*100:.1f}%</p>
                        <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 15px; margin-top: 2rem;">
                            <p style="font-size: 1.1rem; font-style: italic; margin: 0;">💡 {CLASS_FACTS[predicted_class]}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Model Performance Section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Model Performance")
    
    perf_cols = st.columns(3)
    with perf_cols[0]:
        st.markdown("""
        <div class="stats-card">
            <h2 style="margin: 0; font-size: 3rem;">82%</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Training Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_cols[1]:
        st.markdown("""
        <div class="stats-card" style="background: linear-gradient(135deg, #FF9800, #F57C00);">
            <h2 style="margin: 0; font-size: 3rem;">86%</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Validation Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_cols[2]:
        st.markdown("""
        <div class="stats-card" style="background: linear-gradient(135deg, #E91E63, #AD1457);">
            <h2 style="margin: 0; font-size: 3rem;">10</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Object Classes</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-card" style="border-left-color: #f44336; background: #ffebee;">
        <h3 style="color: #d32f2f;">❌ Model Loading Error</h3>
        <p style="color: #666;">The AI model couldn't be loaded. Please ensure your model file is properly configured.</p>
        
        <h4 style="color: #333; margin-top: 2rem;">🔧 Quick Fix Steps:</h4>
        <ol style="color: #666; line-height: 1.8;">
            <li><strong>Rename your model file</strong> to include the <code>.h5</code> extension</li>
            <li><strong>Place it in the <code>model/</code> folder</strong> next to your Python script</li>
            <li><strong>Expected filename:</strong> <code>object-detection-model.h5</code></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <h2 style="margin-bottom: 1rem; position: relative; z-index: 1;">Made with ❤️ by Asjad Raza</h2>
    <p style="font-size: 1.2rem; margin: 1rem 0; position: relative; z-index: 1;">
        BSCS @ FAST NUCES – Passionate about AI 🤖, Code 💻, and Creativity 🎨
    </p>
    <p style="margin: 1rem 0; opacity: 0.9; position: relative; z-index: 1;">
        Powered by TensorFlow • Keras • Streamlit • Deep Learning
    </p>
    <div style="margin-top: 2rem; font-size: 2rem; position: relative; z-index: 1;">
        🚀 🎯 🌟 ⭐ 🔥
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 🤖 AI Model Details")
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <p><strong>🏗️ Architecture:</strong> Convolutional Neural Network</p>
        <p><strong>📊 Dataset:</strong> CIFAR-10 (60,000 images)</p>
        <p><strong>🔢 Input Size:</strong> 32×32 RGB pixels</p>
        <p><strong>🎯 Classes:</strong> 10 object categories</p>
        <p><strong>⚡ Framework:</strong> TensorFlow/Keras</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎯 Accuracy Metrics")
    st.success("🏆 Training: 82.1%")
    st.success("✅ Validation: 86.0%")
    st.info("📈 Epochs: 30")
    
    st.markdown("## 💡 Pro Tips")
    st.markdown("""
    - 📷 Use clear, well-lit images
    - 🎯 Object should be main subject  
    - 🌈 Try different angles
    - 📱 Mobile photos work great
    - 🖼️ JPG/PNG formats supported
    """)
    
    st.markdown("## 🔥 Features")
    st.markdown("""
    - ⚡ Real-time predictions
    - 🎨 Beautiful interface
    - 📊 Confidence scoring
    - 💡 Educational facts
    - 📱 Mobile responsive
    """)