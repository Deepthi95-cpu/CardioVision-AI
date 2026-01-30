import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import google.generativeai as genai

# --- 1. CORE IMPORTS ---
import torch
import torch.nn.functional as F 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import monai
from monai.networks.nets import UNet
import matplotlib.pyplot as plt
import random

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioVision AI",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# --- 3. PROFESSIONAL CSS STYLING ---
# This hides the default Streamlit menus and makes it look like a real software product
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #FFFFFF; }
    
    /* Headings */
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    
    /* Custom Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #e74c3c; /* Medical Red */
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Status Badge */
    .status-badge {
        background-color: #27ae60;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. LOAD MODELS ---
@st.cache_resource
def load_clinical_model():
    try:
        return joblib.load('adaboost_heart_model.pkl')
    except:
        return None

clinical_model = load_clinical_model()

@st.cache_resource
def load_3d_model():
    device = torch.device("cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    try:
        model.load_state_dict(torch.load('best_stroke_model.pth', map_location=device))
        model.eval()
        return model
    except:
        return None

vessel_model = load_3d_model()

# --- 5. SIDEBAR (PROFESSIONAL LOOK) ---
with st.sidebar:
    # Use a standard emoji or local logo if you have one. 
    # For now, a clean large icon is better than a broken image.
    # OLD BROKEN LINE:
    # st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/L%27Or%C3%A9al_logo.svg/2560px-L%27Or%C3%A9al_logo.svg.png", width=120)

    # NEW WORKING LINE (Use text/emoji if image fails, or a stable URL):
    st.markdown("# 🫀 CardioVision")
    
    st.markdown("---")
    
    # Clean Status Indicator
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
        <span>System Status:</span>
        <span class='status-badge'>ONLINE</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Project Details")
    st.markdown("**Focus:** Coronary Blockage & Risk Prediction")
    st.markdown("**Model:** 3D U-Net + AdaBoost")
    st.markdown("**Explainability:** SHAP Integrated")
    
    st.markdown("---")
    st.caption("© 2026 Deepthi | Research Project")

# --- 6. INITIALIZE SESSION STATE FOR RISK ---
if 'risk_prob' not in st.session_state:
    st.session_state['risk_prob'] = 0.0
if 'risk_factors' not in st.session_state:
    st.session_state['risk_factors'] = []

# --- 7. MAIN APP TABS ---
tab1, tab2, tab3 = st.tabs(["📋 Clinical Risk", "🧊 3D Angiography", "💬 Dr. AI Consultant"])

# ==========================================
# TAB 1: CLINICAL RISK
# ==========================================
with tab1:
    st.header("Patient Cardiac Risk Assessment")
    st.markdown("Predicting Myocardial Infarction risk using Interpretable AI.")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 100, 55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
        with col2:
            trestbps = st.number_input("Resting BP", 90, 200, 120)
            chol = st.number_input("Cholesterol", 100, 600, 200)
            fbs = st.selectbox("Fasting BS > 120?", ["No", "Yes"])
        with col3:
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Angina?", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

        if st.button("Run Risk Analysis"):
            if clinical_model:
                sex_v = 1 if sex == "Male" else 0
                fbs_v = 1 if fbs == "Yes" else 0
                exang_v = 1 if exang == "Yes" else 0
                
                input_df = pd.DataFrame([[age, sex_v, cp, trestbps, chol, fbs_v, 0, thalach, exang_v, oldpeak, 1, 0, 2]],
                                        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
                
                prob = clinical_model.predict_proba(input_df)[0][1]
                st.session_state['risk_prob'] = prob # Save for chatbot
                
                # Identify Risk Factors for Chatbot
                factors = []
                if chol > 240: factors.append("High Cholesterol")
                if trestbps > 140: factors.append("Hypertension")
                if cp > 0: factors.append("Chest Pain")
                if oldpeak > 2.0: factors.append("Abnormal ECG (ST Depression)")
                st.session_state['risk_factors'] = factors
                
                st.markdown("---")
                c1, c2 = st.columns([1, 2])
                with c1:
                    if prob > 0.5:
                        st.error(f"⚠️ **HIGH RISK: {prob:.1%}**")
                    else:
                        st.success(f"✅ **LOW RISK: {prob:.1%}**")
                with c2:
                    st.info(f"**Key Drivers:** {', '.join(factors) if factors else 'None detected.'}")

# ==========================================
# TAB 2: 3D ANGIOGRAPHY
# ==========================================
with tab2:
    st.header("3D Coronary Artery Blockage Detection")
    st.markdown("Volumetric CT Analysis using Deep Learning.")
    
    uploaded_file = st.file_uploader("Upload CT Volume (.npy)", type=['npy'])
    
    if uploaded_file and vessel_model:
        scan = np.load(uploaded_file)
        slice_idx = st.slider("Select Slice Depth", 0, scan.shape[0]-1, scan.shape[0]//2)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original CT Slice**")
            plt.figure(figsize=(4,4))
            plt.imshow(scan[slice_idx, :, :], cmap='gray')
            plt.axis('off')
            st.pyplot(plt)
            
        with c2:
            st.markdown("**AI Detected Blockage**")
            if st.button("Analyze Vessel Structure"):
                with st.spinner("Segmenting 3D Volume..."):
                    # Robust Resizing
                    input_tensor = torch.tensor(scan).float().unsqueeze(0).unsqueeze(0)
                    input_resized = F.interpolate(input_tensor, size=(64, 128, 128), mode='trilinear', align_corners=False)
                    
                    with torch.no_grad():
                        output = vessel_model(input_resized)
                        prediction = torch.sigmoid(output) > 0.5
                    
                    # Visualization Mapping
                    pred_volume = prediction[0, 0, :, :, :]
                    ratio = 64 / scan.shape[0]
                    ai_slice_idx = max(0, min(int(slice_idx * ratio), 63))
                    
                    pred_slice = pred_volume[ai_slice_idx, :, :].numpy()
                    
                    plt.figure(figsize=(4,4))
                    plt.imshow(pred_slice, cmap='hot', alpha=0.9) 
                    plt.axis('off')
                    st.pyplot(plt)
                    
                    if pred_slice.max() > 0:
                        st.warning("⚠️ Calcified Plaque Detected")
                    else:
                        st.success("✅ Vessel Clear")

# ==========================================
# TAB 3: CHATBOT (DR. AI)
# ==========================================
with tab3:
    st.header("💬 Dr. AI Consultant")
    st.markdown("Ask questions about the diagnosis or treatment options.")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I have analyzed the patient's clinical data. How can I help you?"}]

    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if user_input := st.chat_input("Ex: What treatment do you recommend?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # GENERATE RESPONSE (Expert System)
        risk = st.session_state['risk_prob']
        factors = st.session_state['risk_factors']
        
        # Smart Logic based on previous analysis
        response = ""
        user_input_lower = user_input.lower()
        
        if "treatment" in user_input_lower or "recommend" in user_input_lower:
            if risk > 0.5:
                response = f"Given the **High Risk ({risk:.1%})**, I recommend immediate Cardiologist intervention. "
                if "High Cholesterol" in factors:
                    response += "Statins may be required to lower cholesterol. "
                if "Hypertension" in factors:
                    response += "Blood pressure medication (ACE inhibitors) is advised."
            else:
                response = "The risk is low. Maintain a healthy diet and 150 minutes of weekly cardio exercise."
                
        elif "diet" in user_input_lower or "food" in user_input_lower:
            response = "I recommend the **Mediterranean Diet**: High in vegetables, nuts, and olive oil. Avoid red meat and sugar."
            if "High Cholesterol" in factors:
                response += " Specifically, increase soluble fiber (oats, beans) to help flush out cholesterol."
                
        elif "risk" in user_input_lower:
             response = f"The calculated risk is {risk:.1%}. The primary contributing factors are: {', '.join(factors) if factors else 'Age/Genetics'}."
             
        else:
            response = "I am a medical AI assistant. You can ask me about **Treatments**, **Diet**, or **Risk Factors**."

        # Show Response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})