# 🫀 CardioVision: Explainable AI for Heart Stroke Prediction

### 🏆 Project Overview
**Developed an interpretable machine learning model for stroke/heart risk prediction using clinical data, leveraging SHAP to provide feature-level reasoning.**

Currently focused on building a **3D U-Net model for detecting coronary blockages** and integrating a chatbot to offer interactive clinical support.

### 🛠️ Tech Stack
- **Clinical Model:** AdaBoost Classifier + SHAP (Explainable AI)
- **3D Imaging:** 3D U-Net (PyTorch/MONAI) for CT Angiography segmentation.
- **Interface:** Streamlit (Web Deployment)
- **Data:** Kaggle Healthcare Dataset

### 📊 Features
1. **Cardiac Risk Profile:** Analyzes Chest Pain type (cp), Cholesterol, and BP to predict Heart Attack risk.
2. **Feature Reasoning:** Uses SHAP waterfall plots to explain *why* a patient is at risk.
3. **3D Blockage Detection:** Segments coronary arteries from 3D Volumetric data to find plaque buildup.

### 🚀 Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the App: `streamlit run app.py`

---
*Created by Deepthi*