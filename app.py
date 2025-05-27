import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load trained model
model = joblib.load('./model/kidney_model.pkl')

# --- Navigation Sidebar ---
page = st.sidebar.radio("Navigation", ["Kidney Disease Predictor", "About"])

# --- Main Prediction Page ---
if page == "Kidney Disease Predictor":
    st.title("🩺 Kidney Disease Risk Predictor")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        age = st.number_input("Age", min_value=5, step=1)
        bp = st.number_input("Blood Pressure", min_value=50, max_value=200, step=1)
        sg = st.number_input("Specific Gravity", format="%.3f", step=0.001)
        al = st.number_input("Albumin", format="%.3f", step=0.001)
        su = st.number_input("Sugar", format="%.3f", step=0.001)
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    with col2:
        ba = st.selectbox("Bacteria", ["notpresent", "present"])
        bgr = st.number_input("Blood Glucose Random", 0.0)
        bu = st.number_input("Blood Urea", 0.0)
        sc = st.number_input("Serum Creatinine", 0.0)
        sod = st.number_input("Sodium", 0.0)
        pot = st.number_input("Potassium", 0.0)
        hemo = st.number_input("Hemoglobin", 0.0)
        pcv = st.number_input("Packed Cell Volume", 0.0)
    with col3:
        wc = st.number_input("White Blood Cell Count", 0.0)
        rc = st.number_input("Red Blood Cell Count", 0.0)
        htn = st.selectbox("Hypertension", ["no", "yes"])
        dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
        cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["no", "yes"])
        ane = st.selectbox("Anemia", ["no", "yes"])

    # --- Prediction ---
    if st.button("Predict"):
        # Encode categorical values
        rbc = 1 if rbc == "abnormal" else 0
        pc = 1 if pc == "abnormal" else 0
        pcc = 1 if pcc == "present" else 0
        ba = 1 if ba == "present" else 0
        htn = 1 if htn == "No" else 0
        dm = 1 if dm == "No" else 0
        cad = 1 if cad == "yes" else 0
        appet = 1 if appet == "poor" else 0
        pe = 1 if pe == "yes" else 0
        ane = 1 if ane == "yes" else 0

        # Prepare DataFrame for model
        features = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr,
                                  bu, sc, sod, pot, hemo, pcv, wc, rc,
                                  htn, dm, cad, appet, pe, ane]],
                                columns=[
                                    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
                                    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                                    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
                                ])

        # Predict
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1] * 100

        # Output
        if prediction == 1:
            st.error(f"⚠️ High Risk of Chronic Kidney Disease! (Probability: {proba:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Chronic Kidney Disease (Probability: {proba:.2f}%)")

# --- About Page ---
elif page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
        This application predicts the risk of **Chronic Kidney Disease (CKD)** using patient data.
        
        The model was trained with clinical data including:
        - Vital signs (e.g., blood pressure, glucose)
        - Blood and urine test results
        - Categorical medical observations

        ---
        **Developer:** Hasan Uddin  
        **Model:** Scikit-learn  
        **UI:** Streamlit  
    """)

    col1, col2, col3 = st.columns([10, 1, 1])
    with col3:
        icon = Image.open("./assets/icon.png")
        st.image(icon, width=80)
