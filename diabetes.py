



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import streamlit as st
import pickle

# ────────────────────────────────────────────────
# 1. Load & Preprocess Data (cached)
# ────────────────────────────────────────────────
@st.cache_data
def load_and_optimize_data():
    df = pd.read_csv('diabetes.csv')
    
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    
    for col in cols_to_fix:
        df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y

X, y = load_and_optimize_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ────────────────────────────────────────────────
# 2. Train & Save Model  →  Run this ONCE locally, then comment it out
# ────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

classification_report = classification_report(y_test, y_pred)

print(f"Optimized Model Performance:\n{classification_report}")

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ────────────────────────────────────────────────
# 3. Load the trained model & scaler
# ────────────────────────────────────────────────
model  = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor Pro",
    layout="wide",
    page_icon="🏥"
)

st.title("🏥 Diabetes Prediction System")
st.markdown("Powered by **XGBoost** with improved data handling and standard scaling.")

# ── User Input Section ───────────────────────────
col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.slider("Pregnancies", 0, 17, 3)
    glucose        = st.number_input("Glucose", 0, 200, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 122, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 99, 20)

with col2:
    insulin     = st.number_input("Insulin", 0, 846, 79)
    bmi         = st.number_input("BMI", 0.0, 67.1, 32.0)
    dpf         = st.number_input("Diabetes Pedigree Function", 0.0, 2.42, 0.47)
    age         = st.slider("Age", 21, 81, 33)

# ── Prediction Section ───────────────────────────
if st.button("Run Prediction", type="primary"):
    
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    prob       = model.predict_proba(input_scaled)[0]
    prob_diabetes = prob[1] * 100
    
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠️ **Positive** – High risk of diabetes  \nProbability: **{prob_diabetes:.1f}%**")
        st.markdown("**Recommendation:** Please consult a doctor as soon as possible.")
    else:
        st.success(f"✅ **Negative** – Low risk of diabetes  \nProbability: **{prob[0]*100:.1f}%**")

    # Show entered values
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    input_df = pd.DataFrame(input_data, columns=feature_names)
    st.markdown("### Your Entered Values")
    st.dataframe(input_df.style.format(precision=1), use_container_width=True)

else:
    st.info("Adjust the values and click **Run Prediction** to see your result.", icon="ℹ️")

# Footer
st.markdown("---")
st.caption("Educational tool only – always consult a qualified healthcare professional for medical advice.")