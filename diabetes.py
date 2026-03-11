import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score    
import pickle                             
import streamlit as st

df = pd.read_csv('diabetes.csv')

cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Diabetes Predictor Pro", layout="wide")

st.title("🏥 Diabetes Prediction System")
st.markdown("This app uses **Ensemble Learning** and **Standardized Scaling** for high accuracy.")

col1, col2 = st.columns(2)

with col1:
    preg = st.slider("Pregnancies", 0, 17, 3)
    glu = st.number_input("Glucose Level", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 122, 70)
    sk = st.number_input("Skin Thickness", 0, 99, 20)

with col2:
    ins = st.number_input("Insulin Level", 0, 846, 79)
    bmi = st.number_input("BMI", 0.0, 67.1, 32.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.42, 0.47)
    age = st.slider("Age", 21, 81, 33)

if st.button("Run Diagnostic Analysis"):
    input_data = np.array([[preg, glu, bp, sk, ins, bmi, dpf, age]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    st.subheader("Analysis Results:")
    if prediction[0] == 1:
        st.error(f"⚠️ Positive Result: High risk of diabetes (Probability: {prob[0][1]*100:.1f}%)")
    else:
        st.success(f"✅ Negative Result: Low risk of diabetes (Probability: {prob[0][0]*100:.1f}%)")

    st.write("### Review Entered Data:")
    st.table(pd.DataFrame(input_data, columns=X.columns))