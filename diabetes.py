import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier # الموديل الأقوى
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import pickle

# 1. تحميل ومعالجة البيانات باحترافية
@st.cache_data
def load_and_optimize_data():
    df = pd.read_csv('diabetes.csv')
    
    # التعامل مع القيم الصفرية كقيم مفقودة
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    
    # ملء القيم بناءً على الوسيط لكل فئة (صحياً أدق بكتير)
    for col in cols_to_fix:
        df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y

X, y = load_and_optimize_data()

# 2. تقسيم البيانات وتوسيعها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. بناء موديل XGBoost مع إعدادات الدقة العالية
# 
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.8, # لموازنة البيانات
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

# حساب الدقة للـ Terminal
predictions = model.predict(X_test_scaled)
acc = accuracy_score(y_test, predictions)
print(f"New Optimized Accuracy: {acc * 100:.2f}%")
print(f"New Optimized Accuracy: {classification_report(y_test, predictions)}")


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











 