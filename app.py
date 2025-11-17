import requests
import streamlit as st
from typing import Optional, Tuple, Dict

# =========================
# Config
# =========================
# Use your working API endpoint
API_URL = "https://cvd-detector-main-v2.fly.dev/predict"

# Path to your logo inside the project folder
LOGO_PATH = "image/logo.png"

# Mappings
GENDER_MAP = {"Female": 1, "Male": 2}
CHOLESTEROL_MAP = {"Normal": 1, "Above normal": 2, "Well above normal": 3}
GLUC_MAP = {"Normal": 1, "Above normal": 2, "Well above normal": 3}
YES_NO_MAP = {"No": 0, "Yes": 1}


# =========================
# Helper functions
# =========================
def classify_risk(probability: float) -> Tuple[str, str]:
    if probability < 0.3:
        return "Low risk", "green"
    elif probability < 0.6:
        return "Moderate risk", "orange"
    else:
        return "High risk", "red"


def call_api(payload: dict) -> Optional[dict]:
    try:
        response = requests.post(API_URL, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error {response.status_code}")
            st.text(response.text)
            return None
    except Exception as e:
        st.error("Error connecting to API.")
        st.exception(e)
        return None


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Cardio Risk Predictor", page_icon="❤️", layout="centered")

# ----- NEW: Add Logo -----
try:
    st.image(LOGO_PATH, width=180)
except:
    st.warning("Logo could not be loaded — check image path.")

# Header
st.markdown(
    """
    <h1 style="margin-bottom:0.2rem;">Cardio Risk Predictor</h1>
    <p style="color:#9ca3af; margin-bottom:1.2rem;">
        Estimate cardiovascular disease risk using an ML model trained on 70,000 patients.
    </p>
    """,
    unsafe_allow_html=True,
)

# Custom CSS
st.markdown(
    """
    <style>
        .main { background-color:#020617; color:#e5e7eb; }
        .risk-box {
            padding:1.2rem; border-radius:0.75rem;
            background:#020617; border:1px solid #1e293b;
            box-shadow:0 18px 40px rgba(15,23,42,0.7);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Inputs
with st.sidebar:
    st.markdown("### Patient Profile")
    age_years = st.slider("Age (years)", 25, 80, 52)
    gender = st.radio("Gender", list(GENDER_MAP.keys()), horizontal=True)

    st.markdown("### Measurements")
    height = st.slider("Height (cm)", 140, 210, 170)
    weight = st.slider("Weight (kg)", 40, 150, 80)

    st.markdown("### Blood Pressure")
    ap_hi = st.slider("Systolic BP", 90, 200, 130)
    ap_lo = st.slider("Diastolic BP", 60, 130, 80)

    st.markdown("### Lab Values")
    cholesterol = st.selectbox("Cholesterol", list(CHOLESTEROL_MAP.keys()))
    gluc = st.selectbox("Glucose", list(GLUC_MAP.keys()))

    st.markdown("### Lifestyle")
    smoke = st.radio("Smoker", list(YES_NO_MAP.keys()), horizontal=True)
    alco = st.radio("Alcohol", list(YES_NO_MAP.keys()), horizontal=True)
    active = st.radio("Physically Active", list(YES_NO_MAP.keys()), horizontal=True)

    predict_button = st.button("🔍 Predict Risk", use_container_width=True)

# Output area
st.markdown("#### Model Output")

if predict_button:
    payload = {
        "age_years": age_years,
        "gender": GENDER_MAP[gender],
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": CHOLESTEROL_MAP[cholesterol],
        "gluc": GLUC_MAP[gluc],
        "smoke": YES_NO_MAP[smoke],
        "alco": YES_NO_MAP[alco],
        "active": YES_NO_MAP[active],
    }

    with st.spinner("Contacting prediction API..."):
        result = call_api(payload)

    if result:
        prob = result["cardio_probability"]
        pred = result["cardio_prediction"]

        risk_label, color = classify_risk(prob)

        st.markdown(
            f"""
            <div class="risk-box">
                <h3 style="color:{color}; margin-top:0;">{risk_label}</h3>
                <p>Probability of disease: <b>{prob:.1%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(prob)
else:
    st.info("Fill in the info and click **Predict Risk**.")


