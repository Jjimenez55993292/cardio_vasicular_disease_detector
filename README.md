# Cardio-Vascular Disease Prediction System (CVD Detector)
_A Machine Learning + FastAPI + Streamlit Application_  
🚀 **Backend API:** https://cvd-detector-main-v2.fly.dev/  
🌐 **Interactive App:** https://cvd-streamlit-app.fly.dev/

## 🩺 Introduction
Cardiovascular disease (CVD) is one of the leading causes of death globally.  
This project provides an **end‑to‑end ML system** capable of predicting cardiovascular risk using clinical and lifestyle data.

You can interact with the system through:

### ✔️ FastAPI Backend (Model Inference)  
🔗 https://cvd-detector-main-v2.fly.dev/

### ✔️ Streamlit Frontend (User Interface)  
🔗 https://cvd-streamlit-app.fly.dev/

---

## 📦 Project Architecture
```
User (Browser)
      │
      ▼
┌──────────────────────────────┐
│       Streamlit App          │
│ (Interactive Health UI)      │
└──────────────▲───────────────┘
               │  HTTPS POST
               ▼
┌──────────────────────────────┐
│            FastAPI           │
│ (Loads Model + Vectorizer)   │
└──────────────▲───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     ML Model (XGBoost)       │
│     DictVectorizer Pipeline  │
└──────────────────────────────┘
```

---

## 🧠 Machine Learning Model

Dataset:  
📊 **70,000 patient records** from Kaggle  
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

### Model Comparison

| Model | AUC‑ROC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.78 | 0.72 |
| Random Forest | 0.79 | 0.727 |
| **XGBoost (Final)** | **0.801** | **0.731** |

The final model is chosen for its balance of accuracy, interpretability, and efficiency.

---

## 🩻 Required Input Features

| Feature | Description |
|---------|-------------|
| age_years | Age of the patient in years |
| gender | 1 = female, 2 = male |
| height | Height in centimeters |
| weight | Weight in kilograms |
| ap_hi | Systolic blood pressure |
| ap_lo | Diastolic blood pressure |
| cholesterol | Levels: 1, 2, or 3 |
| gluc | Glucose levels: 1, 2, or 3 |
| smoke | 0 or 1 |
| alco | 0 or 1 |
| active | 0 or 1 |

---

## ⚙️ Backend API (FastAPI)

### Live Endpoint
👉 https://cvd-detector-main-v2.fly.dev/predict

### Example Request
```json
{
  "age_years": 52,
  "gender": 2,
  "height": 170,
  "weight": 80,
  "ap_hi": 130,
  "ap_lo": 80,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

### Example Response
```json
{
  "cardio_prediction": 1,
  "cardio_probability": 0.73
}
```

---

## 🖥️ Streamlit App

### Live Frontend
👉 https://cvd-streamlit-app.fly.dev/

### Features
- Clean and modern UI  
- Real‑time prediction from the API  
- Color‑coded risk indicators  
- Probability insights  
- Uses your local or remote API endpoint  

---

## 🚀 Deployment

### FastAPI Backend
```
fly launch
fly deploy
```

### Streamlit Frontend
```
fly launch --no-db
fly deploy
```

---

## 📁 Project Structure
```
CVD_detector_main/
│── api/                # FastAPI backend
│── streamlit_app/      # Streamlit UI
│── scripts/            # Model + DV artifacts
│── requirements.txt
│── Dockerfile
│── fly.toml
│── README.md
```

---

## 🐳 Docker Support  
Build:
```
docker build -t cvd-detector .
```

Run:
```
docker run -p 9696:9696 cvd-detector
```

---

## 🔮 Future Improvements
- Add SHAP model explainability  
- Multi‑patient batch predictions  
- Authentication (JWT / API keys)  
- Database logging for analytics  
- Mobile‑optimized user interface  

---

## 📫 Contact
Built by **Jack Jimenez** for ML engineering practice and portfolio development.  
For support or collaboration, reach out anytime!
