import requests

url = "https://cvd-detector-main-v2.fly.dev/predict"

patient = {
    "age_years": 52.0,
    "gender": 2,
    "height": 170,
    "weight": 80.0,
    "ap_hi": 130,
    "ap_lo": 80,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
}

response = requests.post(url, json=patient)
print(response.status_code)
print(response.json())
