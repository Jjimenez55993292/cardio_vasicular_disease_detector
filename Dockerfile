# Dockerfile

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (optional, but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy prediction service, HTML landing page, and model artifacts
COPY scripts/predict.py .
COPY scripts/index.html .
COPY scripts/cardio_model.joblib .
COPY scripts/cardio_feature_columns.joblib .

EXPOSE 9696

# Use gunicorn to serve the Flask app
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
