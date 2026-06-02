FROM python:3.11-slim

# System deps required by OpenCV and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/ML/requirements.txt services/ML/requirements.txt
RUN pip install --retries 5 --timeout=120 --no-cache-dir -r services/ML/requirements.txt

COPY shared/ shared/
COPY services/ML/ services/ML/
# backend package needed for shared DB helpers imported by ML utilities
COPY services/backend/ services/backend/

EXPOSE 8000
CMD ["uvicorn", "services.ML.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
