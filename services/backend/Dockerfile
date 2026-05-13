FROM python:3.11-slim

WORKDIR /app

COPY services/backend/requirements.txt services/backend/requirements.txt
COPY services/ML/requirements.txt services/ML/requirements.txt
RUN pip install --no-cache-dir -r services/backend/requirements.txt -r services/ML/requirements.txt

COPY shared/ shared/
COPY services/backend/ services/backend/
COPY services/ML/ services/ML/

EXPOSE 8000
CMD ["uvicorn", "services.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
