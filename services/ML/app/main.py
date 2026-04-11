from fastapi import FastAPI
from app.routes import inference

app = FastAPI(title="ML Service")

app.include_router(inference.router)


@app.get("/")
def root():
    return {"message": "ML service is running"}