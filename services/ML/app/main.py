from fastapi import FastAPI
from app.routes import api

app = FastAPI(title="ML Service")

app.include_router(api.router)


@app.get("/")
def root():
    return {"message": "ML service is running"}