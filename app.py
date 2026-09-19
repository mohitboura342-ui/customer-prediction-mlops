from fastapi import FastAPI

app = FastAPI(title="ML Prediction API")


@app.get("/")
def home():
    return {
        "message": "ML API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }