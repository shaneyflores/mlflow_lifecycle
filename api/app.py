from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import time
import mlflow


RUN_ID = f"best_model"
MODEL_URI = f"{RUN_ID}/model"

app = FastAPI(title="CancerClassificationAPI")
model = mlflow.pyfunc.load_model(MODEL_URI)

class Features(BaseModel):
    data: list[list[float]]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: Features):
    mlflow.set_experiment("mlflow_monitoring")
    
    df = pd.DataFrame(features.data)
    start = time.time()
    preds = model.predict(df)
    latency = time.time() - start

    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.log_metric("latency_ms", latency * 1000)
        mlflow.log_param("batch_size", len(df))

    return {"predictions": preds.tolist(), "latency_ms": latency * 1000}


