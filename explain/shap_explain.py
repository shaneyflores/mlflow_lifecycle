import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import mlflow
import mlflow.sklearn
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import os

RUN_ID = f"best_model"
MODEL_URI = f"{RUN_ID}/model"

def explain():
    mlflow.set_experiment("mlflow_interpretability")

    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = mlflow.sklearn.load_model(MODEL_URI)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        print(f"Error with TreeExplainer: {e}")
        print("Switching to KernelExplainer...")
        
        # KernelExplainer requires a reference dataset, here we use a sample
        sample = X.sample(100, random_state=42)  
        explainer = shap.KernelExplainer(model.predict, sample)
        shap_values = explainer.shap_values(X_test)
    
    print(f"shap_values shape: {shap_values.shape}")
    print(f"X_test shape: {X_test.shape}")

    os.makedirs("explain/artifacts", exist_ok=True)

    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, show=False, plot_type="beeswarm")
    else:
        shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")

    plt.tight_layout()
    path = "explain/artifacts/shap_summary.png"
    plt.savefig(path)
    plt.close()

    with mlflow.start_run(run_name="shap_explanations"):
        mlflow.log_artifact(path)

if __name__ == "__main__":
    explain()
