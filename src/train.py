import mlflow
import os
import shutils
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train(data_path="data/breast_cancer.csv", n_estimators=100, max_depth=5):
    mlflow.set_experiment("mlflow_lifecycle")

    with mlflow.start_run(run_name="model_training"):
        df = pd.read_csv(data_path)
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        with mlflow.start_run(nested=True):
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds_proba)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(model, name="model")

        print(f"Model trained, AUC={auc:.4f}")

if __name__ == "__main__":
    train()
