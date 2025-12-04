import mlflow
import mlflow.sklearn
import os

def select_best_model(experiment_name, metric_name, greater_is_better=True):
    # Set the experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")

    # Fetch all runs in the experiment
    runs = client.search_runs([experiment.experiment_id])
    
    if not runs:
        raise ValueError("No runs found for the experiment!")

    # Select the best model based on the specified metric
    best_run = None
    best_value = None

    for run in runs:
        current_metric_value = run.data.metrics.get(metric_name)
        
        if current_metric_value is None:
            continue

        if (best_value is None or 
            (greater_is_better and current_metric_value > best_value) or
            (not greater_is_better and current_metric_value < best_value)):
            best_value = current_metric_value
            best_run = run

    if best_run is None:
        raise ValueError(f"No valid run found with metric '{metric_name}'!")

    # Print the best run details
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best run {metric_name}: {best_value}")

    # Download the best model and save it to the 'best_model' directory
    model_path = f"best_model"
    best_model_run_id = best_run.info.run_id
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Saving the best model to the specified folder
    mlflow.artifacts.download_artifacts(f"runs:/{best_model_run_id}/model", dst_path=model_path)

    print(f"Best model saved to: {os.path.abspath(model_path)}")

    # Register the model for later loading in FastAPI
    mlflow.register_model(f"runs:/{best_model_run_id}/model", "BestModel")

if __name__ == "__main__":
    experiment_name = "mlflow_lifecycle"  
    metric_name = "auc"  

    try:
        select_best_model(experiment_name, metric_name)
    except ValueError as e:
        print(e)