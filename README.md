# MLOps Lifecycle

Shaney Flores<br>
DATASTUDIES 5750.01

The goal of this project was to gain some familiarity with the CI/CD workflow for a deployed machine learning model using several Python packages, Docker, and GitHub. We build and train a random forest classifier for the Wisconsin Breast Cancer data set, export the best performing model, and containerize it within Docker. 

## Setting up environment and virtual machine

The formal instructions state to use Python 3.11; however, the default Python version for my machine is 3.12, so I will use pyenv to setup another installation of Python to ensure that we don't run into version compatibility issues.

Once Python 3.11 is installed, we will go to our mlops-mlflow project and run the following to set the local Python environment to 3.11

```bash
pyenv local 3.11.14
```

To initialize our virtual machine, we will create it using the local Python 3.11 environment and then activate it.

```bash
pyenv exec python -m venv .venv
source .venv/bin/activate
```

Within the virtual environment, we will install the following packages for our container:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

We will include these packages in our `requirements.txt` file once we have the ML pipeline up and running.

## Implement Data Ingestion with MLflow

Run the following commands with the `--experiment-name mlflow_lifecycle` argument included.

```bash
mlflow run . -e ingest --experiment-name mlflow_lifecycle
```

The result of the command will be a new directory called 'data' with a file named 'breast_cancer.csv' within

## Train a Model and Track Everything with MLflow

```bash
mlflow run . -e train -P n_estimators=200 -P max_depth=6 --experiment-name mlflow_lifecycle
```

The result of the command should be a new directory called mlruns with the different model outputs. We will now want to evaluate our trained model(s) and save the best performing model.

```bash
mlflow run . -e write --experiment-name mlflow_lifecycle
```

A new directory called 'best_model' should appear containing all the artifacts of our best performing model based on the user defined metric (in this case, AUC).

We can next interpret how our model arrived at its prediction using SHAP. 

```bash
mlflow run . -e explain --experiment-name mlflow_interpretability
```

## Deploy the MLflow Model via FastAPI

Using the FastAPI Python package, we can deploy the model into a usable form to test predictions and functionality.

```bash
uvicorn api.app:app --reload --port 8000
```

## Containerize the Model Server with Docker

Now that we have tested our model, we can containerize it within a Docker built with the same speifications of our development environment.

```bash
docker build -t cancer-api -f api/Dockerfile .
docker run -p 8000:8000 cancer-api
```

## Pushing to GitHub

With our project now locally tested and deployed, we can push it to our [GitHub repository](https://github.com/shaneyflores/mlops-mlflow.git) for CI/CD. The following commands initialize the local git repo, add and commit our changes, connect to the remote git repo, and then push our local git to the remote.

```bash
git init -b main
git add .
git commit -m "initial commit"
git remote add origin [url_to_github_repo] 
git push origin main
```


