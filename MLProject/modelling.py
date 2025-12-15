import argparse
import json
import os
import tempfile
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.data_path)
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

run_id = os.environ.get("MLFLOW_RUN_ID")

mlflow.log_metric("accuracy", acc, run_id=run_id)

with tempfile.TemporaryDirectory() as tmp:
    cm_path = f"{tmp}/training_confusion_matrix.png"
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, run_id=run_id)

    metric_path = f"{tmp}/metric_info.json"
    with open(metric_path, "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)
    mlflow.log_artifact(metric_path, run_id=run_id)

mlflow.sklearn.log_model(model, artifact_path="model")