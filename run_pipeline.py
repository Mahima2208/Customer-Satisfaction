
import sys 
import os

other_code_directory = os.path.join(os.path.dirname(os.path.abspath("/Users/mahimaphalkey/Downloads/customer-satisfaction-mlops-main/venv/lib/python3.9/site-packages/zenml")))
sys.path.append(other_code_directory)
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

other_code_directory_1 = os.path.join(os.path.dirname(os.path.abspath("/Users/mahimaphalkey/Downloads/customer-satisfaction-mlops-main/zenml-projects/customer-satisfaction/pipelines")))
sys.path.append(other_code_directory_1)

from pipelines.training_pipeline import train_pipeline

other_code_directory_2 = os.path.join(os.path.dirname(os.path.abspath("/Users/mahimaphalkey/Downloads/customer-satisfaction-mlops-main/zenml-projects/customer-satisfaction/steps")))
sys.path.append(other_code_directory_2)

from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model


if __name__ == "__main__":
    training = train_pipeline(
        ingest_data(),
        clean_data(),
        train_model(),
        evaluation(),
    )

    training.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
