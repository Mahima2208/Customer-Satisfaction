from typing import cast


import sys 
import os

other_code_directory_1 = os.path.join(os.path.dirname(os.path.abspath("venv/lib/python3.9/site-packages/click")))
sys.path.append(other_code_directory_1)

import click
from rich import print


other_code_directory_2 = os.path.join(os.path.dirname(os.path.abspath("/Users/mahimaphalkey/Downloads/customer-satisfaction-mlops-main/zenml-projects/customer-satisfaction/pipelines")))
sys.path.append(other_code_directory_2)

from pipelines.deployment_pipeline import (continuous_deployment_pipeline, deployment_trigger,
    DeploymentTriggerConfig)
from pipelines.deployment_pipeline import  inference_pipeline

other_code_directory_1 = os.path.join(os.path.dirname(os.path.abspath("Users/mahimaphalkey/Downloads/customer-satisfaction-mlops-main/venv/lib/python3.9/site-packages/zenml")))
sys.path.append(other_code_directory_1)

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.ingest_data import ingest_data

from steps.clean_data import clean_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation
import zenml.integrations.mlflow.steps




@click.command()
@click.option(
    "--min-accuracy",
    default=1.8,
    help="Minimum mse required to deploy the model",
)
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(min_accuracy: float, stop_service: bool):

    """Run the mlflow example pipeline"""
    if stop_service:
        service = load_last_service_from_step(
            pipeline_name="continuous_deployment_pipeline",
            step_name="model_deployer",
            running=True,
        )
        if service:
            service.stop(timeout=10)
        return

    deployment = continuous_deployment_pipeline(
        ingest_data=ingest_data,
        clean_data=clean_data,
        model_train=train_model,
        evaluation=evaluation,
        deployment_trigger=deployment_trigger(
            config=DeploymentTriggerConfig(
                min_accuracy=min_accuracy,
            )
        ),
        model_deployer=mlflow_model_deployer_step(   
            params=MLFlowDeploymentService(workers=3)
        ),
    )
    deployment.run(config_path="config.yaml")

    inference = inference_pipeline(
        dynamic_importer=dynamic_importer(),
        prediction_service_loader=prediction_service_loader(
            MLFlowDeploymentLoaderStepParameters(
                pipeline_name="continuous_deployment_pipeline",
                step_name="model_deployer",
            )
        ),
        predictor=predictor(),
    )
    inference.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="model_deployer",
        running=True,
    )

    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon process "
            f"and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_main()
