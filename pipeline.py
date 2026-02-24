"""Script to store SageMaker Pipeline definitions and ad-hoc execute pipeline"""

import argparse

from sagemaker import Session, get_execution_role
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, TrainingStep, ProcessingStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.processing import ScriptProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline_context import PipelineSession


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_image_uri",
        help="URL for base docker image",
        default="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.0-cpu-py3",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--start_pipeline",
        help="Rather to start the pipeline",
        action="store_true",
    )
    parser.add_argument(
        "--wait_to_complete",
        help="Wait for the pipeline to complete execution",
        action="store_true",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        help="MLflow tracking URI, must have if --start_pipeline is passed",
        default=None,
        required=False,
        type=str,
    )

    args = parser.parse_args()

    if args.start_pipeline and not args.mlflow_tracking_uri:
        raise ValueError(
            "You must specify --mlflow_tracking_uri if --start_pipeline is enabled."
        )

    return args


def main():
    args = parse_args()
    sagemaker_session = Session()
    pipeline_session = PipelineSession()

    role = get_execution_role()

    # Parametrize pipeline execution
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    f1_score_threshold = ParameterFloat(name="F1ScoreThreshold", default_value=0.9)
    mlflow_tracking_uri = ParameterString(name="MlflowTrackingUri", default_value="")
    mlflow_experiment_name = ParameterString(
        name="MlflowExperimentName", default_value=""
    )
    mlflow_run_name = ParameterString(name="MlflowRunName", default_value="")
    s3_bucket = ParameterString(name="S3Buckets", default_value="phcheng-sagemaker")
    model_tags = ParameterString(name="ModelTags", default_value=None)
    registered_model_name = ParameterString(
        name="RegisteredModelName", default_value="mnist-cnn-classifier"
    )

    execution_id = ExecutionVariables.PIPELINE_EXECUTION_ID

    # Train step
    estimator = PyTorch(
        entry_point="train_model.py",
        source_dir="steps",
        role=role,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        framework_version="2.8",
        py_version="py312",
        requirements_file="requirements.txt",
        hyperparameters={
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
            "model_tags": model_tags,
            "s3_bucket": s3_bucket,
            "execution_id": execution_id,
            "registered_model_name": registered_model_name,
        },
    )

    train_step = TrainingStep(
        name="MNISTTrain",
        estimator=estimator,
        cache_config=CacheConfig(enable_caching=True, expire_after="T1H"),
    )

    # Evaluation step
    mlflow_run_id = JsonGet(
        s3_uri=Join(
            on="/", values=["s3:/", s3_bucket, execution_id, "mlflow", "metadata.json"]
        ),
        step=train_step,
        json_path="run_id",
    )

    eval_processor = ScriptProcessor(
        command=["python3"],
        image_uri=args.base_image_uri,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        role=role,
        sagemaker_session=pipeline_session,
    )

    eval_step = ProcessingStep(
        name="MNISTEval",
        processor=eval_processor,
        code="steps/evaluate_model.py",
        job_arguments=[
            "--mlflow_tracking_uri",
            mlflow_tracking_uri,
            "--mlflow_run_id",
            mlflow_run_id,
        ],
        depends_on=[train_step],
    )

    # Actual pipeline definition
    pipeline = Pipeline(
        name="MNISTMLOpsPipeline",
        parameters=[
            training_instance_count,
            training_instance_type,
            f1_score_threshold,
            mlflow_tracking_uri,
            mlflow_experiment_name,
            mlflow_run_name,
            model_tags,
            registered_model_name,
            s3_bucket,
        ],
        steps=[train_step, eval_step],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(role_arn=role)

    if args.start_pipeline:
        execution = pipeline.start(
            {
                "MlflowTrackingUri": "arn:aws:sagemaker:us-east-1:169446447120:mlflow-tracking-server/tracking-server-bv6blb739mkrzb-3wrmpujh6xe7xj-dev",
                "MlflowExperimentName": "mnist-cnn-demo",
                "ModelTags": "model:SimpleCNN,dataset:MNIST",
            }
        )
        if args.wait_to_complete:
            execution.wait()


if __name__ == "__main__":
    main()
