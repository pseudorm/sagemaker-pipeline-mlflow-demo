# SageMaker ML Pipeline

An end-to-end machine learning pipeline built with AWS SageMaker for training, evaluating, and deploying a PyTorch CNN model on the MNIST dataset. The pipeline integrates MLflow for experiment tracking and model registry.

## Overview

This project implements a production-ready MLOps pipeline that:
- Trains a Convolutional Neural Network (CNN) on MNIST digit classification
- Tracks experiments and metrics using MLflow
- Evaluates model performance
- Registers models to MLflow Model Registry

## Project Structure

```
sagemaker/
├── pipeline.py                     # Main SageMaker pipeline definition
├── Dockerfile                      # Multi-stage Docker build for SageMaker
├── .dockerignore                   # Docker ignore patterns
├── steps/                          # Pipeline step implementations
│   ├── train_model.py              # Training script with Lightning
│   ├── evaluate_model.py           # Model evaluation script
│   ├── inference.py                # SageMaker inference handler
│   └── deploy_model.py             # Model deployment script
└── model/                          # Model definitions and utilities
```

## Architecture

The pipeline consists of the following steps:

1. **Training Step**: Trains a SimpleCNN model using PyTorch Lightning
   - Logs metrics and artifacts to MLflow
   - Supports distributed training
   - Implements caching for faster iterations

2. **Evaluation Step**: Evaluates the trained model
   - Fetches model from MLflow using run ID
   - Computes performance metrics
   - Validates against configurable thresholds

## Prerequisites

- AWS Account with SageMaker access
- AWS CLI configured with appropriate credentials
- Python 3.12

## Setup

### 1. Configure IAM Roles

Apply the necessary IAM policies for SageMaker execution:

```bash
# Apply custom policy
aws iam put-role-policy --role-name <your-sagemaker-role> \
  --policy-name SageMakerCustomPolicy \
  --policy-document file://custom-policy.json

# Apply MLflow trust policy (if using MLflow)
aws iam create-role --role-name MLflowRole \
  --assume-role-policy-document file://mlflow-trust-policy.json
```

### 2. Build Docker Image (Optional)

If using custom containers:

```bash
docker build -t sagemaker-mnist-pipeline .
```

### 3. Configure Pipeline Parameters

Edit the parameters in [pipeline.py](pipeline.py):

```python
training_instance_type = "ml.m5.xlarge"
f1_score_threshold = 0.9
mlflow_tracking_uri = "<your-mlflow-uri>"
s3_bucket = "<your-s3-bucket>"
```

## Usage

### Running the Pipeline

Execute the pipeline using the Python script:

```bash
python pipeline.py
```

### Customizing Pipeline Execution

You can override parameters when starting the pipeline:

```python
execution = pipeline.start({
    "MlflowTrackingUri": "arn:aws:sagemaker:us-east-1:xxx:mlflow-tracking-server/xxx",
    "MlflowExperimentName": "mnist-cnn-demo",
    "TrainingInstanceType": "ml.m5.2xlarge",
    "F1ScoreThreshold": 0.85,
    "ModelTags": "model:SimpleCNN,dataset:MNIST"
})
```

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TrainingInstanceCount` | Integer | 1 | Number of training instances |
| `TrainingInstanceType` | String | ml.m5.xlarge | EC2 instance type for training |
| `F1ScoreThreshold` | Float | 0.9 | Minimum F1 score for model approval |
| `MlflowTrackingUri` | String | "" | MLflow tracking server URI |
| `MlflowExperimentName` | String | "" | MLflow experiment name |
| `MlflowRunName` | String | "" | MLflow run name |
| `S3Buckets` | String | phcheng-sagemaker | S3 bucket for artifacts |
| `ModelTags` | String | None | Comma-separated model tags |
| `RegisteredModelName` | String | mnist-cnn-classifier | Model name in registry |