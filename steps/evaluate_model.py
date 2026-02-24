import torch
import torch.nn.functional as F
import os
import argparse
import logging
import mlflow

from lightning import LightningModule
from lightning.pytorch import Trainer
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channel: int = 1, class_num: int = 10):
        super().__init__()

        # conv, maxpool, activation #1
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=(5, 5)),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.GELU(),
        )
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5)),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.GELU(),
        )
        self.dense_1 = torch.nn.Linear(in_features=256, out_features=class_num)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_1(x)
        return x


class MNISTModel(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.cnn = SimpleCNN(1)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)

        return {"test_loss": loss, "test_accuracy": acc}

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        help="MLflow run name to load the model from",
    )
    parser.add_argument(
        "--run_name_file",
        type=str,
        required=False,
        help="Path to file containing MLflow run ID",
    )
    parser.add_argument(
        "--mlflow_run_id",
        type=str,
        required=False,
        help="MLflow run ID passed from pipeline",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        required=False,
        default="",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--mlflow_properties_path",
        type=str,
        required=False,
        help="Path to MLflow properties JSON file from training step",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()
    return args


def load_model_from_mlflow(run_id: str):
    """
    Load a model from MLflow using run ID.

    Args:
        run_id: MLflow run ID
        model_name: Name of the model artifact (default: "mnist-cnn-model")

    Returns:
        Loaded PyTorch model
    """
    found_models = mlflow.search_registered_models(max_results=10)

    if found_models:
        # subset the model results
        matched_model = None
        for model in found_models:
            matched_model_versions = [
                version for version in model.latest_versions if version.run_id == run_id
            ]
            if matched_model_versions:
                matched_model = matched_model_versions[0]
                break
        if not matched_model:
            raise ValueError(f"No registered model found with in run {run_id}")

        model_uri = matched_model.source
        logger.info("Loading model from URI: %s", model_uri)
        model = mlflow.pytorch.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow")
        return model
    raise ValueError(f"No registered model found with in run {run_id}")


def load_test_dataset(batch_size: int):
    """Load MNIST test dataset with same transforms as training."""
    test_dataset = MNIST(
        os.getcwd(),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        ),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Test dataset loaded: %d samples", len(test_dataset))
    return test_dataloader


def evaluate_model_with_lightning(model, test_dataloader):
    """
    Evaluate model on test dataset using PyTorch Lightning Trainer.

    Args:
        model: PyTorch Lightning model
        test_dataloader: DataLoader with test data

    Returns:
        Dictionary with evaluation metrics
    """
    # Create a Lightning Trainer for testing
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    # Run test
    logger.info("Running evaluation with Lightning Trainer...")
    test_results = trainer.test(model, dataloaders=test_dataloader, verbose=False)

    # Extract metrics from results
    if test_results and len(test_results) > 0:
        metrics = test_results[0]
        return metrics
    else:
        raise RuntimeError("No test results returned from trainer")


def log_metrics_to_mlflow(run_id: str, metrics: dict):
    """Log evaluation metrics to the same MLflow run."""

    with mlflow.start_run(run_id=run_id):
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        logger.info("Metrics logged to MLflow run: %s", run_id)


def main():
    """Main function to orchestrate model evaluation."""
    args = parse_args()

    # Set MLflow tracking URI
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        logger.info("MLflow tracking URI: %s", args.mlflow_tracking_uri)

    # Get run_id: check args first, then property file
    run_id = args.mlflow_run_id
    if not run_id:
        raise ValueError("MLflow run_id is required for evaluation")

    logger.info("Evaluating model from run: %s", run_id)

    # Load model from MLflow
    logger.info("Loading model from MLflow...")
    model = load_model_from_mlflow(run_id)

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataloader = load_test_dataset(args.batch_size)

    # Evaluate model using PyTorch Lightning
    logger.info("Starting evaluation with Lightning Trainer...")
    metrics = evaluate_model_with_lightning(model, test_dataloader)

    # Display results
    logger.info("Evaluation Results: %s", metrics)

    # Log metrics back to MLflow
    formatted_metrics = {}
    for key, value in metrics.items():
        # Convert accuracy to percentage for logging
        if "accuracy" in key:
            formatted_metrics[f"eval_{key}"] = value * 100
        else:
            formatted_metrics[f"eval_{key}"] = value

    log_metrics_to_mlflow(run_id, formatted_metrics)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
