import torch
import torch.nn.functional as F
import os
import argparse
import logging
import mlflow
import boto3
import json

from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from lightning import LightningModule
from lightning.pytorch import Trainer, Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.fabric.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import List
from uuid import uuid4


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.cnn(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.cnn(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MLFlowSystemMonitorCallback(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            raise MisconfigurationException(
                "MLFlowSystemMonitorCallback requires MLFlowLogger"
            )

        self.system_monitor = SystemMetricsMonitor(
            run_id=trainer.logger.run_id,
        )
        self.system_monitor.start()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.system_monitor.finish()


class MLFlowRegisterModelCallback(Callback):
    def __init__(
        self,
        model_name: str = "mnist-cnn-model",
        registered_model_name: str = None,
        await_registration: bool = False,
    ):
        """
        Callback to register the trained model to MLflow Model Registry.

        Args:
            model_name: Name of the model artifact to log
            registered_model_name: Name to register the model under in the registry.
                                   If None, uses model_name
            await_registration: Whether to wait for the registration to complete
        """
        super().__init__()
        self.model_name = model_name
        self.registered_model_name = registered_model_name or model_name
        self.await_registration = await_registration
        self._model_uri = None

    @property
    def model_uri(self):
        return self._model_uri

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Register model to MLflow Model Registry after training completes."""
        if not isinstance(trainer.logger, MLFlowLogger):
            logger.warning(
                "MLFlowRegisterModelCallback requires MLFlowLogger. Skipping model registration."
            )
            return

        try:
            mlflow_logger = trainer.logger
            run_id = mlflow_logger.run_id

            # Log the PyTorch Lightning model
            with mlflow.start_run(run_id=run_id):
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=pl_module,
                    artifact_path=self.model_name,
                    registered_model_name=self.registered_model_name,
                )
                self._model_uri = model_info.model_uri
                logger.info(
                    "Successfully registered model %s", self.registered_model_name
                )

        except Exception as e:
            logger.error(f"Failed to register model to MLflow: {str(e)}")
            raise


def store_mlflow_metadata_in_s3(
    tracking_uri: str,
    run_id: str,
    run_name: str,
    model_uri: str,
    tags: List[str],
    bucket: str,
    execution_id: str,
) -> None:
    s3_cli = boto3.client("s3")

    # get mlflow metadata
    mlflow_metadata = {
        "tracking_uri": tracking_uri,
        "run_name": run_name,
        "run_id": run_id,
        "model_uri": model_uri,
        "tags": tags,
    }

    # S3 antics
    object_key = f"{execution_id}/mlflow/metadata.json"
    s3_cli.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=json.dumps(mlflow_metadata),
        ContentType="application/json",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--epoch", type=int, required=False, default=5)
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    parser.add_argument("--mlflow_tracking_uri", type=str, required=False, default=None)
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        required=False,
        default="MNISTCNNPytorchTraining",
    )
    parser.add_argument("--mlflow_run_name", type=str, required=False, default=None)
    parser.add_argument("--mlflow_run_id", type=str, required=False, default=None)
    parser.add_argument(
        "--model_tags",
        type=str,
        help="A comma-delimited list of tags to give the logged model for efficient model discovery in MlFlow",
        required=False,
        default="",
    )
    parser.add_argument(
        "--execution_id",
        type=str,
        help="Pipeline execution ID, used to group and identify training artifiacts stored in S3",
        required=False,
        default=str(uuid4()),
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        help="S3 bucket to store the training artifacts",
        required=False,
        default="phcheng-sagemaker",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--registered_model_name",
        type=str,
        required=False,
        default="mnist-cnn-classifier",
        help="Name to register the model under in MLflow Model Registry",
    )
    args = parser.parse_args()
    return args


def parse_model_tags(tags_string: str) -> dict:
    if not tags_string:
        return {}
    pairs = tags_string.split(",")
    tag_dict = {}
    for pair in pairs:
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid tag format: '{pair}'. Expected format: 'key:value'. "
                f"Tags should be comma-separated key:value pairs, e.g., 'env:prod,version:1.0'"
            )
        tag_dict[parts[0].strip()] = parts[1].strip()
    logger.info("Parsed model_tags: %s", tag_dict)
    return tag_dict


def main():
    args = parse_args()
    logger.info("Training script started with args %s", args)

    dataset = MNIST(
        os.getcwd(),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        ),
    )
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Enable system metric logging in mlflow
    mlflow_tracking_uri = args.mlflow_tracking_uri
    logger.info("Setting up Mlflow with tracking uri %s", mlflow_tracking_uri)
    mlflow_logger = MLFlowLogger(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=mlflow_tracking_uri,
        run_name=args.mlflow_run_name if args.mlflow_run_name else None,
        run_id=args.mlflow_run_id if args.mlflow_run_id else None,
        tags=parse_model_tags(args.model_tags),
        log_model="all",
    )

    # Set up callbacks
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    sys_metric_monitor_callback = MLFlowSystemMonitorCallback()
    model_register_callback = MLFlowRegisterModelCallback(
        model_name="mnist-cnn-model",
        registered_model_name=args.registered_model_name,
    )

    callbacks = [
        lr_monitor_callback,
        sys_metric_monitor_callback,
        model_register_callback,
    ]

    # Log additional training parameters
    model = MNISTModel(lr=args.lr)
    trainer = Trainer(
        max_epochs=args.epoch,
        logger=mlflow_logger,
        callbacks=callbacks,
    )
    logger.info("Start training...")
    logger.info("Checkpoints will be saved to: %s ", args.checkpoint_dir)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    logger.info("Training complete")

    # Save model to SageMaker model output directory
    model_output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_output_dir, exist_ok=True)

    # Save the PyTorch Lightning checkpoint
    checkpoint_path = os.path.join(model_output_dir, "model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")

    logger.info("Storing mlflow metadata in S3")
    run_id = mlflow_logger.run_id
    store_mlflow_metadata_in_s3(
        tracking_uri=args.mlflow_tracking_uri,
        model_uri=model_register_callback.model_uri,
        run_id=run_id,
        run_name=mlflow_logger.experiment.get_run(run_id).info.run_name,
        tags=args.model_tags,
        bucket=args.s3_bucket,
        execution_id=args.execution_id,
    )


if __name__ == "__main__":
    main()
