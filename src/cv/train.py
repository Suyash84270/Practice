import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ultralytics import YOLO
# import torch
# import torchvision.models as models
# from torch import nn, optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException
from src.utils.helper import read_yaml
from src.constants.constants import MAIN_CONFIG_FILE, DATASET_CONFIG_FILE

logger = get_logger(__name__)


class VisionTrainer:
    """
    Generic training engine.

    Currently supported:
        - YOLO object detection

    Future support (commented):
        - ResNet classification
        - EfficientNet classification

    Switch models using config.yaml
    """

    def __init__(self):

        try:
            config = read_yaml(MAIN_CONFIG_FILE)

            model_config = config["model"]
            train_config = config.get("training", {})

            self.model_type = model_config.get("type", "yolo")
            self.model_path = model_config["path"]

            self.epochs = train_config.get("epochs", 50)
            self.img_size = train_config.get("img_size", 640)
            self.batch_size = train_config.get("batch_size", 16)

            logger.info(f"Trainer initialized for model: {self.model_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def train(self):

        try:

            # ---------- YOLO TRAINING ----------
            if self.model_type == "yolo":

                logger.info("Starting YOLO training")

                model = YOLO(self.model_path)

                model.train(
                    data=DATASET_CONFIG_FILE,
                    epochs=self.epochs,
                    imgsz=self.img_size,
                    batch=self.batch_size
                )

                logger.info("YOLO training completed")

            # ---------- FUTURE: RESNET TRAINING ----------
            # elif self.model_type == "classification":
            #
            #     transform = transforms.Compose([
            #         transforms.Resize((224,224)),
            #         transforms.ToTensor()
            #     ])
            #
            #     dataset = datasets.ImageFolder("data/train", transform=transform)
            #     loader = DataLoader(dataset, batch_size=32, shuffle=True)
            #
            #     model = models.resnet50(pretrained=True)
            #     model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
            #
            #     optimizer = optim.Adam(model.parameters(), lr=0.001)
            #     criterion = nn.CrossEntropyLoss()
            #
            #     for epoch in range(10):
            #         for images, labels in loader:
            #             outputs = model(images)
            #             loss = criterion(outputs, labels)
            #
            #             optimizer.zero_grad()
            #             loss.backward()
            #             optimizer.step()
            #
            #     torch.save(model.state_dict(), "models/resnet_model.pth")

            else:
                raise ValueError("Unsupported model type")

        except Exception as e:
            logger.error("Training failed")
            raise CustomException(e, sys)


def run_training():
    try:
        trainer = VisionTrainer()
        trainer.train()

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training()