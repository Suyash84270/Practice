import os
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ultralytics import YOLO
# import torch
# import torchvision.models as models
# from torchvision import transforms
# from PIL import Image

from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException
from src.utils.helper import read_yaml
from src.constants.constants import MAIN_CONFIG_FILE


logger = get_logger(__name__)


class VisionModel:
    """
    Generic vision inference engine.

    Currently used:
        - YOLO (object detection)

    Future supported models:
        - ResNet (classification)
        - EfficientNet (classification)

    To switch models, change config.yaml:

    model:
        type: "yolo"            # yolo | classification
        path: "yolov8m.pt"
    """

    def __init__(self):

        try:
            config = read_yaml(MAIN_CONFIG_FILE)["model"]

            self.model_type = config.get("type", "yolo")
            self.model_path = config["path"]
            self.conf = config.get("confidence", 0.4)
            self.iou = config.get("iou", 0.5)

            logger.info(f"Loading model: {self.model_path}")

            # ---------- YOLO DETECTION (CURRENTLY USED) ----------
            if self.model_type == "yolo":
                self.model = YOLO(self.model_path)

            # ---------- FUTURE: RESNET / EFFICIENTNET ----------
            # elif self.model_type == "classification":
            #
            #     # Example using pretrained ResNet
            #     self.model = models.resnet50(pretrained=True)
            #     self.model.eval()
            #
            #     self.transform = transforms.Compose([
            #         transforms.Resize((224, 224)),
            #         transforms.ToTensor()
            #     ])

            else:
                raise ValueError("Unsupported model type")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, image_path: str):

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(image_path)

            # ---------- YOLO DETECTION ----------
            if self.model_type == "yolo":

                results = self.model.predict(
                    source=image_path,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False
                )[0]

                return self._count_detections(results)

            # ---------- FUTURE: CLASSIFICATION ----------
            # elif self.model_type == "classification":
            #
            #     image = Image.open(image_path)
            #     image = self.transform(image).unsqueeze(0)
            #
            #     with torch.no_grad():
            #         output = self.model(image)
            #
            #     predicted_class = output.argmax().item()
            #
            #     return {"class": predicted_class}

        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)

    def _count_detections(self, result) -> Dict[str, int]:

        counts = {}

        if result.boxes is None:
            return counts

        for cls in result.boxes.cls.tolist():
            label = result.names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

        logger.info(f"Detection completed: {counts}")

        return counts


def run_detection(image_path: str):
    """Pipeline helper"""

    try:
        return VisionModel().predict(image_path)

    except Exception as e:
        raise CustomException(e, sys)