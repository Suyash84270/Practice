import os
import yaml
from typing import Dict, List, Tuple

# ---------------------------------------------------------
# OPTIONAL DATASET DOWNLOAD (FOR FUTURE USE)
# ---------------------------------------------------------
# If your dataset is hosted online (Google Drive / Kaggle / URL),
# you can paste the dataset link below and download automatically.
# Example implementation can be added here later.
#
# DATASET_URL = "PASTE_DATASET_LINK_HERE"
#
# def download_dataset(url: str, download_path: str = "data"):
#     """
#     Placeholder function for future dataset downloading.
#     You can implement downloading from Kaggle, GDrive, etc.
#     """
#     pass
# ---------------------------------------------------------


class DataLoader:
    """
    Reusable dataset loader for Computer Vision projects.

    Supports:
    - YOLO style datasets
    - Train / Validation image loading
    - Label file association
    - Reusable across multiple CV projects
    """

    def __init__(self, dataset_config_path: str = "configs/dataset.yaml"):
        self.dataset_config_path = dataset_config_path
        self.config = self._load_config()

        self.train_images = self.config["train"]["images"]
        self.train_labels = self.config["train"]["labels"]
        self.val_images = self.config["val"]["images"]
        self.val_labels = self.config["val"]["labels"]

    def _load_config(self) -> Dict:
        """Load dataset configuration YAML."""
        if not os.path.exists(self.dataset_config_path):
            raise FileNotFoundError(
                f"Dataset config file not found: {self.dataset_config_path}"
            )

        with open(self.dataset_config_path, "r") as file:
            return yaml.safe_load(file)

    def _get_image_files(self, folder_path: str) -> List[str]:
        """Return all image file paths."""
        valid_extensions = (".jpg", ".jpeg", ".png")

        image_files = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(valid_extensions)
        ]

        return sorted(image_files)

    def _get_label_path(self, image_path: str, label_folder: str) -> str:
        """Map image file to its corresponding label file."""
        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        return os.path.join(label_folder, label_name)

    def load_train_data(self) -> List[Tuple[str, str]]:
        """
        Returns list of tuples:
        (image_path, label_path)
        """
        images = self._get_image_files(self.train_images)

        dataset = [
            (img_path, self._get_label_path(img_path, self.train_labels))
            for img_path in images
        ]

        return dataset

    def load_validation_data(self) -> List[Tuple[str, str]]:
        """
        Returns validation dataset:
        (image_path, label_path)
        """
        images = self._get_image_files(self.val_images)

        dataset = [
            (img_path, self._get_label_path(img_path, self.val_labels))
            for img_path in images
        ]

        return dataset

    def dataset_summary(self) -> Dict[str, int]:
        """Return basic dataset statistics."""
        train_count = len(self._get_image_files(self.train_images))
        val_count = len(self._get_image_files(self.val_images))

        return {
            "train_images": train_count,
            "validation_images": val_count,
            "total_images": train_count + val_count,
        }