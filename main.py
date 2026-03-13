import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.inspection_pipeline import run_inspection
from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException

logger = get_logger(__name__)


def main():

    try:
        logger.info("Starting Construction Site Inspector")

        image_path = "test_image.jpg"

        # Run full pipeline
        report = run_inspection(image_path)

        print("\nAI Inspection Report:\n")
        print(report)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()