import os
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.cv.detector import run_detection
from src.llm.report_generator import ReportGenerator
from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException


logger = get_logger(__name__)


class ConstructionInspectionPipeline:
    """
    End-to-end AI inspection pipeline.

    Steps:
    1. Run computer vision detection
    2. Apply simple safety logic
    3. Generate AI inspection report using LLM
    """

    def __init__(self):
        try:
            logger.info("Initializing Inspection Pipeline")

            self.report_generator = ReportGenerator()

        except Exception as e:
            raise CustomException(e, sys)

    def analyze_safety(self, detections: Dict[str, int]) -> Dict[str, int]:
        """
        Basic rule-based safety analysis.
        Helps clean detections before sending to LLM.
        """

        try:
            safety_summary = detections.copy()

            person = detections.get("person", 0)
            helmet = detections.get("Hardhat", 0)

            # Estimate workers without helmets
            if person > helmet:
                safety_summary["workers_without_helmet"] = person - helmet
            else:
                safety_summary["workers_without_helmet"] = 0

            return safety_summary

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, image_path: str) -> str:
        """
        Run full inspection pipeline.
        """

        try:
            logger.info("Running construction inspection pipeline")

            # Step 1 — Detect objects
            detections = run_detection(image_path)

            logger.info(f"Detections: {detections}")

            # Step 2 — Safety logic
            safety_data = self.analyze_safety(detections)

            logger.info(f"Safety analysis: {safety_data}")

            # Step 3 — Generate report
            report = self.report_generator.generate_report(safety_data)

            return {
                        "detections": detections,
                        "report": report
                    }

        except Exception as e:
            raise CustomException(e, sys)


def run_inspection(image_path: str) -> str:
    """
    Helper function to run the full pipeline.
    """

    try:
        pipeline = ConstructionInspectionPipeline()
        return pipeline.run(image_path)

    except Exception as e:
        raise CustomException(e, sys)