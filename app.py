import os
import sys
import streamlit as st
from PIL import Image
import tempfile

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.inspection_pipeline import run_inspection
from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException

logger = get_logger(__name__)


def main():
    """
    Streamlit UI for AI Construction Site Inspector.

    Steps:
    1. Upload construction site image
    2. Display image
    3. Run inspection pipeline on button click
    4. Show detection results and AI report
    """

    try:
        # Page configuration
        st.set_page_config(page_title="AI Construction Site Inspector", page_icon="🏗️")

        st.title("🏗️ AI Construction Site Inspector")
        st.write("Upload a construction site image to analyze safety conditions.")

        # Image upload
        uploaded_file = st.file_uploader(
            "Upload Construction Site Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Run pipeline only when button pressed
            if st.button("Run Inspection"):

                logger.info("Inspection started from Streamlit UI")

                # Save uploaded image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    image_path = tmp.name

                # Run inspection pipeline
                result = run_inspection(image_path)

                # Inspection details section
                st.subheader("📋 Inspection Details")

                if isinstance(result, dict) and "detections" in result:

                    detections = result["detections"]

                    for obj, count in detections.items():
                        st.write(f"**{obj}** : {count}")

                # Detection results
                st.subheader("🔎 Detection Results")

                if isinstance(result, dict) and "detections" in result:
                    st.json(result["detections"])

                # AI report
                st.subheader("📝 AI Site Inspection Report")

                if isinstance(result, dict) and "report" in result:
                    st.write(result["report"])
                else:
                    st.write(result)
                logger.info("Inspection completed")

    except Exception as e:
        logger.error("Streamlit app failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()