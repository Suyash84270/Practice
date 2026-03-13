from langchain_community.document_loaders import PyPDFLoader
from src.logger.logger import get_logger

logger = get_logger(__name__)


def load_document(file_path: str):
    """
    Load PDF document for RAG knowledge base.

    Args:
        file_path: Path to PDF file

    Returns:
        List of LangChain documents
    """

    logger.info(f"Loading PDF: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} pages")

    return documents