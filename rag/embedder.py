from langchain_community.embeddings import HuggingFaceEmbeddings
from src.logger.logger import get_logger

logger = get_logger(__name__)


def load_embedding_model():
    """
    Load embedding model for RAG vector generation.
    Used to convert text chunks into embeddings.
    """

    logger.info("Loading embedding model")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model