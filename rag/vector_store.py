from langchain_community.vectorstores import FAISS
from src.logger.logger import get_logger

logger = get_logger(__name__)


def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector store from document chunks.
    Used for semantic retrieval in RAG.
    """

    logger.info("Creating FAISS vector store")

    vector_store = FAISS.from_documents(chunks, embeddings)

    logger.info("Vector store created")

    return vector_store