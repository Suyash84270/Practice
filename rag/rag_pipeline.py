from loader import load_document
from text_splitter import split_documents
from embedder import load_embedding_model
from vector_store import create_vector_store

from src.logger.logger import get_logger

logger = get_logger(__name__)


def build_rag_pipeline(file_path: str):
    """
    Build RAG pipeline.

    Steps:
    1. Load documents
    2. Split into chunks
    3. Create embeddings
    4. Store in FAISS vector database
    """

    logger.info("Starting RAG pipeline")

    documents = load_document(file_path)

    chunks = split_documents(documents)

    embeddings = load_embedding_model()

    vector_store = create_vector_store(chunks, embeddings)

    logger.info("RAG pipeline ready")

    return vector_store