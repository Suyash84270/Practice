from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger.logger import get_logger

logger = get_logger(__name__)


def split_documents(documents):
    """
    Split loaded documents into smaller chunks for embedding.
    This improves retrieval accuracy in RAG systems.
    """

    logger.info("Splitting documents into chunks")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"Generated {len(chunks)} chunks")

    return chunks