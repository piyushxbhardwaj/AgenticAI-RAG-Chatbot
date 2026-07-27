import os
import sys
from pathlib import Path

# Ensure root project directory is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.retrieval import VectorStoreManager


def ingest_document(pdf_path: str = None) -> int:
    """Ingests the Agentic AI eBook PDF, splits it into semantic chunks, and indexes it into the vector store.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
    """
    target_path = pdf_path or settings.PDF_PATH

    if not os.path.exists(target_path):
        error_msg = (
            f"Could not find '{target_path}'. Please place the Agentic AI eBook PDF "
            f"file in the 'data/' directory before running document ingestion."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Starting PDF ingestion process for document: '{target_path}'")

    # Load PDF using PyPDFLoader
    try:
        loader = PyPDFLoader(target_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} page(s) from PDF.")
    except Exception as e:
        logger.error(f"Failed to load PDF file '{target_path}': {e}")
        raise

    # Split document into semantic text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} page(s) into {len(chunks)} text chunks "
        f"(chunk_size={settings.CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP})."
    )

    # Initialize VectorStoreManager and index document chunks
    vector_manager = VectorStoreManager()
    indexed_count = vector_manager.add_documents(chunks)

    logger.info(f"Document ingestion complete! Total vectors stored: {indexed_count}")
    return indexed_count


if __name__ == "__main__":
    try:
        count = ingest_document()
        print(f"Ingestion successful! Indexed {count} chunks.")
    except FileNotFoundError as e:
        print(f"\n[Ingestion Error] {e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[Unexpected Error] Ingestion failed: {e}\n", file=sys.stderr)
        sys.exit(1)
