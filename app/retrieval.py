import os
from typing import List, Tuple
from loguru import logger
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from app.config import settings
from app.embeddings import get_embedding_model


class VectorStoreManager:
    """Manages document vectorization, storage, and retrieval across Chroma DB and Pinecone."""

    def __init__(self, embedding_function: Embeddings = None):
        self.embedding_function = embedding_function or get_embedding_model()
        self.store_type = settings.VECTOR_STORE_TYPE.lower()
        self._vector_store = None
        self._initialize_store()

    def _initialize_store(self):
        """Initializes the vector store backend (Chroma local persistent or Pinecone cloud)."""
        if self.store_type == "pinecone":
            api_key = settings.PINECONE_API_KEY
            if not api_key or api_key == "your_pinecone_api_key_here":
                logger.warning("Pinecone API key missing. Defaulting to local Chroma DB.")
                self.store_type = "chroma"
            else:
                try:
                    from langchain_pinecone import PineconeVectorStore
                    from pinecone import Pinecone

                    pc = Pinecone(api_key=api_key)
                    index_name = settings.PINECONE_INDEX_NAME
                    self._vector_store = PineconeVectorStore(
                        index_name=index_name,
                        embedding=self.embedding_function,
                    )
                    logger.info(f"Initialized Pinecone Vector Store with index: {index_name}")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize Pinecone Vector Store: {e}. Falling back to Chroma DB.")
                    self.store_type = "chroma"

        # Default to local Chroma vector store
        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        self._vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        logger.info(
            f"Initialized Chroma Vector Store at directory '{settings.CHROMA_PERSIST_DIRECTORY}' "
            f"collection '{settings.CHROMA_COLLECTION_NAME}'"
        )

    def add_documents(self, documents: List[Document]) -> int:
        """Indexes a list of LangChain Document chunks into the vector store."""
        if not documents:
            logger.warning("No documents provided for indexing.")
            return 0

        logger.info(f"Indexing {len(documents)} document chunks into {self.store_type} vector store...")
        self._vector_store.add_documents(documents)
        logger.info(f"Successfully indexed {len(documents)} chunks.")
        return len(documents)

    def search_with_scores(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Performs vector similarity search and returns documents with normalized confidence scores [0.0, 1.0]."""
        k = top_k or settings.TOP_K
        results = self._vector_store.similarity_search_with_score(query, k=k)

        normalized_results = []
        for doc, score in results:
            # Normalize vector distance/similarity score into a [0.0, 1.0] confidence score
            # Chroma similarity_search_with_score returns distance (lower is better, typically 0.0 to 1.5+)
            if self.store_type == "chroma":
                # Convert distance to similarity score
                confidence = max(0.0, min(1.0, 1.0 - (score / 2.0)))
            else:
                # Pinecone returns cosine similarity directly (0.0 to 1.0)
                confidence = max(0.0, min(1.0, float(score)))

            normalized_results.append((doc, round(confidence, 4)))

        return normalized_results

    def get_indexed_count(self) -> int:
        """Returns total count of indexed document vectors."""
        try:
            if self.store_type == "chroma" and hasattr(self._vector_store, "_collection"):
                collection = getattr(self._vector_store, "_collection", None)
                if collection and hasattr(collection, "count"):
                    return collection.count()
            elif hasattr(self._vector_store, "index"):
                stats = self._vector_store.index.describe_index_stats()
                return stats.total_vector_count
        except Exception as e:
            logger.warning(f"Error querying vector store document count: {e}")
        return 0

    def is_indexed(self) -> bool:
        """Checks if the vector store contains any indexed documents."""
        return self.get_indexed_count() > 0
