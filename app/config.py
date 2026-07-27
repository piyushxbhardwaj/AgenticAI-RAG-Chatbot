import os
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM & Embedding Settings
    LLM_PROVIDER: Literal["openai", "gemini"] = Field(
        default="openai",
        description="Primary provider for LLM and embeddings ('openai' or 'gemini')",
    )

    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    # Google Gemini Settings
    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")
    GEMINI_EMBEDDING_MODEL: str = Field(default="models/text-embedding-004")

    # Vector Store Settings
    VECTOR_STORE_TYPE: Literal["chroma", "pinecone"] = Field(
        default="chroma",
        description="Vector database type ('chroma' or 'pinecone')",
    )
    CHROMA_PERSIST_DIRECTORY: str = Field(default="./chroma_db")
    CHROMA_COLLECTION_NAME: str = Field(default="agentic_ai_ebook")

    # Pinecone Settings
    PINECONE_API_KEY: Optional[str] = Field(default=None)
    PINECONE_INDEX_NAME: str = Field(default="agentic-ai-ebook")

    # Document Chunking & Retrieval Parameters
    CHUNK_SIZE: int = Field(default=1000, description="Max character length per chunk")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlapping characters between chunks")
    TOP_K: int = Field(default=4, description="Number of top context chunks to retrieve")
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.35,
        description="Minimum normalized similarity threshold for context relevance",
    )
    PDF_PATH: str = Field(
        default="data/Ebook-Agentic-AI.pdf",
        description="Path to the primary Agentic AI eBook PDF",
    )


# Instantiate global settings object
settings = Settings()
