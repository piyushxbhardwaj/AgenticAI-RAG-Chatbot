from loguru import logger
from langchain_core.embeddings import Embeddings
from app.config import settings


def get_embedding_model() -> Embeddings:
    """Factory function to construct and return the configured Embedding model.

    Supports OpenAI and Google Gemini embeddings based on configuration.
    Falls back gracefully if specific keys are absent during local testing.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        api_key = settings.OPENAI_API_KEY
        if api_key and api_key != "your_openai_api_key_here":
            logger.info(f"Initializing OpenAI Embeddings model: {settings.OPENAI_EMBEDDING_MODEL}")
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                openai_api_key=api_key,
            )
        else:
            logger.warning("OpenAI API key not set or invalid.")

    elif provider == "gemini":
        api_key = settings.GOOGLE_API_KEY
        if api_key and api_key != "your_gemini_api_key_here":
            logger.info(f"Initializing Gemini Embeddings model: {settings.GEMINI_EMBEDDING_MODEL}")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=settings.GEMINI_EMBEDDING_MODEL,
                google_api_key=api_key,
            )
        else:
            logger.warning("Google Gemini API key not set or invalid.")

    # Fallback to local HuggingFace embeddings for seamless zero-key local testing
    logger.info("Initializing fallback local HuggingFace Embeddings (all-MiniLM-L6-v2)")
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
