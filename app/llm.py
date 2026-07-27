from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel
from app.config import settings


def get_llm() -> BaseChatModel:
    """Factory function to construct and return the configured Chat Model.

    Enforces temperature=0.0 for deterministic, strictly grounded responses.
    Supports OpenAI and Google Gemini models.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        api_key = settings.OPENAI_API_KEY
        if api_key and api_key != "your_openai_api_key_here":
            logger.info(f"Initializing OpenAI Chat Model: {settings.OPENAI_MODEL}")
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.0,
                openai_api_key=api_key,
            )
        else:
            logger.warning("OpenAI API key missing or default placeholder.")

    elif provider == "gemini":
        api_key = settings.GOOGLE_API_KEY
        if api_key and api_key != "your_gemini_api_key_here":
            logger.info(f"Initializing Gemini Chat Model: {settings.GEMINI_MODEL}")
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                temperature=0.0,
                google_api_key=api_key,
            )
        else:
            logger.warning("Google Gemini API key missing or default placeholder.")

    # Fallback chat model interface for testing without external API key
    logger.info("Initializing offline fallback chat model interface.")
    from langchain_community.chat_models import FakeListChatModel

    return FakeListChatModel(
        responses=[
            "Agentic AI refers to autonomous systems driven by LLMs capable of perceiving environments, formulating goals, and executing multi-step reasoning.",
            "I couldn't find this information in the provided Agentic AI eBook.",
        ]
    )
