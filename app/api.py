import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retrieval import VectorStoreManager
from app.graph import run_rag_pipeline

# Initialize Global Vector Store Manager instance
vector_manager = VectorStoreManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup validation and lifecycle management."""
    logger.info("Initializing AgenticAI-RAG-Chatbot API server...")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Vector Store Type: {settings.VECTOR_STORE_TYPE}")

    # Validate vector store index status on startup
    indexed_count = vector_manager.get_indexed_count()
    if indexed_count == 0:
        logger.warning(
            "⚠️ Vector store is currently empty! "
            "Please run 'python app/ingest.py' or 'make ingest' to populate the vector store."
        )
    else:
        logger.info(f"✅ Vector store is online with {indexed_count} indexed chunks.")

    yield
    logger.info("Shutting down AgenticAI-RAG-Chatbot API server...")


# Create FastAPI App Instance
app = FastAPI(
    title="AgenticAI-RAG-Chatbot API",
    description="Production-ready RAG chatbot API powered by LangGraph, strictly constrained to the Agentic AI eBook.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns system health, active configurations, and vector index count."""
    count = vector_manager.get_indexed_count()
    return HealthResponse(
        status="healthy",
        vector_store=settings.VECTOR_STORE_TYPE,
        llm_provider=settings.LLM_PROVIDER,
        documents_indexed=count,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG Chatbot"],
)
async def chat_endpoint(request: ChatRequest):
    """Processes user questions strictly against the Agentic AI eBook using LangGraph RAG workflow."""
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty.",
        )

    logger.info(f"Received chat question: '{request.question}'")

    try:
        result = await asyncio.to_thread(run_rag_pipeline, request.question.strip())
        logger.info(f"Successfully processed query. Answer length: {len(result['answer'])}, Confidence: {result['confidence']}")

        return ChatResponse(
            answer=result["answer"],
            context=result["context"],
            confidence=result["confidence"],
        )
    except Exception as e:
        logger.error(f"Error executing RAG pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error processing question: {str(e)}",
        )
