from typing import List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Input payload for POST /chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": "What is ReAct?"},
        description="User question to be answered using the Agentic AI eBook.",
    )


class ChatResponse(BaseModel):
    """Output payload for POST /chat endpoint."""

    answer: str = Field(
        ...,
        json_schema_extra={"example": "ReAct is a reasoning and acting framework for LLMs..."},
        description="Grounded answer generated from the eBook context.",
    )
    context: List[str] = Field(
        default_factory=list,
        description="List of retrieved context text chunks used to answer the question.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.91},
        description="Normalized similarity confidence score (0.0 to 1.0).",
    )


class HealthResponse(BaseModel):
    """Output payload for GET /health endpoint."""

    status: str = Field(json_schema_extra={"example": "healthy"})
    vector_store: str = Field(json_schema_extra={"example": "chroma"})
    llm_provider: str = Field(json_schema_extra={"example": "openai"})
    documents_indexed: int = Field(json_schema_extra={"example": 318})
