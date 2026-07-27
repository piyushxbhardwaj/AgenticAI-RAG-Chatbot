import sys
from pathlib import Path
from unittest.mock import patch

# Ensure root project directory is in sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.rag import NOT_FOUND_RESPONSE

client = TestClient(app)


def test_health_check():
    """Tests GET /health endpoint status and diagnostic response format."""
    with patch("app.api.vector_manager.get_indexed_count", return_value=12):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "vector_store" in data
        assert "llm_provider" in data
        assert "documents_indexed" in data
        assert data["documents_indexed"] == 12


def test_chat_endpoint_empty_question():
    """Tests POST /chat endpoint validation error handling for empty payload."""
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_chat_endpoint_in_domain_question():
    """Tests POST /chat endpoint with a valid in-domain eBook question using mocked RAG pipeline."""
    mock_response = {
        "answer": "ReAct (Reasoning and Acting) is an architectural framework where an AI agent alternates between explicit reasoning traces and action steps.",
        "context": [
            "Chapter 2: The ReAct Framework (Reasoning and Acting)\nReAct is a fundamental paradigm..."
        ],
        "confidence": 0.91,
    }

    with patch("app.api.run_rag_pipeline", return_value=mock_response) as mock_pipeline:
        response = client.post("/chat", json={"question": "What is ReAct?"})

        # 1. Verify status code
        assert response.status_code == 200

        # 2. Verify response schema keys
        data = response.json()
        assert "answer" in data
        assert "context" in data
        assert "confidence" in data

        # 3. Verify response field values and types
        assert data["answer"] == mock_response["answer"]
        assert data["context"] == mock_response["context"]
        assert data["confidence"] == 0.91
        assert isinstance(data["answer"], str)
        assert isinstance(data["context"], list)
        assert isinstance(data["confidence"], float)

        # 4. Verify pipeline invocation
        mock_pipeline.assert_called_once_with("What is ReAct?")


def test_chat_endpoint_out_of_domain_question():
    """Tests POST /chat endpoint anti-hallucination guardrail on an irrelevant query using mocked RAG pipeline."""
    mock_response = {
        "answer": NOT_FOUND_RESPONSE,
        "context": [],
        "confidence": 0.0,
    }

    with patch("app.api.run_rag_pipeline", return_value=mock_response) as mock_pipeline:
        response = client.post("/chat", json={"question": "What is the capital of France?"})

        # 1. Verify status code
        assert response.status_code == 200

        # 2. Verify response schema keys
        data = response.json()
        assert "answer" in data
        assert "context" in data
        assert "confidence" in data

        # 3. Verify response field values and types
        assert data["answer"] == NOT_FOUND_RESPONSE
        assert data["context"] == []
        assert data["confidence"] == 0.0
        assert isinstance(data["answer"], str)
        assert isinstance(data["context"], list)
        assert isinstance(data["confidence"], float)

        # 4. Verify pipeline invocation
        mock_pipeline.assert_called_once_with("What is the capital of France?")
