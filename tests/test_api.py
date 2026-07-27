import sys
from pathlib import Path

# Ensure root project directory is in sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.rag import NOT_FOUND_RESPONSE

client = TestClient(app)


def test_health_check():
    """Tests GET /health endpoint status and diagnostic response format."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data
    assert "llm_provider" in data
    assert "documents_indexed" in data
    assert isinstance(data["documents_indexed"], int)


def test_chat_endpoint_empty_question():
    """Tests POST /chat endpoint validation error handling for empty payload."""
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_chat_endpoint_in_domain_question():
    """Tests POST /chat endpoint with a valid in-domain eBook question."""
    response = client.post("/chat", json={"question": "What is ReAct?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "context" in data
    assert "confidence" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["context"], list)
    assert 0.0 <= data["confidence"] <= 1.0


def test_chat_endpoint_out_of_domain_question():
    """Tests POST /chat endpoint anti-hallucination guardrail on an irrelevant query."""
    response = client.post("/chat", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == NOT_FOUND_RESPONSE
    assert data["context"] == []
    assert 0.0 <= data["confidence"] < 0.35
