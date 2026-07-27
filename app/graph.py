from typing import List, TypedDict
from loguru import logger
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.retrieval import VectorStoreManager
from app.llm import get_llm
from app.rag import RAG_PROMPT_TEMPLATE, NOT_FOUND_RESPONSE, format_context_documents


class RAGState(TypedDict):
    """Typed dictionary representing the internal execution state of the LangGraph RAG workflow."""

    question: str
    context_chunks: List[str]
    max_confidence: float
    answer: str
    confidence: float


# Initialize persistent vector manager instance
vector_manager = VectorStoreManager()


def retriever_node(state: RAGState) -> dict:
    """Retriever Node: Queries the vector store for relevant context chunks and calculates confidence scores."""
    question = state["question"]
    logger.info(f"[LangGraph Node: Retriever] Querying vector store for: '{question}'")

    results = vector_manager.search_with_scores(question, top_k=settings.TOP_K)

    if not results:
        logger.info("[LangGraph Node: Retriever] No document vectors returned.")
        return {
            "context_chunks": [],
            "max_confidence": 0.0,
        }

    context_chunks = [doc.page_content for doc, _ in results]
    scores = [score for _, score in results]
    max_confidence = max(scores) if scores else 0.0

    logger.info(
        f"[LangGraph Node: Retriever] Fetched {len(context_chunks)} chunk(s). "
        f"Top confidence score: {max_confidence:.4f}"
    )

    return {
        "context_chunks": context_chunks,
        "max_confidence": max_confidence,
    }


def context_builder_node(state: RAGState) -> dict:
    """Context Builder Node: Evaluates retrieval quality against confidence threshold."""
    max_conf = state.get("max_confidence", 0.0)
    chunks = state.get("context_chunks", [])

    if max_conf < settings.CONFIDENCE_THRESHOLD or not chunks:
        logger.info(
            f"[LangGraph Node: ContextBuilder] Confidence {max_conf:.4f} below threshold "
            f"({settings.CONFIDENCE_THRESHOLD}). Short-circuiting to fallback response."
        )
        return {
            "answer": NOT_FOUND_RESPONSE,
            "context_chunks": [],
            "confidence": max_conf,
        }

    logger.info("[LangGraph Node: ContextBuilder] Context confidence validated. Proceeding to LLM synthesis.")
    return {
        "confidence": max_conf,
    }


def llm_node(state: RAGState) -> dict:
    """LLM Node: Generates grounded response using strict context-only prompt instructions."""
    # If short-circuited by context builder, skip LLM call
    if state.get("answer") == NOT_FOUND_RESPONSE:
        return {}

    question = state["question"]
    chunks = state.get("context_chunks", [])
    formatted_context = format_context_documents(chunks)

    logger.info("[LangGraph Node: LLM] Invoking LLM with strict context prompt.")
    llm = get_llm()
    prompt = RAG_PROMPT_TEMPLATE.format_messages(
        context=formatted_context,
        question=question,
    )

    try:
        response = llm.invoke(prompt)
        answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Enforce exact fallback string if model signals missing context
        if NOT_FOUND_RESPONSE.lower() in answer_text.lower() or "couldn't find" in answer_text.lower():
            answer_text = NOT_FOUND_RESPONSE
            chunks = []  # Clear context chunks if answer is missing

    except Exception as e:
        logger.error(f"[LangGraph Node: LLM] Error invoking LLM: {e}")
        answer_text = NOT_FOUND_RESPONSE
        chunks = []

    return {
        "answer": answer_text,
        "context_chunks": chunks,
    }


def response_formatter_node(state: RAGState) -> dict:
    """Response Formatter Node: Finalizes answer formatting and structure."""
    answer = state.get("answer", NOT_FOUND_RESPONSE)
    chunks = state.get("context_chunks", [])
    confidence = state.get("confidence", 0.0)

    # If answer is missing response, clear context chunks
    if answer == NOT_FOUND_RESPONSE:
        chunks = []

    return {
        "answer": answer,
        "context_chunks": chunks,
        "confidence": round(confidence, 2),
    }


def route_after_context_builder(state: RAGState) -> str:
    """Conditional router determining whether to proceed to LLM or directly to Response Formatter."""
    if state.get("answer") == NOT_FOUND_RESPONSE:
        return "response_formatter"
    return "llm"


def build_rag_graph():
    """Constructs and compiles the stateful LangGraph RAG workflow graph."""
    workflow = StateGraph(RAGState)

    # Add Nodes
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("response_formatter", response_formatter_node)

    # Add Edges & Conditional Routing
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "context_builder")
    workflow.add_conditional_edges(
        "context_builder",
        route_after_context_builder,
        {
            "llm": "llm",
            "response_formatter": "response_formatter",
        },
    )
    workflow.add_edge("llm", "response_formatter")
    workflow.add_edge("response_formatter", END)

    app_graph = workflow.compile()
    logger.info("Successfully compiled LangGraph RAG StateGraph.")
    return app_graph


# Pre-compiled graph instance
rag_app_graph = build_rag_graph()


def run_rag_pipeline(question: str) -> dict:
    """Executes the full LangGraph RAG workflow for a user question.

    Returns:
        dict containing 'answer', 'context', and 'confidence'.
    """
    initial_state = {
        "question": question,
        "context_chunks": [],
        "max_confidence": 0.0,
        "answer": "",
        "confidence": 0.0,
    }

    final_state = rag_app_graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", NOT_FOUND_RESPONSE),
        "context": final_state.get("context_chunks", []),
        "confidence": final_state.get("confidence", 0.0),
    }
