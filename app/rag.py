from typing import List
from langchain_core.prompts import ChatPromptTemplate

# Constant exact response mandated when answer is missing or context confidence is low
NOT_FOUND_RESPONSE = "I couldn't find this information in the provided Agentic AI eBook."

# Strict System Prompt Template enforcing absolute grounding in retrieved context
SYSTEM_PROMPT = """You are a Retrieval-Augmented AI assistant. Answer ONLY using the supplied context. If the context does not contain the answer, respond exactly:

"I couldn't find this information in the provided Agentic AI eBook."

Do not use prior knowledge. Do not infer or guess.

Context:
{context}
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def format_context_documents(docs: List[str]) -> str:
    """Formats a list of context strings into a numbered context block."""
    if not docs:
        return ""
    formatted_chunks = [f"[Chunk {i+1}]:\n{doc.strip()}" for i, doc in enumerate(docs)]
    return "\n\n".join(formatted_chunks)
