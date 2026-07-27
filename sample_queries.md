# Sample Evaluation Queries

Use these sample questions to test the RAG chatbot via `POST /chat`.

## In-Domain Questions (Expected: Detailed answers extracted strictly from PDF)

### 1. What is ReAct?
```json
{
  "question": "What is ReAct?"
}
```

### 2. Explain agent memory.
```json
{
  "question": "Explain agent memory."
}
```

### 3. How does LangGraph work?
```json
{
  "question": "How does LangGraph work?"
}
```

### 4. What are planning agents?
```json
{
  "question": "What are planning agents?"
}
```

### 5. What is tool calling?
```json
{
  "question": "What is tool calling?"
}
```

---

## Out-of-Domain Questions (Expected: Anti-hallucination fallback response)

### 6. What is the capital of France?
```json
{
  "question": "What is the capital of France?"
}
```
**Expected Response:**
```json
{
  "answer": "I couldn't find this information in the provided Agentic AI eBook.",
  "context": [],
  "confidence": 0.0
}
```

### 7. Who won IPL 2025?
```json
{
  "question": "Who won IPL 2025?"
}
```
**Expected Response:**
```json
{
  "answer": "I couldn't find this information in the provided Agentic AI eBook.",
  "context": [],
  "confidence": 0.0
}
```
