from configuration import *
from crewai import LLM
# ============= MODEL SETUP ============= #
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM (Ollama local model) --- #
# llm = Ollama(model="llama3.2:3b")

# ========== for streamlit deployment ========== #
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.9,
)

# ========== EMBEDDINGS ========== #
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)
