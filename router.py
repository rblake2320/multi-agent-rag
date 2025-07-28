import sys
import os
from typing import Optional, Tuple, Dict

from langchain.chains.router import RouterChain
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

VECTOR_BASE = "vector_stores"
DOMAINS: Dict[str, str] = {
    "legal": "legal_chroma",
    "code": "code_chroma",
    "finance": "finance_chroma",
}

# Make model path configurable via environment variable
DEFAULT_MODEL_PATH = "models/llama-3.1-70b-q2_k.gguf"
MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)


def initialize_llm() -> Optional[LlamaCpp]:
    """Initialize LLM with proper error handling.
    
    Returns:
        LlamaCpp instance if successful, None otherwise
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please set LLAMA_MODEL_PATH environment variable or place model at default location")
        return None

    try:
        return LlamaCpp(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None


# Initialize LLM with error handling
LLM = initialize_llm()


def build_retriever(domain_key: str):
    """Build a retriever for the specified domain.
    
    Args:
        domain_key: Key identifying the domain
        
    Returns:
        Vector store retriever for the domain
    """
    store = Chroma(persist_directory=f"{VECTOR_BASE}/{DOMAINS[domain_key]}")
    return store.as_retriever()


retrievers = {k: build_retriever(k) for k in DOMAINS}

router_prompt = PromptTemplate.from_template(
    "You are a router that decides which domain a question belongs to.\n"
    "Domains: legal, code, finance.\n"
    "Return exactly one of these words.\n"
    "Question: {input}"
)


def initialize_router() -> Optional[RouterChain]:
    """Initialize router with proper error handling.
    
    Returns:
        RouterChain instance if successful, None otherwise
    """
    if LLM is None:
        return None

    return RouterChain.from_defaults(
        llm=LLM,
        prompt=router_prompt,
        destinations=list(DOMAINS.keys()),
    )


router = initialize_router()


def answer(question: str) -> Tuple[str, str]:
    """Answer a question using the appropriate domain retriever.
    
    Args:
        question: Question to answer
        
    Returns:
        Tuple of (response, domain_used)
    """
    if LLM is None or router is None:
        return "Error: LLM or router not properly initialized. Please check model path.", "error"

    domain = router.run(question)
    qa = RetrievalQA.from_chain_type(llm=LLM, retriever=retrievers[domain])
    return qa.run(question), domain


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python router.py \"<question>\"")
        sys.exit(1)
    response, used_domain = answer(sys.argv[1])
    print(f"[{used_domain}] → {response}")
