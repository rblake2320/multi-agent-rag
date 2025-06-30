import sys
from langchain.chains.router import RouterChain
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

VECTOR_BASE = "vector_stores"
DOMAINS = {
    "legal": "legal_chroma",
    "code": "code_chroma",
    "finance": "finance_chroma",
}

LLM = LlamaCpp(model_path="models/llama-3.1-70b-q2_k.gguf", n_gpu_layers=-1, n_ctx=4096)

def build_retriever(domain_key: str):
    store = Chroma(persist_directory=f"{VECTOR_BASE}/{DOMAINS[domain_key]}")
    return store.as_retriever()

retrievers = {k: build_retriever(k) for k in DOMAINS}

router_prompt = PromptTemplate.from_template(
    "You are a router that decides which domain a question belongs to.\n"
    "Domains: legal, code, finance.\n"
    "Return exactly one of these words.\n"
    "Question: {{input}}"
)

router = RouterChain.from_defaults(
    llm=LLM,
    prompt=router_prompt,
    destinations=list(DOMAINS.keys()),
)

def answer(question: str):
    domain = router.run(question)
    qa = RetrievalQA.from_chain_type(llm=LLM, retriever=retrievers[domain])
    return qa.run(question), domain

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python router.py "<question>"")
        sys.exit(1)
    response, used_domain = answer(sys.argv[1])
    print(f"[{used_domain}] → {response}")
