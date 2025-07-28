import argparse
import pathlib

from langchain_community.document_loaders import UnstructuredFileLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOADER_MAP = {
    ".pdf": UnstructuredFileLoader,
    ".docx": UnstructuredFileLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}

def load_docs(path: pathlib.Path):
    docs = []
    for fp in path.rglob("*.*"):
        loader_cls = LOADER_MAP.get(fp.suffix.lower())
        if not loader_cls:
            continue  # skip unsupported
        docs.extend(loader_cls(str(fp)).load())
    return docs

def ingest(domain: str, data_path: pathlib.Path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = load_docs(data_path)
    splits = splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vs_path = pathlib.Path("vector_stores") / f"{domain}_chroma"
    vs_path.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(splits, embedder, persist_directory=str(vs_path))
    print(f"✅ Ingested {len(splits)} chunks into {vs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs into a domain vector store")
    parser.add_argument("--domain", required=True, help="Domain name, e.g. legal_docs")
    parser.add_argument("--path", required=True, help="Folder with raw documents")
    args = parser.parse_args()

    ingest(args.domain, pathlib.Path(args.path))
