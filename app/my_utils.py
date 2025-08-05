import hashlib
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2")


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def extract_clauses_from_url(url: str) -> list[str]:
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator="\n")
    clauses = [line.strip() for line in cleaned.split("\n") if len(line.strip()) > 50]
    return clauses


def is_probably_insurance_policy(text: str) -> bool:
    keywords = ["insurance", "policy", "premium", "claim", "sum insured"]
    return any(k in text.lower() for k in keywords)


def save_clause_cache(clauses: list[str], hash_key: str):
    path = f"cache/{hash_key}.txt"
    os.makedirs("cache", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for clause in clauses:
            f.write(clause + "\n")


def build_faiss_index(clauses: list[str]):
    vectors = model.encode(clauses)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors).astype("float32"))
    return index, vectors
