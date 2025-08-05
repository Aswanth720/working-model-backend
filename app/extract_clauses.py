# app/extractor.py

import requests
import mimetypes
import fitz  # PyMuPDF
import docx
import email
from email import policy
from bs4 import BeautifulSoup
from io import BytesIO
from transformers import AutoTokenizer
import re

# Load tokenizer once (512-token limit for Gemini/Mistral)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# --- 📄 File Extractors ---

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_eml(file_bytes: bytes) -> str:
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                return part.get_content()
            elif ctype == "text/html":
                return BeautifulSoup(part.get_content(), "html.parser").get_text()
    return msg.get_content()

# --- ✂️ Clause Splitter ---

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).replace('\xa0', ' ').strip()

def split_into_clauses(text: str):
    text = text.replace('\r', '').replace('\xa0', ' ').strip()

    # Sentence-level splitting
    sentences = re.split(r'(?<=[.!?]) +', text)
    buffer = ''
    blocks = []
    for sentence in sentences:
        candidate = buffer + ' ' + sentence if buffer else sentence
        if len(tokenizer.tokenize(candidate)) <= 512:
            buffer = candidate
        else:
            if buffer:
                blocks.append(buffer.strip())
            buffer = sentence
    if buffer:
        blocks.append(buffer.strip())

    # Preserve section headers (optional)
    header_pattern = re.compile(r'^(Section\s+\d+|[0-9.]{1,5}\s+[\w ]{3,})')
    combined_blocks = []
    current_header = ''

    for block in blocks:
        if header_pattern.match(block.strip()):
            current_header = block.strip()
        else:
            full_block = f"{current_header}\n{block.strip()}" if current_header else block.strip()
            combined_blocks.append(full_block)
            current_header = ''

    # ✅ Custom Filter: Skip unwanted policy noise
    final = []
    for block in combined_blocks:
        block_clean = clean_text(block)

        # --- ⛔ Skip garbage ---
        if len(block_clean) < 25:
            continue  # Too short
        if block_clean.isupper() and len(block_clean.split()) < 5:
            continue  # Just a heading in uppercase
        if re.match(r'^page\s*\d+', block_clean.lower()):
            continue  # Page numbers like "Page 1"
        if re.match(r'^\d{1,3}$', block_clean):
            continue  # Just a number block
        if 'table of contents' in block_clean.lower():
            continue  # Skip TOC lines

        # Split if too long
        tokens = tokenizer.tokenize(block_clean)
        if len(tokens) > 512:
            for i in range(0, len(tokens), 512):
                chunk = tokenizer.convert_tokens_to_string(tokens[i:i + 512])
                final.append({
                    "clause": clean_text(chunk),
                    "id": f"clause_{len(final)+1}"
                })
        else:
            final.append({
                "clause": block_clean,
                "id": f"clause_{len(final)+1}"
            })

    return final

# --- 🌐 Entry Point ---

def extract_clauses_from_url(url: str):
    response = requests.get(url)
    file_bytes = response.content
    content_type = response.headers.get("Content-Type")
    mime_type, _ = mimetypes.guess_type(url)

    if not mime_type and content_type:
        mime_type = content_type

    if mime_type:
        if "pdf" in mime_type:
            raw_text = extract_text_from_pdf(file_bytes)
        elif "word" in mime_type or "docx" in mime_type:
            raw_text = extract_text_from_docx(file_bytes)
        elif "plain" in mime_type or url.endswith(".txt"):
            raw_text = extract_text_from_txt(file_bytes)
        elif "message/rfc822" in mime_type or url.endswith(".eml"):
            raw_text = extract_text_from_eml(file_bytes)
        else:
            raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = extract_text_from_pdf(file_bytes)

    return split_into_clauses(raw_text)
