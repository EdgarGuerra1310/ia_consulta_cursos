import os
import json
from glob import glob
import re
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from docx import Document
import fitz  # PyMuPDF para extraer links en PDF

DATA_FOLDER = "data"
INDEX_FOLDER = "vector_index"
os.makedirs(INDEX_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
    return embeddings

def extract_links_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    links = []
    for page in doc:
        for link in page.get_links():
            if 'uri' in link:
                links.append(link['uri'])
    return links

def pdf_to_single_chunk(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    links = extract_links_pdf(pdf_path)
    return [{"text": text, "source": os.path.basename(pdf_path), "links": links}]

def extract_links_docx(doc):
    links = []
    rels = doc.part.rels
    for rel in rels.values():
        if "hyperlink" in rel.reltype:
            links.append(rel.target_ref)
    return links

def docx_to_chunks(docx_path):
    doc = Document(docx_path)
    chunks = []
    current_text = ""

    # Extraemos todos los links del documento una sola vez
    all_links = extract_links_docx(doc)

    def is_heading(paragraph):
        return paragraph.style.name.startswith('Heading')

    for para in doc.paragraphs:
        if is_heading(para) and current_text:
            chunks.append({"text": current_text.strip(), "source": os.path.basename(docx_path), "links": all_links})
            current_text = para.text + "\n"
        else:
            current_text += para.text + "\n"

    # último chunk
    if current_text:
        chunks.append({"text": current_text.strip(), "source": os.path.basename(docx_path), "links": all_links})

    return chunks
all_chunks = []

for file_path in glob(os.path.join(DATA_FOLDER, "*")):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # Un solo chunk con todo el texto y links
        all_chunks.extend(pdf_to_single_chunk(file_path))
    elif ext == ".docx":
        all_chunks.extend(docx_to_chunks(file_path))

print(f"Total chunks creados: {len(all_chunks)}")

texts = [chunk["text"] for chunk in all_chunks]
embeddings = embed(texts).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(INDEX_FOLDER, "index.faiss"))
with open(os.path.join(INDEX_FOLDER, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("✅ Índice creado con links en", INDEX_FOLDER)