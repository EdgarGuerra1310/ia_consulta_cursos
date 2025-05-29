import os
import fitz  # PyMuPDF para leer PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Carga variables de entorno del archivo .env
load_dotenv()

def extract_text_from_pdfs(pdf_folder):
    """Extrae texto de todos los PDFs en una carpeta."""
    texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(pdf_folder, filename)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                texts.append((filename, text))
    return texts

def split_text(text, chunk_size=500, overlap=50):
    """Divide el texto en chunks pequeños con solapamiento."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.create_documents([text])

def load_documents_and_build_index(pdf_folder, index_folder="vector_index"):
    raw_docs = extract_text_from_pdfs(pdf_folder)
    
    all_chunks = []
    for filename, text in raw_docs:
        print(f"Procesando: {filename}")
        chunks = split_text(text)
        all_chunks.extend(chunks)

    print(f"Total de chunks: {len(all_chunks)}")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    # Guarda el índice localmente para usar después
    vectorstore.save_local(index_folder)
    print(f"Índice guardado en: {index_folder}")

if __name__ == "__main__":
    load_documents_and_build_index(pdf_folder="data")

