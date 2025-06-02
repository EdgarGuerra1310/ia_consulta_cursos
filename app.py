import os
import json
from glob import glob
from flask import Flask, request, render_template, jsonify, session
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import re
import openai

#import sqlite3
import psycopg2



load_dotenv()

DATA_FOLDER = "data"
INDEX_FOLDER = "vector_index"
os.makedirs(INDEX_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Configurar API Key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cargar modelo embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Función para buscar en FAISS ---
def search_faiss(query, k=4):
    query_embedding = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx]["text"])
    return results

# --- Generar preguntas recomendadas con OpenAI chat completions ---
def generate_recommended_questions(text, n=10):
    prompt = f"""
Genera {n} preguntas concisas y claras sobre el siguiente contenido, sin mencionar la fuente ni lugar:

{text}

Devuelve solo las preguntas numeradas, sin explicaciones ni texto adicional.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # o "gpt-3.5-turbo"
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    answer = response['choices'][0]['message']['content']
    questions = [line.strip().lstrip("0123456789. ") for line in answer.split("\n") if line.strip()]
    return questions

@app.route("/consulta_curso/")
def home():
    if "history" not in session:
        session["history"] = []
    return render_template("index.html")

@app.route("/get_recommended_questions")
def get_recommended_questions():
    import random
    if not metadata:
        return jsonify({"questions": ["No hay contenido disponible."]})
    # Seleccionar aleatoriamente varios elementos distintos de la metadata
    num_chunks = min(5, len(metadata))  # puedes ajustar cuántos trozos usar
    chunks = random.sample(metadata, num_chunks)
    # Obtener el texto de cada chunk
    texts = [item["text"] for item in chunks]
    # Generar preguntas a partir de todos los textos combinados o por separado
    combined_text = " ".join(texts)
    questions = generate_recommended_questions(combined_text, n=10)

    return jsonify({"questions": questions})

@app.route("/chat", methods=["POST"])
def chat():
    #print("🧾 JSON recibido:", request.json)
    user_input = request.json.get("message")
    usuario = request.json.get("usuario", "usuario_default")
    relevant_texts = search_faiss(user_input, k=3)
    context = "\n\n".join(relevant_texts)
    prompt = f"""Contesta de forma clara y concisa. Además, proporciona enlaces o recursos adicionales relacionados (videos, artículos, etc.) si es posible, usando únicamente el contexto:

Contexto:
{context}

Pregunta:
{user_input}

Respuesta:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # o "gpt-3.5-turbo"
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    answer = response['choices'][0]['message']['content'].strip()
    #return jsonify({"answer": answer})

#    # Guardar interacción en la base de datos
#    cursor.execute(
#        "INSERT INTO interacciones (usuario, mensaje, respuesta) VALUES (?, ?, ?)",
#        ("usuario1", user_input, answer)
#    )
#    conn.commit()


    # Conectar a PostgreSQL (ajusta usuario, db, password)
    conn = psycopg2.connect(
        dbname="proyectos_ia",
        user="postgres",
        password="1edgarGUERRA",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Crear tabla si no existe (PostgreSQL)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interacciones (
        id SERIAL PRIMARY KEY,
        usuario TEXT NOT NULL,
        mensaje TEXT NOT NULL,
        respuesta TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()

    # En la función chat(), cambia la inserción a:
    cursor.execute(
        "INSERT INTO interacciones (usuario, mensaje, respuesta) VALUES (%s, %s, %s)",
        (usuario, user_input, answer)
    )
    conn.commit()    

    return jsonify({"answer": answer})



#   # Crear o conectar a la base SQLite (archivo local)
#   conn = sqlite3.connect('interacciones.db', check_same_thread=False)
#   cursor = conn.cursor()
#   
#   # Crear tabla para guardar interacciones si no existe
#   cursor.execute('''
#   CREATE TABLE IF NOT EXISTS interacciones (
#       id INTEGER PRIMARY KEY AUTOINCREMENT,
#       usuario TEXT NOT NULL,
#       mensaje TEXT NOT NULL,
#       respuesta TEXT NOT NULL,
#       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#   )
#   ''')
#   conn.commit()








if __name__ == "__main__":
    # Cargar índice FAISS y metadata antes de iniciar la app
    index = faiss.read_index(os.path.join(INDEX_FOLDER, "index.faiss"))
    with open(os.path.join(INDEX_FOLDER, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    app.run(host="0.0.0.0", port=5000, debug=True)
