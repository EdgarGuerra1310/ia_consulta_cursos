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
from docx import Document

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

def search_faiss_curso(query, k=4):
    query_embedding = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])  # ← Devuelve todo el diccionario completo
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

def leer_prompt_desde_word(path_docx):
    doc = Document(path_docx)
    texto = "\n".join([p.text for p in doc.paragraphs])
    return texto.strip()

PROMPT_TEMPLATE = leer_prompt_desde_word("prompt_base.docx")

def formatear_chunk_para_contexto(chunk):
    referencia = f"{chunk['source']}, página {chunk['page']}"
    if chunk.get("links"):
        referencia += ", " + ", ".join(chunk["links"])
    return f"---\nTexto:\n{chunk['text'].strip()}\n\nFuente: {referencia}\n---"



@app.route("/chat", methods=["POST"])
def chat():
    #print("🧾 JSON recibido:", request.json)
    user_input = request.json.get("message")
    usuario = request.json.get("usuario", "usuario_default")
    #relevant_texts = search_faiss(user_input, k=3)
    #context = "\n\n".join(relevant_texts)
    relevant_chunks = search_faiss_curso(user_input, k=3)
    context_parts = [formatear_chunk_para_contexto(chunk) for chunk in relevant_chunks]
    context = "\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=user_input)
    # Llamada a OpenAI
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



########################################################################
###############DESARROLLO DE ENTRENAMIENTO##############################
########################################################################
import random
@app.route("/entrenamiento_curso/")
def entrenamiento_home():
    session["training_session"] = {
        "nivel": 1,
        "puntaje": 0,
        "preguntas_previas": [],
        "ultima_pregunta": "",
        "tipo": "abierta"
    }
    return render_template("entrenamiento.html")

@app.route("/entrenamiento_chat", methods=["POST"])
def entrenamiento_chat():
    user_input = request.json.get("message")
    usuario = request.json.get("usuario", "usuario_default")

    training_session = session.get("training_session", {
        "nivel": 1,
        "puntaje": 0,
        "preguntas_previas": [],
        "ultima_pregunta": "",
        "tipo": "abierta"
    })

    # Si es inicio o nueva pregunta
    if not user_input:
        context = "\n\n".join(search_faiss("conceptos básicos", k=3))
        tipo_pregunta = random.choice(["abierta", "alternativas"])

        if tipo_pregunta == "alternativas":
            prompt = f"""
Según el siguiente contexto, genera una pregunta de opción múltiple para entrenamiento docente. Incluye cuatro alternativas, una de ellas correcta. Usa el formato:

PREGUNTA: ...
A) ...
B) ...
C) ...
D) ...
CORRECTA: ...

{context}
"""
        else:
            prompt = f"""
Según el siguiente contexto, genera una pregunta abierta para entrenamiento docente. Devuelve solo la pregunta.

{context}
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        pregunta_generada = response["choices"][0]["message"]["content"].strip()

        training_session["ultima_pregunta"] = pregunta_generada
        training_session["tipo"] = tipo_pregunta
        session["training_session"] = training_session

        return jsonify({
            "question": pregunta_generada,
            "feedback": "",
            "puntaje": training_session["puntaje"],
            "tipo": tipo_pregunta
        })

    # Evaluar respuesta del usuario
    pregunta = training_session["ultima_pregunta"]
    tipo_pregunta = training_session["tipo"]
    context = "\n\n".join(search_faiss(pregunta, k=3))

    prompt_evaluacion = f"""
Contexto del curso:
{context}

Pregunta:
{pregunta}

Respuesta del usuario:
{user_input}

Evalúa si la respuesta es correcta. Si es correcta, responde con:
FEEDBACK: Explicación breve de por qué es correcta.
CORRECTO: sí
REFERENCIA: En qué parte del contenido del curso se puede ampliar la información.

Si no es correcta, responde con:
FEEDBACK: Explicación breve de por qué no es correcta.
CORRECTO: no
REFERENCIA: En qué parte del contenido del curso se puede revisar para entenderlo mejor.

Luego genera una NUEVA_PREGUNTA para continuar el entrenamiento. Si es posible, alterna entre pregunta abierta o de alternativas. Si es de alternativas, usa este formato:

PREGUNTA: ...
A) ...
B) ...
C) ...
D) ...
CORRECTA: ...
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_evaluacion}],
        temperature=0.5
    )
    content = response["choices"][0]["message"]["content"]

    feedback = re.search(r"FEEDBACK:\s*(.*)", content)
    correcto = re.search(r"CORRECTO:\s*(sí|no)", content, re.IGNORECASE)
    nueva_pregunta = re.search(r"PREGUNTA:.*", content, re.DOTALL)
    referencia = re.search(r"REFERENCIA:\s*(.*)", content)

    feedback_text = feedback.group(1).strip() if feedback else "No se pudo evaluar."
    es_correcto = correcto.group(1).strip().lower() == "sí" if correcto else False
    referencia_texto = referencia.group(1).strip() if referencia else "Revisa el contenido del curso."




    nueva_pregunta_texto = nueva_pregunta.group(0).strip() if nueva_pregunta else "No se pudo generar nueva pregunta."

    # Determinar tipo de nueva pregunta
    tipo_siguiente = "alternativas" if re.search(r"A\)\s", nueva_pregunta_texto) else "abierta"

    training_session["puntaje"] += 1 if es_correcto else -1
    training_session["preguntas_previas"].append({
        "pregunta": pregunta,
        "respuesta": user_input,
        "feedback": feedback_text,
        "correcto": es_correcto
    })
    training_session["ultima_pregunta"] = nueva_pregunta_texto
    training_session["tipo"] = tipo_siguiente
    session["training_session"] = training_session

    
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
    CREATE TABLE IF NOT EXISTS interacciones_entrenamiento (
    id SERIAL PRIMARY KEY,
    usuario VARCHAR(100),
    pregunta TEXT,
    respuesta TEXT,
    correcta BOOLEAN,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()

    cursor = conn.cursor()

    # Insertar en la tabla interacciones_entrenamiento
    cursor.execute('''
        INSERT INTO interacciones_entrenamiento (usuario, pregunta, respuesta, correcta)
        VALUES (%s, %s, %s, %s)
    ''', (usuario, pregunta, user_input, es_correcto))
    conn.commit()

    # Cerrar la conexión si ya no se necesita más adelante
    cursor.close()
    conn.close()



    return jsonify({
        "question": nueva_pregunta_texto,
        "feedback": feedback_text + f" (📚 {referencia_texto})",
        "puntaje": training_session["puntaje"],
        "tipo": tipo_siguiente
    })





########################################################################
###############NARRACIONES REFLEXIVAS##############################
########################################################################
from flask import request, redirect, url_for, flash
import PyPDF2

@app.route("/revision_productos/")
def productos_home():
    if "history" not in session:
        session["history"] = []
    return render_template("productos.html")



RUBRICA = {
    "Claridad": "¿El documento presenta ideas de manera clara y comprensible?",
    "Evidencia": "¿El contenido está respaldado con evidencias o ejemplos?",
    "Organización": "¿Está bien estructurado y sigue una secuencia lógica?",
    "Ortografía y gramática": "¿Se usa correctamente la gramática y ortografía?",
    "Conclusión": "¿El documento cierra con una conclusión sólida?"
}

@app.route("/evaluar_pdf", methods=["POST"])
def evaluar_pdf():
    
    if "pdf_file" not in request.files:
        flash("No se seleccionó ningún archivo.")
        return redirect(url_for("home"))

    pdf_file = request.files["pdf_file"]

    # Leer texto del PDF
    reader = PyPDF2.PdfReader(pdf_file)
    texto_pdf = ""
    for page in reader.pages:
        texto_pdf += page.extract_text()

    # Enviar a GPT para evaluación según la rúbrica
    prompt = f"Evalúa el siguiente texto basado en la rúbrica del curso.\n\nTexto:\n{texto_pdf}\n\nRúbrica:\n"
    for criterio, descripcion in RUBRICA.items():
        prompt += f"- {criterio}: {descripcion}\n"

    prompt += "\nDame una retroalimentación detallada para cada punto de la rúbrica."

    respuesta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    evaluacion = respuesta['choices'][0]['message']['content']

    return render_template("productos.html", evaluacion=evaluacion)


##DETECCIÓN CON IMAGENES DENTRO DEL PDF
# from pdf2image import convert_from_bytes
# import pytesseract
# 
# @app.route("/evaluar_pdf", methods=["POST"])
# def evaluar_pdf():
#     if "pdf_file" not in request.files:
#         flash("No se seleccionó ningún archivo.")
#         return redirect(url_for("home"))
# 
#     pdf_file = request.files["pdf_file"]
# 
#     # Extraer texto digital con PyPDF2
#     reader = PyPDF2.PdfReader(pdf_file)
#     texto_pdf = ""
#     for page in reader.pages:
#         texto_pdf += page.extract_text() or ""  # En caso de None
# 
#     # Para OCR, convertir el PDF a imágenes
#     pdf_file.seek(0)  # Volver al inicio del archivo para pdf2image
#     pages = convert_from_bytes(pdf_file.read(), dpi=300)
# 
#     texto_ocr = ""
#     for i, page in enumerate(pages):
#         texto_ocr += pytesseract.image_to_string(page, lang='spa') + "\n"
# 
#     # Combinar texto PyPDF2 + OCR (evita repeticiones si quieres)
#     texto_completo = texto_pdf + "\n" + texto_ocr
# 
#     # Construir prompt con la rúbrica
#     prompt = f"Evalúa el siguiente texto basado en la rúbrica del curso.\n\nTexto:\n{texto_completo}\n\nRúbrica:\n"
#     for criterio, descripcion in RUBRICA.items():
#         prompt += f"- {criterio}: {descripcion}\n"
# 
#     prompt += "\nDame una retroalimentación detallada para cada punto de la rúbrica."
# 
#     respuesta = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7
#     )
# 
#     evaluacion = respuesta['choices'][0]['message']['content']
# 
#     return render_template("productos.html", evaluacion=evaluacion)
# 

# Cargar índice FAISS y metadata antes de definir rutas
index = faiss.read_index(os.path.join(INDEX_FOLDER, "index.faiss"))
with open(os.path.join(INDEX_FOLDER, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
