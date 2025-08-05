from flask import Flask, request, jsonify
import os
import numpy as np
import faiss
import PyPDF2
import google.generativeai as genai
import pickle

# --- Configuration ---
GEMINI_API_KEY = "AIzaSyCfdXeEy0aSeDRZzOb6-4Gyu6Hbq3FExLo"
PDF_PATH = "workbook.pdf"
EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5
DISTANCE_THRESHOLD = 0.70
CACHE_DIR = "cache"
pdf_filename = os.path.splitext(os.path.basename(PDF_PATH))[0]
CHUNKS_CACHE_PATH = os.path.join(CACHE_DIR, f"{pdf_filename}_chunks.pkl")
VECTORSTORE_CACHE_PATH = os.path.join(CACHE_DIR, f"{pdf_filename}_vectorstore.faiss")

app = Flask(__name__)

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def setup_chatbot():
    os.makedirs(CACHE_DIR, exist_ok=True)
    genai.configure(api_key=GEMINI_API_KEY)

    if os.path.exists(CHUNKS_CACHE_PATH) and os.path.exists(VECTORSTORE_CACHE_PATH):
        with open(CHUNKS_CACHE_PATH, 'rb') as f:
            text_chunks = pickle.load(f)
        vectorstore = faiss.read_index(VECTORSTORE_CACHE_PATH)
        return vectorstore, text_chunks

    raw_text = extract_text_from_pdf(PDF_PATH)
    text_chunks = get_text_chunks(raw_text)
    embeddings = [genai.embed_content(model=EMBEDDING_MODEL_NAME, content=chunk, task_type="retrieval_document")['embedding'] for chunk in text_chunks]
    embeddings_np = np.array(embeddings, dtype='float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    with open(CHUNKS_CACHE_PATH, 'wb') as f:
        pickle.dump(text_chunks, f)
    faiss.write_index(index, VECTORSTORE_CACHE_PATH)

    return index, text_chunks

vectorstore, text_chunks = setup_chatbot()

def get_contextual_answer(query, context):
    prompt = f"""You are a helpful assistant. 
Answer based only on this context:
---
{context}
---
Question: {query}
Answer:"""
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

def get_general_answer(query):
    prompt = f"Answer the following question: {query}"
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get("question", "")

    query_embedding = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=query, task_type="retrieval_query")['embedding']
    query_embedding = np.array([query_embedding], dtype='float32')

    distances, indices = vectorstore.search(query_embedding, k=TOP_K)
    best_distance = distances[0][0]

    if best_distance > DISTANCE_THRESHOLD:
        answer = get_general_answer(query)
    else:
        retrieved_chunks = [text_chunks[i] for i in indices[0]]
        context = "\n---\n".join(retrieved_chunks)
        answer = get_contextual_answer(query, context)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=5000)