from flask import Flask, request, jsonify
import os
import json
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
import google.generativeai as genai

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configuration ---
EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5
DISTANCE_THRESHOLD = 0.70
CACHE_DIR = "cache"
TRAIN_DATA_DIR = "train_data"
CHUNKS_CACHE_PATH = os.path.join(CACHE_DIR, "json_chunks.pkl")
VECTORSTORE_CACHE_PATH = os.path.join(CACHE_DIR, "json_vectorstore.faiss")

app = Flask(__name__)

# --- JSON Transform Function ---

def transform_idoc_json_to_text(json_data):
    title = json_data.get("document_title", "Untitled Document")
    text_output = [f"Title: {title}\n"]

    pages = json_data.get("pages", [])
    for page in pages:
        page_num = page.get("page_number", "Unknown Page")
        text_output.append(f"\n--- Page {page_num} ---")

        sections = page.get("sections", [])
        for section in sections:
            section_title = section.get("section_title", "Untitled Section")
            text_output.append(f"\nSection: {section_title}")

            content = section.get("content", "")
            if isinstance(content, list):
                for item in content:
                    text_output.append(f"- {item}")
            elif isinstance(content, str):
                text_output.append(content)

            subsections = section.get("subsections", [])
            for sub in subsections:
                sub_title = sub.get("title", "Subsection")
                text_output.append(f"  ➤ {sub_title}")
                sub_content = sub.get("content", "")
                if isinstance(sub_content, list):
                    for item in sub_content:
                        text_output.append(f"    - {item}")
                elif isinstance(sub_content, str):
                    text_output.append(f"    {sub_content}")

    return "\n".join(text_output)

# --- Updated JSON Loader Function ---

def load_text_from_json_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    formatted_text = transform_idoc_json_to_text(data)
                    all_text += formatted_text + "\n\n"
            except json.JSONDecodeError as e:
                print(f"❌ JSON error in file: {filename} → {e}")
            except Exception as e:
                print(f"⚠️ Unexpected error in file: {filename} → {e}")
    return all_text

# --- Chunking Function ---

def get_text_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# --- Setup Embeddings & FAISS Vectorstore ---

def setup_chatbot():
    os.makedirs(CACHE_DIR, exist_ok=True)
    genai.configure(api_key=GEMINI_API_KEY)

    if os.path.exists(CHUNKS_CACHE_PATH) and os.path.exists(VECTORSTORE_CACHE_PATH):
        with open(CHUNKS_CACHE_PATH, 'rb') as f:
            text_chunks = pickle.load(f)
        vectorstore = faiss.read_index(VECTORSTORE_CACHE_PATH)
        return vectorstore, text_chunks

    raw_text = load_text_from_json_folder(TRAIN_DATA_DIR)
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

# --- Gemini Prompt Functions ---

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

# --- API Endpoint ---

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get("question", "")

    query_embedding = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=query, task_type="retrieval_query")['embedding']
    query_embedding = np.array([query_embedding], dtype='float32')

    distances, indices = vectorstore.search(query_embedding, k=TOP_K)
    best_distance = distances[0][0]

    if best_distance > DISTANCE_THRESHOLD:
        source = "🔍 *Answer generated by Gemini (no matching training data)*"
        answer = get_general_answer(query)
    else:
        retrieved_chunks = [text_chunks[i] for i in indices[0]]
        context = "\n---\n".join(retrieved_chunks)
        source = "📚 *Answer based on your training data (train_data)*"
        answer = get_contextual_answer(query, context)

    final_answer = f"{source}\n\n{answer}"
    return jsonify({"answer": final_answer})

# --- Run the App ---

if __name__ == '__main__':
    app.run(port=5000)
