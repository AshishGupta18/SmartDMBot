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
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 5
DISTANCE_THRESHOLD = 0.70
CACHE_DIR = "cache"
TRAIN_DATA_DIR = "train_data"
CHUNKS_CACHE_PATH = os.path.join(CACHE_DIR, "json_chunks.pkl")
VECTORSTORE_CACHE_PATH = os.path.join(CACHE_DIR, "json_vectorstore.faiss")

app = Flask(__name__)

# --- Transform JSON to Text (optimized for Gemini) ---
def transform_idoc_json_to_text(json_data, filename=None):
    lines = []

    if filename:
        base_title = os.path.splitext(filename)[0].replace("_", " ").strip()
        lines.append(f"--- Source: {base_title} ---")
        lines.append(f"{base_title} Retrofit Guide\n")

    if "object type" in json_data:
        lines.append(f"Object Type: {json_data['object type']}")
        lines.append("")

    if "description" in json_data:
        desc = json_data["description"]
        lines.append("Description:")
        if isinstance(desc, dict):
            for k, v in desc.items():
                if isinstance(v, dict):
                    lines.append(f"  {k.capitalize()}:")
                    for subk, subv in v.items():
                        lines.append(f"    {subk}: {subv}")
                else:
                    lines.append(f"  {k.capitalize()}: {v}")
        else:
            lines.append(f"  {desc}")
        lines.append("")

    if "tcode" in json_data:
        lines.append(f"TCode: {json_data['tcode']}")
        lines.append("")

    if "tool_used" in json_data:
        lines.append("Tools Used:")
        for tool in json_data["tool_used"]:
            lines.append(f"- {tool}")
        lines.append("")

    if "retrofit_process" in json_data:
        lines.append("Retrofit Process:")
        if isinstance(json_data["retrofit_process"], dict):
            for k, v in json_data["retrofit_process"].items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {json_data['retrofit_process']}")
        lines.append("")

    if "comparison notes" in json_data:
        lines.append("Comparison Notes:")
        if isinstance(json_data["comparison notes"], dict):
            for k, v in json_data["comparison notes"].items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {json_data['comparison notes']}")
        lines.append("")

    if "common_errors" in json_data:
        lines.append("Common Errors:")
        for err in json_data["common_errors"]:
            lines.append(f"- {err}")
        lines.append("")

    if "best_practices" in json_data:
        lines.append("Best Practices:")
        for practice in json_data["best_practices"]:
            lines.append(f"- {practice}")
        lines.append("")

    if "chatbot_responses" in json_data:
        lines.append("Chatbot Responses:")
        for k, v in json_data["chatbot_responses"].items():
            lines.append(f"{k}:")
            if isinstance(v, dict):
                for subk, subv in v.items():
                    if isinstance(subv, list):
                        for item in subv:
                            lines.append(f"  - {item}")
                    else:
                        lines.append(f"  {subk}: {subv}")
            else:
                lines.append(f"  {v}")
        lines.append("")

    return "\n".join(lines)

# --- Load and transform all JSON files into text ---
def load_text_from_json_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    formatted_text = transform_idoc_json_to_text(data, filename)
                    all_text += formatted_text + "\n\n"
            except json.JSONDecodeError as e:
                print(f"❌ JSON error in file: {filename} → {e}")
            except Exception as e:
                print(f"⚠️ Unexpected error in file: {filename} → {e}")
    return all_text

# --- Split text into overlapping chunks ---
def get_text_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# --- Prepare vectorstore and cache ---
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

    embeddings = [
        genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=chunk,
            task_type="retrieval_document"
        )['embedding'] for chunk in text_chunks
    ]
    embeddings_np = np.array(embeddings, dtype='float32')

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    with open(CHUNKS_CACHE_PATH, 'wb') as f:
        pickle.dump(text_chunks, f)
    faiss.write_index(index, VECTORSTORE_CACHE_PATH)

    for i, chunk in enumerate(text_chunks[:5]):
        print(f"Chunk {i} preview:\n{chunk[:300]}\n---\n")

    return index, text_chunks

vectorstore, text_chunks = setup_chatbot()

# --- Prompt Construction ---
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

    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=query,
        task_type="retrieval_query"
    )['embedding']
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
        # ✅ Post-process: reformat the output using Gemini again
        reformat_prompt = f"""
                    Format the following answer into a structured and readable format:
        - Use bullet points or numbered steps
        - Use bold for headers if needed
        - Maintain spacing for readability
        Answer:
        {answer}
        """
        structured_response = get_general_answer(reformat_prompt)
        answer = structured_response

    final_answer = f"{source}\n\n{answer}"

    print(jsonify(final_answer))
    return jsonify({"answer": final_answer})



# --- Run App ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)
