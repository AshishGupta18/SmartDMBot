from flask import Flask, request, jsonify
import os
import json
import numpy as np
from datetime import datetime
import faiss
import pickle
from dotenv import load_dotenv
import google.generativeai as genai
import subprocess
import glob

from flask import send_from_directory

app = Flask(__name__)

@app.route('/svg/<filename>')
def serve_svg(filename):
    return send_from_directory('output/svg', filename)

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.5-flash")

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

# # Function to delete the most recent SVG file only
# def delete_most_recent_svg():
#     """Delete only the most recent SVG file to prevent stale images while retaining chat history"""
#     svg_dir = os.path.join("output", "svg")
#     if os.path.exists(svg_dir):
#         svg_files = glob.glob(os.path.join(svg_dir, "diagram_*.svg"))
#         if svg_files:
#             # Sort files by modification time (newest first)
#             svg_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
#             # Delete only the most recent file
#             most_recent_file = svg_files[0]
#             try:
#                 os.remove(most_recent_file)
#                 print(f"🗑️ Deleted most recent SVG: {most_recent_file}")
#             except Exception as e:
#                 print(f"⚠️ Error deleting {most_recent_file}: {e}")

# # Function to delete all SVG files (keeping for reference)
# def delete_all_svg_files():
#     """Delete all SVG files - use only when needed"""
#     svg_dir = os.path.join("output", "svg")
#     if os.path.exists(svg_dir):
#         svg_files = glob.glob(os.path.join(svg_dir, "diagram_*.svg"))
#         for svg_file in svg_files:
#             try:
#                 os.remove(svg_file)
#                 print(f"🗑️ Deleted SVG: {svg_file}")
#             except Exception as e:
#                 print(f"⚠️ Error deleting {svg_file}: {e}")

# Function to keep all SVGs (no deletion)
def keep_all_svgs():
    """Keep all SVG files - no deletion, all images remain in chat history"""
    print("📁 Keeping all SVG files for chat history")

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

    # Keep all SVG files - no deletion to preserve chat history
    keep_all_svgs()

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
        - Use bold for headers if needed*
        - Maintain spacing for readability
        Answer:
        {answer}
        """
        structured_response = get_general_answer(reformat_prompt)
        answer = structured_response

    final_answer = f"{answer}"
    
    # Initialize svg_url as None
    svg_url = None
    
    if best_distance <= DISTANCE_THRESHOLD:
    # Flowchart generation
        steps_file_path = "steps.txt"
        with open(steps_file_path, "w", encoding="utf-8") as f:
            f.write(answer)

        # Create separate folders
        output_d2_dir = os.path.join("output", "d2")
        output_svg_dir = os.path.join("output", "svg")
        os.makedirs(output_d2_dir, exist_ok=True)
        os.makedirs(output_svg_dir, exist_ok=True)

        # Generate fresh timestamp for each request
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_d2_file = os.path.join(output_d2_dir, f"diagram_{timestamp}.d2")
        svg_file = os.path.join(output_svg_dir, f"diagram_{timestamp}.svg")

        # Read steps
        with open(steps_file_path, "r", encoding="utf-8") as f:
            steps = f.read()

        # Gemini prompt
        prompt = f"""
        You are a developer assistant. Convert the following algorithm steps into a D2 flowchart.
        Use correct syntax that will render without error in the D2 CLI.

        Set the layout direction to top-down (vertical flow) using:
        direction: down

        Use basic shapes (rectangle, diamond for decisions), and arrows for flow.

        Output only valid D2 syntax. Do not add explanation or markdown backticks.

        Steps:
        {steps}
        """

        print("⏳ Generating D2 diagram with Gemini...")
        response = model.generate_content(prompt)
        d2_code = response.text.strip()

        # Save D2 to timestamped file (only)
        with open(backup_d2_file, "w", encoding="utf-8") as f:
            f.write(d2_code)
        print(f"🗂️  D2 saved: {backup_d2_file}")

        # Render from timestamped file
        try:
            subprocess.run(["d2", backup_d2_file, svg_file], check=True)
            print(f"✅ SVG generated: {svg_file}")
            os.remove(steps_file_path)
            # Only set svg_url if SVG was successfully generated
            svg_url = f"/svg/diagram_{timestamp}.svg"
        except subprocess.CalledProcessError as e:
            print(f"❌ Error rendering D2 diagram: {e}")
            svg_url = None

    # Return response with or without SVG
    response_data = {
        "answer": final_answer.replace("\n", "<br>")
    }
    
    # Only include svg in response if it was successfully generated
    if svg_url:
        response_data["svg"] = svg_url
    
    return jsonify(response_data)





# --- Run App ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)
