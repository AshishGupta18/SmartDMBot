from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json
import numpy as np
from datetime import datetime
import faiss
import pickle
from dotenv import load_dotenv
import google.generativeai as genai
import subprocess
import glob
import re

# === PyInstaller resource-path helper ===
def resource_path(relative_path):
    """Get absolute path to resource for dev and PyInstaller."""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

app = Flask(__name__)

@app.route('/svg/<filename>')
def serve_svg(filename):
    # Output SVGs go in a runtime-created folder
    svg_dir = os.path.join(os.getcwd(), "output", "svg")
    return send_from_directory(svg_dir, filename)

# New endpoint to serve images from IMAGES folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the IMAGES folder"""
    images_dir = resource_path("IMAGES")
    return send_from_directory(images_dir, filename)

# New endpoint to get all available images
@app.route('/api/images', methods=['GET'])
def get_all_images():
    """Get list of all available images organized by category"""
    images_dir = resource_path("IMAGES")
    images_data = {}
    
    try:
        for category in os.listdir(images_dir):
            category_path = os.path.join(images_dir, category)
            if os.path.isdir(category_path):
                images_data[category] = []
                for image_file in os.listdir(category_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        images_data[category].append({
                            'filename': image_file,
                            'path': f'/images/{category}/{image_file}',
                            'category': category
                        })
    except Exception as e:
        return jsonify({'error': f'Error reading images: {str(e)}'}), 500
    
    return jsonify(images_data)

# New endpoint to search for relevant images based on query
@app.route('/api/search-images', methods=['POST'])
def search_images():
    """Search for relevant images based on user query"""
    data = request.get_json()
    query = data.get("query", "").lower()
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    images_dir = resource_path("IMAGES")
    relevant_images = []
    
    try:
        # Define search keywords for each category
        category_keywords = {
            'ENHO': ['enhancement', 'enho'],
            'IDOC & IEXT': ['idoc', 'iext', 'interface', 'document'],
            'INTF': ['interface', 'intf'],
            'REPT': ['report', 'rept'],
            'IWPR': ['iwpr', 'workflow'],
            'R3TR_PROG': ['program', 'r3tr', 'prog'],
            'REPS': ['reps'],
            'TABL_TABD_TABT': ['table', 'tabl', 'tabd', 'tabt'],
            'TABU': ['tabu'],
            'DOMA_DOMD': ['domain', 'doma', 'domd'],
            'DTEL_DTED': ['data element', 'dtel', 'dted']
        }
        
        print(f"🔍 Searching for images related to query: '{query}'")
        print(f"📁 Images directory: {images_dir}")
        print(f"🎯 Checking {len(category_keywords)} categories for keyword matches...")
        
        # Search through categories for relevant images
        for category, keywords in category_keywords.items():
            category_path = os.path.join(images_dir, category)
            if os.path.exists(category_path):
                # Check if query matches category keywords
                if any(keyword in query for keyword in keywords):
                    print(f"✅ Category '{category}' matched! Keywords: {[k for k in keywords if k in query]}")
                    # Get all images from this category
                    for image_file in os.listdir(category_path):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            relevant_images.append({
                                'filename': image_file,
                                'path': f'/images/{category}/{image_file}',
                                'category': category,
                                'relevance_score': 1.0
                            })
                            print(f"   📸 Added image: {category}/{image_file}")
                else:
                    print(f"❌ Category '{category}' - no keyword match")
        
        # If no category matches, try to find images with filename matching
        if not relevant_images:
            print("🔍 No category matches found, trying filename matching...")
            for category in os.listdir(images_dir):
                category_path = os.path.join(images_dir, category)
                if os.path.isdir(category_path):
                    for image_file in os.listdir(category_path):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            # Check if filename contains query terms
                            filename_lower = image_file.lower()
                            if query in filename_lower or any(word in filename_lower for word in query.split()):
                                relevant_images.append({
                                    'filename': image_file,
                                    'path': f'/images/{category}/{image_file}',
                                    'category': category,
                                    'relevance_score': 0.8
                                })
                                print(f"📸 Filename match: {category}/{image_file}")
        
        # Sort by relevance score
        relevant_images.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"🎯 Total relevant images found: {len(relevant_images)}")
        if relevant_images:
            print("📋 Selected images:")
            for i, img in enumerate(relevant_images, 1):
                print(f"   {i}. {img['category']}/{img['filename']} -> {img['path']} (score: {img['relevance_score']})")
        else:
            print("⚠️ No images found!")
        
        return jsonify({
            'query': query,
            'images': relevant_images,
            'total_found': len(relevant_images)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error searching images: {str(e)}'}), 500

# New endpoint to get images by category
@app.route('/api/images/<category>', methods=['GET'])
def get_images_by_category(category):
    """Get all images from a specific category"""
    images_dir = resource_path("IMAGES")
    category_path = os.path.join(images_dir, category)
    
    if not os.path.exists(category_path) or not os.path.isdir(category_path):
        return jsonify({'error': f'Category {category} not found'}), 404
    
    try:
        images = []
        for image_file in os.listdir(category_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append({
                    'filename': image_file,
                    'path': f'/images/{category}/{image_file}',
                    'category': category
                })
        
        return jsonify({
            'category': category,
            'images': images,
            'total': len(images)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error reading category {category}: {str(e)}'}), 500

# --- Load environment variables from bundled .env ---
load_dotenv(resource_path(".env"))
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


# Function to keep all SVGs (no deletion)
def keep_all_svgs():
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
    # Use resource_path for bundled train_data
    raw_text = load_text_from_json_folder(resource_path(TRAIN_DATA_DIR))
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
    model_instance = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model_instance.generate_content(prompt)
    return response.text

def get_general_answer(query):
    prompt = f"Answer the following question: {query}"
    model_instance = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model_instance.generate_content(prompt)
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
    
    # Initialize svg_url and relevant_images as None
    svg_url = None
    relevant_images = []
    
    if best_distance <= DISTANCE_THRESHOLD:
                # Simple image search: if folder name keywords match, show all images from that folder
        print(f"✅ Good training data match found (distance: {best_distance:.3f} <= {DISTANCE_THRESHOLD}), searching for relevant images...")
        try:
            images_dir = resource_path("IMAGES")
            query_lower = query.lower()
            
            print(f"🔍 Searching for images related to query: '{query}'")
            print(f"📁 Images directory: {images_dir}")
            
            # Simple folder name keywords
            folder_keywords = {
                'ENHO': ['enhancement','enho'],
                'IDOC & IEXT': ['idoc', 'iext', 'interface'],
                'INTF': ['interface', 'intf'],
                'REPT': ['report', 'rept'],
                'IWPR': ['iwpr', 'workflow'],
                'R3TR_PROG': ['program', 'r3tr', 'prog'],
                'REPS': ['reps' ],
                'TABL_TABD_TABT': ['table', 'tabl', 'tabd', 'tabt'],
                'TABU': ['tabu'],
                'DOMA_DOMD': ['domain', 'doma', 'domd'],
                'DTEL_DTED': ['data element', 'dtel', 'dted']
            }
            
            # Check each folder for keyword matches
            for folder_name, keywords in folder_keywords.items():
                folder_path = os.path.join(images_dir, folder_name)
                if os.path.exists(folder_path):
                    print(f"🔍 Checking folder '{folder_name}' with keywords: {keywords}")
                    print(f"🔍 Query: '{query_lower}'")
                    # If any keyword matches, add ALL images from that folder
                    if any(keyword in query_lower for keyword in keywords):
                        print(f"✅ Folder '{folder_name}' matched! Keywords: {[k for k in keywords if k in query_lower]}")
                        print(f"📸 Adding ALL images from {folder_name} folder...")
                        
                        for image_file in os.listdir(folder_path):
                            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                relevant_images.append({
                                    'filename': image_file,
                                    'path': f'/images/{folder_name}/{image_file}',
                                    'category': folder_name
                                })
                                print(f"   📸 Added: {folder_name}/{image_file}")
                    else:
                        print(f"❌ Folder '{folder_name}' - no keyword match")
                        print(f"   Keywords: {keywords}")
                        print(f"   Query: '{query_lower}'")
                        print(f"   Matches found: {[k for k in keywords if k in query_lower]}")
            
            print(f"🎯 Total images found: {len(relevant_images)}")
            if relevant_images:
                print("📋 Images to show:")
                for i, img in enumerate(relevant_images, 1):
                    print(f"   {i}. {img['category']}/{img['filename']}")
            else:
                print("⚠️ No matching folders found!")
                
        except Exception as e:
            print(f"⚠️ Error searching for images: {e}")
            relevant_images = []
        
        # Generate D2 diagram and SVG
        steps_file_path = "steps.txt"
        with open(steps_file_path, "w", encoding="utf-8") as f:
            f.write(answer)
        # Output folders in working directory (not bundled)
        output_d2_dir = os.path.join(os.getcwd(), "output", "d2")
        output_svg_dir = os.path.join(os.getcwd(), "output", "svg")
        os.makedirs(output_d2_dir, exist_ok=True)
        os.makedirs(output_svg_dir, exist_ok=True)
        # Generate fresh timestamp for each request
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_d2_file = os.path.join(output_d2_dir, f"diagram_{timestamp}.d2")
        svg_file = os.path.join(output_svg_dir, f"diagram_{timestamp}.svg")
        with open(steps_file_path, "r", encoding="utf-8") as f:
            steps = f.read()
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
        with open(backup_d2_file, "w", encoding="utf-8") as f:
            f.write(d2_code)
        print(f"🗂️  D2 saved: {backup_d2_file}")
        try:
            subprocess.run(["d2", backup_d2_file, svg_file], check=True)
            print(f"✅ SVG generated: {svg_file}")
            os.remove(steps_file_path)

            # Only set svg_url if SVG was successfully generated
            svg_url = f"/svg/diagram_{timestamp}.svg"
        except subprocess.CalledProcessError as e:
            print(f"❌ Error rendering D2 diagram: {e}")
            svg_url = None
    else:
        print(f"❌ No good training data match (distance: {best_distance:.3f} > {DISTANCE_THRESHOLD}), skipping image search")
        print("🔍 Answer generated by Gemini (no matching training data) - no images will be shown")
    response_data = {
        "answer": final_answer.replace("\n", "<br>")
    }
    
    # Only include svg in response if it was successfully generated
    if svg_url:
        response_data["svg"] = svg_url
    if relevant_images:
        response_data["relevant_images"] = relevant_images
    
    return jsonify(response_data)


# --- Run App ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)
