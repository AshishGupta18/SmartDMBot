import json
import os
import sys
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app)

# --- Constants and File Paths ---
# Get the absolute path of the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'chatbot_model.pkl')
RULES_PATH = os.path.join(BASE_DIR, 'rules.json')


def load_rules(filename=RULES_PATH):
    """
    Loads the rule-based dataset from the specified JSON file.
    Handles potential file errors.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The rules file '{filename}' was not found. Please make sure it's in the same directory.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The rules file '{filename}' is not a valid JSON file. Please check its format.")
        sys.exit(1)

def load_model(model_path=MODEL_PATH):
    """
    Loads the pre-trained machine learning model.
    """
    if not os.path.exists(model_path):
        print(f"Error: ML model '{model_path}' not found. Please run train_model.py first.")
        sys.exit(1)
    return joblib.load(model_path)

# --- Load Rules and Model on Startup ---
rules = load_rules()
model = load_model()
print("✅ Rules and ML Model loaded successfully.")


def ml_generate_answer(obj_type, status, is_segment=None):
    """
    Generates an answer using the ML model.
    The input format must match the format used during training.
    """
    if obj_type == "TABL" and status == "NEW" and is_segment is not None:
        inp = f"{obj_type} | {status} | {is_segment}"
    else:
        inp = f"{obj_type} | {status}"
    
    prediction = model.predict([inp])
    return prediction[0]


@app.route('/get_answer', methods=['POST'])
def get_answer_api():
    """
    API endpoint to get an answer from the chatbot logic.
    It receives object type and status from the frontend.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    obj_type = data.get('obj_type', '').upper()
    
    # --- Object Type Validation ---
    # First, check if the object type exists in our rules.
    if obj_type not in rules.get("objects", {}):
        return jsonify({
            "error": "object_not_found",
            "message": f"Object type '{obj_type}' is not present in the model."
        }), 404

    status = data.get('status', '').upper()
    is_segment = data.get('is_segment', '').upper() if data.get('is_segment') else None

    # --- Rule-Based Logic ---
    obj_data = rules.get("objects", {}).get(obj_type)
    final_answer = None

    if obj_data and status in obj_data:
        if obj_type == "TABL" and status == "NEW":
            if is_segment == "YES":
                # For IDOC segments, return the first item
                final_answer = obj_data[status][0]
            elif is_segment == "NO":
                # For regular tables, return the rest of the items joined
                final_answer = "\n".join(obj_data[status][1:])
        else:
            # For all other cases, join the array items
            if isinstance(obj_data[status], list):
                final_answer = "\n".join(obj_data[status])

    # --- ML Fallback ---
    if final_answer is None:
        print(f"No specific rule found for '{obj_type} | {status} | {is_segment or ''}'. Using ML model.")
        final_answer = ml_generate_answer(obj_type, status, is_segment)
    else:
        print(f"Rule found for '{obj_type} | {status} | {is_segment or ''}'.")

    return jsonify({'answer': final_answer})

# --- Main execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
