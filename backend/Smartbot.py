import os
import json
import numpy as np
import pandas as pd
import re
import nltk
import random
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_FILE = "C:/Users/aditya/Documents/Dataset_retrofit.xlsx"  # Your dataset with Prompt, Intent, Response
TEXT_COLUMN = "Prompt"
LABEL_COLUMN = "Intent"
RESPONSE_COLUMN = "Response"

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LEN = 40
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.15
VAL_SIZE = 0.1
MODEL_DIR = "saved_models_rnn_chat"

df = pd.read_excel(DATA_FILE)

#nltk.download('wordnet', quiet=True)

stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])
sequences = tokenizer.texts_to_sequences(df["clean_text"])
padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")

le = LabelEncoder()
encoded_labels = le.fit_transform(df[LABEL_COLUMN])
num_classes = len(le.classes_)
#print("Classes:", le.classes_)

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "intent_rnn_full.keras")
tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.json")

if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
    print("[INFO] Loading existing model...")
    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    with open(label_encoder_path, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
        le = LabelEncoder()
        le.classes_ = np.array(classes)


else:
    print("[INFO] Training new model...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        padded, encoded_labels,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=encoded_labels,
        random_state=42
    )

    val_relative = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        stratify=y_temp,
        random_state=42
    )

    model = Sequential()
    model.add(Embedding(input_dim=min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1),
                        output_dim=EMBEDDING_DIM,
                        input_length=MAX_SEQUENCE_LEN))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, "intent_rnn_best.keras"), save_best_only=True, monitor="val_loss")
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    with open(label_encoder_path, "w", encoding="utf-8") as f:
        json.dump({"classes": le.classes_.tolist()}, f)

    model.save(model_path)

with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

with open(os.path.join(MODEL_DIR, "label_encoder.json"), "w", encoding="utf-8") as f:
    json.dump({"classes": le.classes_.tolist()}, f)

model.save(os.path.join(MODEL_DIR, "intent_rnn_full.keras"))
print("\nModel and tokenizer saved to", MODEL_DIR)

def gpt_style_response(intent_label):
    responses = df[df[LABEL_COLUMN] == intent_label][RESPONSE_COLUMN].tolist()
    if not responses:
        return "I’m not sure, but let me try to help you with that."
    base_resp = random.choice(responses)
    return f"Sure! Here’s what I found for you:\n\n{base_resp}\n\nIf you’d like, I can explain this further."

def chatbot_reply(user_input):
    txt = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([txt])
    pad_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    probs = model.predict(pad_seq)[0]
    pred_idx = np.argmax(probs)
    intent_label = le.classes_[pred_idx]
    confidence = probs[pred_idx]

    steps = [
        f"Step 1: User entered -> {user_input}",
        f"Step 2: Cleaned text -> {txt}",
        f"Step 3: Model predicted intent -> {intent_label} (Confidence: {confidence:.2f})",
        f"Step 4: Response selected"
    ]

    reply = gpt_style_response(intent_label)
    return reply, intent_label, steps

if __name__ == "__main__":
    print("Backend ready", flush=True)  # signal Electron
    for line in sys.stdin:
        try:
            data = json.loads(line.strip())
            user_input = data.get("message", "")
            reply, intent, steps = chatbot_reply(user_input)
            response = {"reply": reply, "intent": intent, "steps": steps}
            print(json.dumps(response), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)



'''from flask import Flask, request, jsonify ###If the Flask is working use this

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    reply, intent, steps = chatbot_reply(user_input)
    return jsonify({"reply": reply, "intent": intent, "steps": steps})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
    -------------------------------------------------''' 
    
#This is for the CLI loop that is terminal run
'''if __name__ == "__main__":
    from IPython.display import display, Markdown

    print("\n--- Retrofit Chatbot Ready ---\nType 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break
        reply, intent, confidence = chatbot_reply(user_input)
        display(Markdown(f"**Bot:** {reply}"))'''