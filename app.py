from flask import Flask, request, jsonify
import os
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ---------------- Initialize ----------------
app = Flask(__name__)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Global vars for lazy loading
cnn_model = lstm_model = rnn_model = meta_model = None
tokenizer = label_encoder = None

# ---------------- Preprocessing ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------------- Model Loaders ----------------
def load_assets():
    global cnn_model, lstm_model, rnn_model, meta_model, tokenizer, label_encoder
    if not all([cnn_model, lstm_model, rnn_model, meta_model]):
        print("🔄 Lazy loading models...")
        cnn_model = load_model("cnn_model.keras")
        lstm_model = load_model("lstm_model.keras")
        rnn_model = load_model("rnn_model.keras")
        meta_model = load_model("meta_model.keras")
        print("✅ Models loaded.")

    if tokenizer is None or label_encoder is None:
        print("🔄 Loading tokenizer and encoder...")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print("✅ Tokenizer and encoder loaded.")

# ---------------- Prediction ----------------
def predict_category(headline):
    load_assets()
    cleaned = clean_text(headline)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=200)

    cnn_pred = cnn_model.predict(padded)
    lstm_pred = lstm_model.predict(padded)
    rnn_pred = rnn_model.predict(padded)

    stacked = np.concatenate([cnn_pred, lstm_pred, rnn_pred], axis=1)
    final_pred = meta_model.predict(stacked)

    label = label_encoder.inverse_transform([np.argmax(final_pred)])
    return label[0]

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return "✅ News Classification API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        headline = data.get("headline", None)

        if not headline:
            return jsonify({"error": "Missing 'headline' in request"}), 400

        print(f"📨 Received headline: {headline}")
        category = predict_category(headline)
        return jsonify({"category": category})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------- Main ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
