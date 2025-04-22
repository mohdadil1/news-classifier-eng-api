from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# ---------------- Initialize ----------------
app = Flask(__name__)
nltk.download("stopwords")
nltk.download("wordnet")

print("🔄 Loading models and tokenizer...")

# Load models
cnn_model = load_model("cnn_model.keras")
lstm_model = load_model("lstm_model.keras")
rnn_model = load_model("rnn_model.keras")
meta_model = load_model("meta_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("✅ Models and tokenizer loaded.")

# ---------------- Preprocessing ----------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------------- Prediction Function ----------------

def predict_category(headline):
    cleaned = clean_text(headline)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    cnn_pred = cnn_model.predict(padded_sequence, verbose=0)
    lstm_pred = lstm_model.predict(padded_sequence, verbose=0)
    rnn_pred = rnn_model.predict(padded_sequence, verbose=0)

    stacked_pred = np.concatenate([cnn_pred, lstm_pred, rnn_pred], axis=1)
    final_pred = meta_model.predict(stacked_pred, verbose=0)

    predicted_class = label_encoder.inverse_transform([np.argmax(final_pred)])
    return predicted_class[0]

# ---------------- Routes ----------------

@app.route("/", methods=["GET"])
def home():
    return "✅ News Classification API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)  # Ensure JSON is parsed even without correct header
        if "headline" not in data:
            return jsonify({"error": "Missing 'headline' in request"}), 400

        headline = data["headline"]
        print("📨 Received headline:", headline)

        category = predict_category(headline)
        print("✅ Predicted category:", category)

        return jsonify({"category": category})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# ---------------- Run App ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)
