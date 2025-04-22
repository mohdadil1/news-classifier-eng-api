from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Initialize ----------------
app = Flask(__name__)
nltk.download("stopwords")
nltk.download("wordnet")

# Load the saved models
cnn_model = load_model("cnn_model.keras")
lstm_model = load_model("lstm_model.keras")
rnn_model = load_model("rnn_model.keras")
meta_model = load_model("meta_model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- Preprocessing ----------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Clean the text by converting to lowercase, removing non-alphabetic characters, etc.
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------------- Prediction Function ----------------

def predict_category(headline):
    # Step 1: Preprocess the input headline
    headline = clean_text(headline)  # Clean the text
    sequence = tokenizer.texts_to_sequences([headline])  # Tokenize the text
    padded_sequence = pad_sequences(sequence, maxlen=200)  # Pad the sequence to match the input shape

    # Step 2: Get predictions from the base models (CNN, LSTM, GRU)
    cnn_pred = cnn_model.predict(padded_sequence)
    lstm_pred = lstm_model.predict(padded_sequence)
    rnn_pred = rnn_model.predict(padded_sequence)

    # Step 3: Stack predictions
    stacked_pred = np.concatenate([cnn_pred, lstm_pred, rnn_pred], axis=1)

    # Step 4: Get the final prediction from the meta-model
    final_pred = meta_model.predict(stacked_pred)

    # Step 5: Decode the predicted class using the label encoder
    predicted_class = label_encoder.inverse_transform([np.argmax(final_pred)])

    return predicted_class[0]

# ---------------- Flask Route ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "headline" not in data:
            return jsonify({"error": "Missing 'headline' in request"}), 400

        headline = data["headline"]
        category = predict_category(headline)  # Only one value returned

        return jsonify({
            "category": category
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
