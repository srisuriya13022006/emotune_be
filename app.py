from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import json
import re
import google.generativeai as genai
from transformers import pipeline
import torch
import librosa
import soundfile as sf

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# === Gemini setup ===
logging.info("Setting up Gemini API...")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Set GEMINI_API_KEY in Render env
model = genai.GenerativeModel("gemini-1.5-pro")
logging.info("Gemini API configured.")

# === Load local model pipeline once ===
logging.info("Loading local emotion classification model...")
classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    framework="pt",
    device=0 if torch.cuda.is_available() else -1
)
logging.info("Local model loaded successfully.")

# === Gemini recommendation logic ===
def get_gemini_recommendations(emotion):
    logging.info(f"Generating recommendations for emotion: {emotion}")
    
    prompt = f"""
Given the user's emotion: "{emotion}", suggest a JSON object with personalized:
- 3 books
- 3 songs/music
- 3 movies

Each item must include:
- The title
- A link (Spotify for music, Goodreads for books, IMDb for movies)

Respond strictly in this JSON format:
{{
  "books": [
    {{"title": "book1", "link": "https://goodreads.com/..."}} ...
  ],
  "music": [
    ...
  ],
  "movies": [
    ...
  ]
}}
Return only the JSON response and nothing else.
"""
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            logging.info("Parsed Gemini response successfully.")
            return json.loads(match.group(0))
        else:
            logging.warning("No JSON object found in Gemini response.")
            return {"books": [], "music": [], "movies": []}
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return {"books": [], "music": [], "movies": []}

# === API route ===
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("==> /predict endpoint hit")
    try:
        if 'file' not in request.files:
            logging.warning("No file part found in request.")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("File part is empty.")
            return jsonify({'error': 'No selected file'}), 400

        original_path = "temp.wav"
        converted_path = "temp_converted.wav"
        file.save(original_path)
        logging.info(f"Saved uploaded audio to {original_path}")

        # === Convert audio for local classification ===
        y, sr = librosa.load(original_path, sr=None)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        if y_resampled.ndim > 1:
            y_resampled = y_resampled.mean(axis=0)

        sf.write(converted_path, y_resampled, 16000)
        logging.info(f"Audio resampled and saved to {converted_path}")

        # === Run local model inference ===
        result = classifier(converted_path)
        logging.info(f"Model result: {result}")
        emotion = result[0]["label"].lower()
        logging.info(f"Predicted emotion: {emotion}")

        # === Generate recommendations ===
        recommendations = get_gemini_recommendations(emotion)
        logging.info("Recommendations fetched successfully.")

        # === Cleanup ===
        for path in [original_path, converted_path]:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Removed temp file: {path}")

        return jsonify({
            "emotion": emotion,
            "recommendations": recommendations
        })

    except Exception as e:
        logging.error("Prediction error occurred", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask backend (Render-compatible)...")
    app.run(host="0.0.0.0", port=7860)
