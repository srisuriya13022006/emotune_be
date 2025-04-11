from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import re
import logging
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# Gemini setup
logging.info("Setting up Gemini API...")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")
logging.info("Gemini API configured.")

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
            return json.loads(match.group(0))
        else:
            return {"books": [], "music": [], "movies": []}
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return {"books": [], "music": [], "movies": []}

# === API route ===
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received REST prediction request.")
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        file_path = "temp.wav"
        file.save(file_path)

        # Send file to Hugging Face Space
        with open(file_path, "rb") as f:
            files = {"data": f}
            hf_response = requests.post("https://srisuriyas-emotune-audio-model.hf.space/run/predict", files=files)

        if hf_response.status_code != 200:
            return jsonify({'error': 'Failed to get emotion from HF Space'}), 500

        emotion = hf_response.json()["data"][0].lower()
        recommendations = get_gemini_recommendations(emotion)

        return jsonify({
            "emotion": emotion,
            "recommendations": recommendations
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask backend (Render-compatible)...")
    app.run(host="0.0.0.0", port=7860)
