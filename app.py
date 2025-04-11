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
genai.configure(api_key=os.getenv("AIzaSyB7QIqPNg89yzr-t3msQANz13gbOsNh3BI"))
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

        file_path = "temp.wav"
        file.save(file_path)
        logging.info(f"Saved uploaded audio to {file_path}")

        # === Send file to Hugging Face Gradio Space ===
        logging.info("Sending audio file to Hugging Face Space...")
        hf_url = "https://srisuriyas-emotune-audio-model.hf.space/api/predict"

        with open(file_path, "rb") as f:
            payload = {
                "data": [("temp.wav", f, "audio/wav")]
            }
            # NOTE: requests-toolbelt could be used here for complex cases, but Gradio accepts simple multipart/form-data
            files = {
                "data": f
            }

            response = requests.post(hf_url, files=files)
        
        logging.info(f"HF Space responded with status {response.status_code}")
        if response.status_code != 200:
            logging.error(f"HF Space response error: {response.text}")
            return jsonify({'error': 'Failed to get emotion from HF Space'}), 500

        hf_json = response.json()
        emotion = hf_json.get("data", [""])[0].lower()
        logging.info(f"Received emotion from HF Space: {emotion}")

        # === Get recommendations from Gemini ===
        recommendations = get_gemini_recommendations(emotion)
        logging.info("Successfully generated recommendations.")

        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed temporary file: {file_path}")

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
