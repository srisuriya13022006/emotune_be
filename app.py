from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os
import json
import re
import logging
import google.generativeai as genai

# === Setup ===
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# Load HuggingFace pipeline
logging.info("Loading HuggingFace audio emotion recognition model...")
pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
logging.info("Model loaded successfully.")

# Gemini setup
logging.info("Configuring Gemini API...")
genai.configure(api_key="AIzaSyB7QIqPNg89yzr-t3msQANz13gbOsNh3BI")
model = genai.GenerativeModel("gemini-1.5-pro")
logging.info("Gemini configured successfully.")

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
    {{"title": "book1", "link": "https://goodreads.com/..."}},
    {{"title": "book2", "link": "https://goodreads.com/..."}},
    {{"title": "book3", "link": "https://goodreads.com/..."}}
  ],
  "music": [
    {{"title": "song1", "link": "https://open.spotify.com/..."}},
    {{"title": "song2", "link": "https://open.spotify.com/..."}},
    {{"title": "song3", "link": "https://open.spotify.com/..."}}
  ],
  "movies": [
    {{"title": "movie1", "link": "https://www.imdb.com/title/..."}},
    {{"title": "movie2", "link": "https://www.imdb.com/title/..."}},
    {{"title": "movie3", "link": "https://www.imdb.com/title/..."}}
  ]
}}
Return only the JSON response and nothing else.
"""
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        logging.info(f"Raw Gemini response: {content}")

        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            recommendations = json.loads(match.group(0))
            logging.info(f"Extracted recommendations: {recommendations}")
            return recommendations
        else:
            logging.warning("No valid JSON found in Gemini response.")
            return {
                "books": ["No recommendations found"],
                "music": ["No recommendations found"],
                "movies": ["No recommendations found"]
            }
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return {
            "books": ["No recommendations due to error"],
            "music": ["No recommendations due to error"],
            "movies": ["No recommendations due to error"]
        }

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received request for prediction.")
    try:
        if 'file' not in request.files:
            logging.warning("No file part in the request.")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("No selected file.")
            return jsonify({'error': 'No selected file'}), 400

        file_path = "temp_audio.wav"
        file.save(file_path)
        logging.info(f"File saved to {file_path}")

        logging.info("Running emotion prediction...")
        results = pipe(file_path)
        emotion = results[0]['label'].lower()
        logging.info(f"Predicted Emotion: {emotion}")

        logging.info("Getting Gemini recommendations...")
        recommendations = get_gemini_recommendations(emotion)

        logging.info("Sending response to client.")
        return jsonify({
            "emotion": emotion,
            "recommendations": recommendations
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True)
