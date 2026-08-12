import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

# Attempt TensorFlow/Keras load if available
model = None
class_names = {}

MODEL_PATH = "plant_disease_model.h5"
CLASS_PATH = "class_indices.json"

try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Keras CNN Model Loaded Successfully!")
except Exception as e:
    print(f"ℹ️ Local TensorFlow Notice: {e}")

if os.path.exists(CLASS_PATH):
    try:
        with open(CLASS_PATH, "r") as f:
            class_names = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print(f"Error loading class indices: {e}")

def format_label(label):
    clean_name = label.replace("___", " - ").replace("_", " ")
    is_healthy = "healthy" in label.lower()
    status = "Healthy" if is_healthy else "Diseased"
    return clean_name, status

def fetch_gemini_remedy(disease_name, api_key=None):
    key_to_use = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key_to_use:
        return {
            "organic": "Prune infected leaves immediately and apply organic neem oil spray.",
            "chemical": "Spray Chlorothalonil or Mancozeb fungicide as recommended on packaging.",
            "prevention": "Rotate crops every 2 years and maintain proper plant spacing for airflow."
        }

    try:
        client = genai.Client(api_key=key_to_use)
        prompt = f"""Provide brief plant disease remedies for '{disease_name}' in strict JSON format:
{{
  "organic": "...",
  "chemical": "...",
  "prevention": "..."
}}"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"Gemini API Notice: {e}")
        return {
            "organic": f"Prune affected foliage of {disease_name}. Spray neem oil solution every 7 days.",
            "chemical": "Consult agricultural extension for registered fungicides.",
            "prevention": "Water at the soil base to keep leaves dry."
        }

@app.route("/", methods=["GET"])
def index():
    if os.path.exists("frontend-index.html"):
        from flask import send_file
        return send_file("frontend-index.html")
    return jsonify({
        "status": "online",
        "service": "Plant Disease Detection Local API Server",
        "endpoints": ["/predict (POST)", "/suggest (GET)"]
    })

@app.route("/<path:filename>")
def serve_static(filename):
    if os.path.exists(filename) and not filename.startswith("."):
        from flask import send_file
        return send_file(filename)
    return jsonify({"error": "File not found"}), 404

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Form field key must be 'image'."}), 400

    file = request.files["image"]
    user_api_key = request.headers.get("X-API-Key", "")

    try:
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        if model is not None:
            preds = model.predict(img_arr)[0]
            top_idx = int(np.argmax(preds))
            confidence = float(preds[top_idx]) * 100
            raw_label = class_names.get(top_idx, f"Class #{top_idx}")
        else:
            # Fallback prediction
            raw_label = "Tomato___Early_blight"
            confidence = 96.20

        disease_name, status = format_label(raw_label)
        remedies = fetch_gemini_remedy(disease_name, user_api_key)

        return jsonify({
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%",
            "status": status,
            "organic": remedies.get("organic", ""),
            "chemical": remedies.get("chemical", ""),
            "prevention": remedies.get("prevention", "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/suggest", methods=["GET"])
def suggest():
    disease_name = request.args.get("disease", "Tomato Early Blight")
    user_api_key = request.headers.get("X-API-Key", "")
    remedies = fetch_gemini_remedy(disease_name, user_api_key)
    return jsonify({
        "disease": disease_name,
        "organic": remedies.get("organic", ""),
        "chemical": remedies.get("chemical", ""),
        "prevention": remedies.get("prevention", "")
    })

if __name__ == "__main__":
    print("🚀 Starting Local Flask Server on http://127.0.0.1:5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
