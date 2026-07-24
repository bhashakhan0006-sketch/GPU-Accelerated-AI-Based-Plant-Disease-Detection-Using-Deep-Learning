# ==============================================================================
# 🌿 PLANT DISEASE DETECTION & TREATMENT ADVICE - GOOGLE COLAB BACKEND
# Dataset: Kaggle PlantVillage (vipoooool/new-plant-diseases-dataset)
# Framework: TensorFlow / Keras + Flask + PyNgrok / LocalTunnel
# ==============================================================================

import os
import sys
import json
import glob
import numpy as np
from PIL import Image

# ------------------------------------------------------------------------------
# STEP 1: INSTALL REQUIRED LIBRARIES
# ------------------------------------------------------------------------------
print("=== Step 1: Installing Dependencies ===")
os.system("pip install -q kagglehub opendatasets pyngrok flask-cors tensorflow pillow")

import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

# ------------------------------------------------------------------------------
# STEP 2: DOWNLOAD KAGGLE DATASET AUTOMATICALLY
# ------------------------------------------------------------------------------
print("\n=== Step 2: Downloading Kaggle Dataset via KaggleHub ===")
try:
    dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print(f"✅ Kaggle Dataset Downloaded Successfully to: {dataset_path}")
except Exception as e:
    print(f"KaggleHub note: {e}. Trying fallback download...")
    import opendatasets as od
    od.download("https://www.kaggle.com/datasets/vipoooool/NEW-PLANT-DISEASES-DATASET", data_dir="./dataset")
    dataset_path = "./dataset/New Plant Diseases Dataset(Augmented)"

train_dir, valid_dir = None, None
for root, dirs, files in os.walk(dataset_path):
    if "train" in dirs and "valid" in dirs:
        train_dir = os.path.join(root, "train")
        valid_dir = os.path.join(root, "valid")
        break

print(f"📂 Training Directory: {train_dir}")
print(f"📂 Validation Directory: {valid_dir}")

# ------------------------------------------------------------------------------
# STEP 3: PREPROCESS DATA & TRAIN DEEP LEARNING MODEL
# ------------------------------------------------------------------------------
print("\n=== Step 3: Training Deep Learning Model (MobileNetV2) ===")

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
valid_gen = valid_datagen.flow_from_directory(valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

class_indices = train_gen.class_indices
class_names = {v: k for k, v in class_indices.items()}

with open("class_indices.json", "w") as f:
    json.dump(class_names, f, indent=4)
print("Saved class_indices.json with", len(class_names), "plant categories!")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("plant_disease_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_gen, epochs=5, validation_data=valid_gen, callbacks=[checkpoint, early_stop])
print("🎉 Model Training Complete! Saved as plant_disease_model.h5")

# ------------------------------------------------------------------------------
# STEP 4: FLASK BACKEND WITH REMEDIES & SUGGESTIONS API
# ------------------------------------------------------------------------------
DISEASE_SUGGESTIONS = {
    "early blight": {
        "organic": "Prune infected leaves. Spray copper fungicide or neem oil every 7-10 days.",
        "chemical": "Apply Chlorothalonil or Mancozeb fungicide according to package instructions.",
        "prevention": "Rotate crops every 2-3 years. Mulch soil to prevent splashing.",
        "care": "Water at root base; keep foliage completely dry."
    },
    "late blight": {
        "organic": "Remove infected parts immediately. Spray organic copper soap fungicide.",
        "chemical": "Apply Metalaxyl or Chlorothalonil fungicides immediately.",
        "prevention": "Use resistant seeds. Space plants 3ft apart for airflow.",
        "care": "Destroy crop debris after harvest."
    },
    "scab": {
        "organic": "Rake autumn leaves. Apply liquid lime-sulfur or neem oil early spring.",
        "chemical": "Spray Myclobutanil or Captan fungicides during bud break.",
        "prevention": "Prune tree canopy for sunlight and aeration.",
        "care": "Avoid heavy nitrogen fertilizers."
    },
    "rust": {
        "organic": "Dust sulfur or copper early on. Plant early in season.",
        "chemical": "Apply Azoxystrobin or Propiconazole if rust exceeds 5% leaf cover.",
        "prevention": "Plant rust-resistant hybrids. Rotate crops.",
        "care": "Clear grassy weeds near garden."
    },
    "black rot": {
        "organic": "Remove black mummified fruit. Spray copper soap pre-bloom.",
        "chemical": "Apply Mancozeb or Myclobutanil early in growth cycle.",
        "prevention": "Prune for open canopy ventilation.",
        "care": "Ensure full sunlight."
    },
    "healthy": {
        "organic": "Plant is healthy! Apply seaweed extract for root vigor.",
        "chemical": "No chemical treatment needed.",
        "prevention": "Inspect leaves weekly. Sanitize garden tools.",
        "care": "Maintain consistent moisture and good sunlight."
    }
}

def get_remedy(disease_name):
    d_lower = disease_name.lower()
    for key, data in DISEASE_SUGGESTIONS.items():
        if key in d_lower:
            return {**data, "disease": disease_name}
    
    return {
        "disease": disease_name,
        "organic": f"Prune infected foliage of {disease_name}. Apply organic neem oil spray.",
        "chemical": f"Apply suitable broad-spectrum fungicide following product instructions.",
        "prevention": "Ensure good spacing for airflow and rotate crops annually.",
        "care": "Water at the soil base early morning and enrich soil with organic compost."
    }

app = Flask(__name__)
CORS(app)

trained_model = tf.keras.models.load_model("plant_disease_model.h5")
with open("class_indices.json", "r") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

def format_label(label):
    clean_name = label.replace("___", " - ").replace("_", " ")
    is_healthy = "healthy" in label.lower()
    status = "Healthy" if is_healthy else "Diseased"
    return clean_name, status

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "message": "Plant Disease Colab Backend API is running!"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        preds = trained_model.predict(img_arr)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx]) * 100

        raw_label = label_map.get(top_idx, f"Class #{top_idx}")
        disease_name, status = format_label(raw_label)
        remedies = get_remedy(disease_name)

        return jsonify({
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%",
            "status": status,
            "suggestions": remedies
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/suggest", methods=["GET", "POST", "OPTIONS"])
def suggest():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    disease_name = request.args.get("disease", "") or "Tomato Early Blight"
    return jsonify(get_remedy(disease_name))

# Start Tunnel
try:
    public_url = ngrok.connect(5000)
    print("\n=======================================================")
    print(f"🚀 GOOGLE COLAB BACKEND PUBLIC URL: {public_url.public_url}")
    print("=======================================================\n")
except Exception as e:
    print(f"Ngrok info: {e}")

if __name__ == "__main__":
    app.run(port=5000)
