from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODELS = {
    "yolo9": YOLO("models/yolo9.pt"),
    "yolo11": YOLO("models/yolo11.pt"),
}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model_name = request.form.get("model")
    if model_name not in MODELS:
        return jsonify({"error": "Invalid model"}), 400

    image_file = request.files["image"]
    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    model = MODELS[model_name]
    results = model(img)

    detected = False
    confidence = 0.0
    detections = []

    for r in results:
        img_width, img_height = img.size
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls].lower() == "bayam":
                detected = True
                confidence = max(confidence, conf)
                
                # Get bounding box coordinates in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to x, y, width, height format (normalized 0-1)
                x = x1 / img_width
                y = y1 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                detections.append({
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "confidence": round(conf * 100, 2)
                })

    return jsonify({
        "is_bayam": detected,
        "confidence": round(confidence * 100, 2),
        "detections": detections
    })

if __name__ == "__main__":
    app.run(debug=True)
