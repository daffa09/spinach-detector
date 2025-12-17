from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort
import numpy as np
import io

app = Flask(__name__)
CORS(app)

MODELS = {
    "yolo9": "models/yolo9.onnx",
    "yolo11": "models/yolo11.onnx",
}

SESSIONS = {}

def load_models():
    for name, path in MODELS.items():
        SESSIONS[name] = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"]
        )

load_models()

def preprocess(image):
    image = image.resize((640, 640))
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model_name = request.form.get("model")
    if model_name not in SESSIONS:
        return jsonify({"error": "Invalid model"}), 400

    image = Image.open(
        io.BytesIO(request.files["image"].read())
    ).convert("RGB")

    input_tensor = preprocess(image)
    session = SESSIONS[model_name]

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: input_tensor})

    detections = []
    detected = False
    confidence = 0.0

    for det in outputs[0][0]:
        conf = float(det[4])
        cls = int(det[5])

        # asumsi class bayam = 0
        if cls == 0 and conf > 0.4:
            detected = True
            confidence = max(confidence, conf)

            x, y, w, h = det[:4]
            detections.append({
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "confidence": round(conf * 100, 2),
            })

    return jsonify({
        "model": model_name,
        "is_bayam": detected,
        "confidence": round(confidence * 100, 2),
        "detections": detections
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
