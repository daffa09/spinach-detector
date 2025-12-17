from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# =====================
# Load ONNX Models
# =====================
MODELS = {
    "yolo9": ort.InferenceSession("models/yolo9.onnx", providers=["CPUExecutionProvider"]),
    "yolo11": ort.InferenceSession("models/yolo11.onnx", providers=["CPUExecutionProvider"]),
}

CLASS_NAMES = {0: "bayam"}
IMG_SIZE = 640
CONF_THRESHOLD = 0.4


# =====================
# Preprocess
# =====================
def preprocess(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image).astype(np.float32) / 255.0  # RGB [0,1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # BCHW
    return img


# =====================
# Postprocess
# =====================
def postprocess(outputs, orig_w, orig_h):
    preds = outputs[0][0]  # shape: (num_boxes, 5 + num_classes)
    detections = []

    for p in preds:
        obj_conf = p[4]
        class_scores = p[5:]

        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]
        conf = obj_conf * class_conf

        if conf < CONF_THRESHOLD or class_id != 0:
            continue

        x, y, w, h = p[:4]

        # Convert to normalized xywh (0–1)
        x1 = (x - w / 2) / IMG_SIZE
        y1 = (y - h / 2) / IMG_SIZE
        w /= IMG_SIZE
        h /= IMG_SIZE

        detections.append({
            "x": float(x1),
            "y": float(y1),
            "width": float(w),
            "height": float(h),
            "confidence": round(float(conf * 100), 2)
        })

    return detections


# =====================
# API Endpoint
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model_name = request.form.get("model", "yolo11")
    if model_name not in MODELS:
        return jsonify({"error": "Invalid model"}), 400

    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    input_tensor = preprocess(img)

    session = MODELS[model_name]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    detections = postprocess(outputs, *img.size)

    return jsonify({
        "model": model_name,
        "is_bayam": len(detections) > 0,
        "confidence": max([d["confidence"] for d in detections], default=0.0),
        "detections": detections
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
