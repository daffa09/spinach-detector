from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# =====================
# CONFIG
# =====================
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASS_NAMES = {0: "bayam"}

MODELS = {
    "yolo9": ort.InferenceSession("models/yolo9.onnx", providers=["CPUExecutionProvider"]),
    "yolo11": ort.InferenceSession("models/yolo11.onnx", providers=["CPUExecutionProvider"]),
}

# =====================
# LETTERBOX
# =====================
def letterbox(image, new_size=640):
    w, h = image.size
    scale = min(new_size / w, new_size / h)

    nw, nh = int(w * scale), int(h * scale)
    image_resized = image.resize((nw, nh))

    new_image = Image.new("RGB", (new_size, new_size), (114, 114, 114))
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2

    new_image.paste(image_resized, (pad_x, pad_y))

    return new_image, scale, pad_x, pad_y


# =====================
# PREPROCESS
# =====================
def preprocess(image):
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


# =====================
# NMS
# =====================
def nms(boxes, scores, iou_thr):
    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break

        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])

        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou < iou_thr]

    return keep


# =====================
# POSTPROCESS
# =====================
def postprocess(outputs, scale, pad_x, pad_y, orig_w, orig_h):
    preds = np.squeeze(outputs[0], axis=0)

    boxes, scores = [], []

    for p in preds:
        x, y, w, h = p[:4]
        obj = p[4]
        cls_conf = p[5]

        conf = obj * cls_conf
        if conf < CONF_THRESHOLD:
            continue

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    keep = nms(boxes, scores, IOU_THRESHOLD)

    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]

        # Remove padding & scale back
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Normalize
        x1n = max(0, x1 / orig_w)
        y1n = max(0, y1 / orig_h)
        wn = (x2 - x1) / orig_w
        hn = (y2 - y1) / orig_h

        detections.append({
            "class": "bayam",
            "confidence": round(float(scores[i] * 100), 2),
            "bbox_normalized": {
                "x": float(x1n),
                "y": float(y1n),
                "width": float(wn),
                "height": float(hn),
            }
        })

    return detections


# =====================
# API
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    model_name = request.form.get("model", "yolo11")
    session = MODELS[model_name]

    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    orig_w, orig_h = img.size

    img_lb, scale, pad_x, pad_y = letterbox(img)
    input_tensor = preprocess(img_lb)

    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    detections = postprocess(outputs, scale, pad_x, pad_y, orig_w, orig_h)

    return jsonify({
        "model": model_name,
        "count": len(detections),
        "is_bayam": len(detections) > 0,
        "confidence": max([d["confidence"] for d in detections], default=0),
        "detections": detections
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
