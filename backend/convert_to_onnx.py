from ultralytics import YOLO

models = [
    "yolo11.pt",
    "yolo9.pt"
]

for m in models:
    model = YOLO(m)
    model.export(
        format="onnx",
        opset=12,
        simplify=True,
        dynamic=True
    )

print("✅ Semua model berhasil dikonversi ke ONNX")
