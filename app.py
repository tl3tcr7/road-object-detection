"""
ROD Detection + OCR — Flask deployment
--------------------------------------
Run:   python app.py
Open:  http://localhost:5000
"""

import base64
import os
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")      # path to your trained weights
OCR_LANGS = os.environ.get("OCR_LANGS", "ar,en").split(",")
MAX_UPLOAD_MB = 15
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
MIN_CROP_SIDE = 12          # skip crops smaller than this (px) for OCR
DEFAULT_CONF = 0.25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ─── Load YOLO once at startup ────────────────────────────────────────────────
print(f"🎯 Loading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ YOLO ready")

# ─── EasyOCR is heavy → load lazily on first use ──────────────────────────────
_reader = None
_reader_lock = threading.Lock()


def get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                import torch

                print("🔤 Initializing EasyOCR...")
                _reader = easyocr.Reader(OCR_LANGS, gpu=torch.cuda.is_available())
                print("✅ EasyOCR ready")
    return _reader


# ─── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def to_base64_jpg(bgr_image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", default_conf=DEFAULT_CONF)


@app.route("/health")
def health():
    return jsonify(status="ok", model=os.path.basename(MODEL_PATH))


@app.route("/api/predict", methods=["POST"])
def predict():
    # ── Validate upload ──
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image uploaded."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Use JPG, PNG, WEBP or BMP."), 400

    try:
        conf = float(request.form.get("conf", DEFAULT_CONF))
    except ValueError:
        conf = DEFAULT_CONF
    conf = min(max(conf, 0.01), 0.99)
    use_ocr = request.form.get("ocr", "false").lower() == "true"

    # ── Decode image ──
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="The file could not be read as an image."), 400

    # ── YOLO detection ──
    result = model.predict(source=img, conf=conf, verbose=False)[0]
    annotated = result.plot()  # BGR numpy array with boxes drawn

    reader = get_reader() if use_ocr else None
    detections = []
    h, w = result.orig_img.shape[:2]

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        cls_id = int(box.cls[0])

        det = {
            "id": i + 1,
            "class": result.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "box": [x1, y1, x2, y2],
            "text": None,
        }

        # ── OCR on the cropped detection ──
        if reader is not None:
            crop = result.orig_img[y1:y2, x1:x2]
            if crop.size > 0 and min(crop.shape[:2]) >= MIN_CROP_SIDE:
                try:
                    texts = reader.readtext(crop, detail=0)
                    det["text"] = [t for t in texts if t.strip()]
                except Exception as exc:  # keep API alive if OCR fails on one crop
                    print(f"OCR error on detection {i + 1}: {exc}")
                    det["text"] = []
            else:
                det["text"] = []

        detections.append(det)

    # ── Summary: count per class ──
    summary = {}
    for d in detections:
        summary[d["class"]] = summary.get(d["class"], 0) + 1

    return jsonify(
        image=to_base64_jpg(annotated),
        width=w,
        height=h,
        count=len(detections),
        ocr_enabled=use_ocr,
        summary=summary,
        detections=detections,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Image is larger than {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    # host 0.0.0.0 so it's reachable from outside a container / VM
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
