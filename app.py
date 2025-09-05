#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template, request, url_for
import joblib


def load_pipeline(model_path: str):
    resolved = Path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found at {resolved}")
    return joblib.load(resolved)


MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
pipeline = None


def create_app() -> Flask:
    global pipeline
    app = Flask(__name__)

    # Lazy-load pipeline on first request to speed up startup errors surfacing nicely
    def ensure_loaded() -> None:
        global pipeline
        if pipeline is None:
            pipeline = load_pipeline(MODEL_PATH)

    @app.route("/", methods=["GET"]) 
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"]) 
    def predict_form():
        ensure_loaded()
        text = request.form.get("text", "").strip()
        if not text:
            return render_template("index.html", error="Please paste some text to classify."), 400
        pred = pipeline.predict([text])[0]
        proba = None
        try:
            proba = float(pipeline.predict_proba([text])[0][int(pred)])
        except Exception:
            proba = None

        label_map = {0: "fake", 1: "true"}
        label = label_map.get(int(pred), str(pred))
        return render_template("index.html", input_text=text, prediction=label, probability=proba)

    @app.route("/api/predict", methods=["POST"]) 
    def predict_api():
        ensure_loaded()
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Missing 'text' in JSON body"}), 400
        pred = int(pipeline.predict([text])[0])
        resp: Dict[str, Any] = {"prediction": pred, "label": "true" if pred == 1 else "fake"}
        try:
            probs = pipeline.predict_proba([text])[0].tolist()
            resp["probabilities"] = {"fake": float(probs[0]), "true": float(probs[1])}
            resp["confidence"] = float(probs[pred])
        except Exception:
            pass
        return jsonify(resp)

    @app.route("/health", methods=["GET"]) 
    def health():
        # Don't force model load for health check; app can be up before model is mounted
        return jsonify({"status": "ok", "model_loaded": pipeline is not None})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

