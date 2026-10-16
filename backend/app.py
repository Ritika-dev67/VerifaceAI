from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # required, since your HTML page runs on a different origin

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    # TODO: load image, run your model here
    # e.g. prediction, confidence = model_predict(file)

    return jsonify({
        "prediction": "fake",      # "real" or "fake"
        "confidence": 92.5
    })

@app.route("/predict-video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    # TODO: extract frames, run model on each frame

    return jsonify({
        "prediction": "Fake",
        "confidence": 88.0,
        "fake_frames": 42,
        "real_frames": 8
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)