from flask import Flask, request, send_file, jsonify
import io
import torch
from TTS.api import TTS

app = Flask(__name__)

# Load XTTS-v2
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@app.route("/tts", methods=["POST"])
def tts_route():
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        output_path = "voice.wav"

        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=None,  # using default model voice
            language="en"
        )

        return send_file(output_path, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "XTTS-v2 TTS server running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
