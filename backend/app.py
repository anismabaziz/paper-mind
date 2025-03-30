from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
import os
import uuid

load_dotenv()
app = Flask(__name__)

CORS(app)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

BUCKET_NAME = "papermind-pdf"


@app.route("/health", methods=["GET"])
def get_health():
    return jsonify({"response": "OK"}), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No File Provided"}), 400
    file = request.files["file"]
    file_ext = os.path.splitext(file.filename)[1]

    unique_filename = f"{uuid.uuid4().hex}{file_ext}"

    file_content = file.read()

    try:
        response = supabase.storage.from_(BUCKET_NAME).upload(
            unique_filename,
            file_content,
            file_options={"content-type": "application/pdf"},
        )
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)
        return jsonify(
            {
                "message": "File uploaded successfully",
                "url": public_url,
                "filename": unique_filename,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=3000)
