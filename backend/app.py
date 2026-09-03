import uuid
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from config import INDEX_NAME
from db import repository
from storage import storage
from services.pdf_service import PDFService
from services.ai_service import AIService
from services.vector_service import VectorService
from services.auth_service import (
    hash_password,
    issue_token,
    require_auth,
    verify_password,
)

config.validate()

app = Flask(__name__)
CORS(app)


def file_url(filename):
    # Absolute, like the old hosted storage URLs, so the frontend can use
    # the value directly in an iframe pointed at the API host.
    return f"{request.host_url.rstrip('/')}{storage.url(filename)}"


@app.route("/health", methods=["GET"])
def get_health():
    return jsonify({"response": "OK"}), 200


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if repository.get_user_by_email(email):
        return jsonify({"error": "Email is already registered"}), 409

    user = repository.create_user(email, hash_password(password))
    return jsonify({"message": "User registered", "token": issue_token(email)}), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = repository.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({"token": issue_token(email)}), 200


@app.route("/storage/<path:filename>", methods=["GET"])
@require_auth
def download_file(filename):
    return send_from_directory(config.STORAGE_DIR, filename)


@app.route("/upload", methods=["POST"])
@require_auth
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No File Provided"}), 400

    file = request.files["file"]
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_content = file.read()

    try:
        # Store the bytes, then create the DB record
        storage.save(unique_filename, file_content)
        file_record = repository.create_file(unique_filename)

        return jsonify({
            "message": "File uploaded successfully",
            "file": {
                "id": file_record["id"],
                "name": unique_filename,
                "url": file_url(unique_filename),
            }
        })
    except Exception as e:
        storage.delete(unique_filename)
        return jsonify({"error": str(e)}), 500


@app.route("/file/is-processed", methods=["POST"])
@require_auth
def check_processed():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    file = repository.get_file(filename)
    if not file:
        return jsonify({"error": "File not found"}), 404

    return jsonify({"is_processed": file["is_processed"]})


@app.route("/process-file", methods=["POST"])
@require_auth
def process_file():
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        # 1. Download & Extract
        if not storage.exists(filename):
            return jsonify({"error": "Failed to fetch file"}), 400
        file_content = storage.open(filename)

        text = PDFService.extract_text(file_content)
        chunks = PDFService.split_text(text)

        # 2. Embed & Vectorize
        embeddings = AIService.get_embeddings(chunks)
        VectorService.upsert_vectors(embeddings, chunks, filename)

        # 3. Create Conversation (reuse one if the file is re-processed)
        file_record = repository.get_file(filename)
        if file_record and not repository.get_conversation_id(file_record["id"]):
            repository.create_conversation(file_record["id"])

        # 4. Mark as Processed
        repository.set_processed(filename, True)

        return jsonify({"message": "PDF processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/response", methods=["POST"])
@require_auth
def get_response():
    try:
        data = request.get_json()
        query = data.get("query")
        filename = data.get("filename")

        if not query or not filename:
            return jsonify({"error": "Query and Filename are required"}), 400

        # 1. Context Retrieval
        file_record = repository.get_file(filename)
        if not file_record:
            return jsonify({"error": "File not found"}), 404

        conversation_id = repository.get_conversation_id(file_record["id"])
        if not conversation_id:
            return jsonify({"error": "Conversation not found"}), 404

        # Store User Message
        repository.add_message(conversation_id, "user", query)

        # 2. Vector Search
        query_embedding = AIService.get_embeddings(query)[0]
        context_chunks = VectorService.query_vectors(query_embedding, filename)
        context = "\n".join(context_chunks)

        # 3. LLM Generation
        response_text = AIService.generate_response(query, context)

        # Store Bot Message
        repository.add_message(conversation_id, "bot", response_text)

        return jsonify({"results": response_text}), 200
    except Exception as e:
        print(f"/response error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/messages", methods=["GET"])
@require_auth
def get_messages():
    try:
        filename = request.args.get("filename")
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        file_record = repository.get_file(filename)
        if not file_record:
            return jsonify({"messages": []}), 200

        conversation_id = repository.get_conversation_id(file_record["id"])
        if not conversation_id:
            return jsonify({"messages": []}), 200

        messages = repository.get_messages(conversation_id)
        return jsonify({"messages": messages}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files", methods=["GET"])
@require_auth
def get_files():
    try:
        db_files = repository.list_files()
        storage_items = storage.list()
        storage_map = {item["name"]: item for item in storage_items}

        enriched_files = []
        for db_file in db_files:
            filename = db_file["filename"]
            storage_item = storage_map.get(filename)

            size = storage_item["size"] if storage_item else 0

            enriched_files.append({
                "id": db_file["id"],
                "name": filename,
                "url": file_url(filename),
                "is_processed": db_file["is_processed"],
                "metadata": {
                    "size": size,
                    "content_type": "application/pdf"
                }
            })

        return jsonify({"files": enriched_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files/remove", methods=["DELETE"])
@require_auth
def remove_file():
    try:
        filename = request.args.get("path")
        if not filename:
            return jsonify({"error": "File path required"}), 400

        # Always remove vectors and bytes, even if DB metadata is missing.
        VectorService.delete_by_filename(filename)
        storage.delete(filename)

        file_record = repository.get_file(filename)
        if file_record:
            conversation_id = repository.get_conversation_id(file_record["id"])
            if conversation_id:
                repository.delete_messages(conversation_id)
                repository.delete_conversation(conversation_id)

            repository.delete_file(file_record["id"])

        return jsonify({"message": "File and all its data deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete-embeddings", methods=["POST"])
@require_auth
def delete_embeddings():
    VectorService.delete_all()
    return jsonify({"message": "Embeddings Deleted"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=3000)
