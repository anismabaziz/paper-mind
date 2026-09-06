"""Module docstring."""

import json
import os
import time
import uuid

from flask import (
    Flask,
    Response,
    g,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)
from flask_cors import CORS

import settings
from db import repository
from storage import storage
from services.parsing.document_parser import DocumentParser
from services.llm.ai_service import AIService
from services.retrieval.vector_service import VectorService
from services.accounts.auth_service import (
    hash_password,
    is_demo_mode,
    issue_token,
    require_auth,
    verify_password,
)
from services.accounts.chat_settings_service import (
    DEMO_EMAIL,
    SUPPORTED_MODELS,
    SettingsError,
    mask_key,
    validate,
    verify_api_key,
)
from services.accounts.secrets_service import decrypt_api_key, encrypt_api_key

settings.validate()

app = Flask(__name__)
CORS(app)


def _timed_call(func, *args, **kwargs):
    """Run ``func`` and return (result, elapsed_seconds) for phase timing."""
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


def file_url(filename):
    # Absolute, like the old hosted storage URLs, so the frontend can use
    # the value directly in an iframe pointed at the API host.
    """Do file url."""
    return f"{request.host_url.rstrip('/')}{storage.url(filename)}"


@app.route("/health", methods=["GET"])
def get_health():
    """Do get health."""
    return jsonify({"response": "OK"}), 200


@app.route("/auth/register", methods=["POST"])
def register():
    """Do register."""
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
    """Do login."""
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
    """Do download file."""
    return send_from_directory(settings.get_settings().storage.storage_dir, filename)


@app.route("/upload", methods=["POST"])
@require_auth
def upload_file():
    """Do upload file."""
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

        return jsonify(
            {
                "message": "File uploaded successfully",
                "file": {
                    "id": file_record["id"],
                    "name": unique_filename,
                    "url": file_url(unique_filename),
                },
            }
        )
    except Exception as e:
        storage.delete(unique_filename)
        return jsonify({"error": str(e)}), 500


@app.route("/file/is-processed", methods=["POST"])
@require_auth
def check_processed():
    """Do check processed."""
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
    """Do process file."""
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        wall_start = time.time()

        # 1. Parse & Extract (format-specific parser, shared chunking)
        if not storage.exists(filename):
            return jsonify({"error": "Failed to fetch file"}), 400
        file_content = storage.open(filename)

        chunk_objs, parse_elapsed = _timed_call(
            DocumentParser.get_chunk_objects, filename, file_content
        )

        # 2. Embed & Vectorize (delete stale vectors first so a retry is idempotent)
        if not chunk_objs:
            return jsonify({"error": "No text extracted from document"}), 400
        # Remove any vectors left from a previous failed attempt
        try:
            VectorService.delete_by_filename(filename)
        except Exception as e:
            print(f"/process-file vector cleanup warning for {filename}: {e}")
        texts = [c.text for c in chunk_objs]
        embeddings, embed_elapsed = _timed_call(AIService.get_embeddings, texts)
        _, upsert_elapsed = _timed_call(
            VectorService.upsert_chunks, embeddings, chunk_objs, filename
        )

        # 3. Create Conversation (reuse one if the file is re-processed)
        file_record = repository.get_file(filename)
        if file_record and not repository.get_conversation_id(file_record["id"]):
            repository.create_conversation(file_record["id"])

        # 4. Mark as Processed
        repository.set_processed(filename, True)

        wall_elapsed = time.time() - wall_start
        print(
            f"/process-file {filename}: {len(chunk_objs)} chunks | "
            f"parse {parse_elapsed:.2f}s embed {embed_elapsed:.2f}s "
            f"upsert {upsert_elapsed:.2f}s total {wall_elapsed:.2f}s"
        )

        return jsonify({"message": "PDF processed"}), 200
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"/process-file failed for {locals().get('filename', '?')}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/response", methods=["POST"])
@require_auth
def get_response():
    """Do get response."""
    data = request.get_json() or {}
    query = data.get("query")
    filename = data.get("filename")

    if not query or not filename:
        return jsonify({"error": "Query and Filename are required"}), 400

    # Chat runs on the requester's own provider settings; without saved
    # settings there is no key to answer with, so the turn is refused here.
    user = _current_user()
    if not user:
        return jsonify({"error": "User not found"}), 401
    stored = repository.get_user_settings(user["id"])
    if not stored:
        return jsonify(
            {"error": "No chat provider configured. Add a provider and API key in Settings."}
        ), 400
    try:
        api_key = decrypt_api_key(stored["encrypted_api_key"])
    except Exception as e:
        print(f"/response decrypt error for user {user['id']}: {e}")
        return jsonify(
            {
                "error": "Stored API key could not be decrypted. "
                "Re-save your provider settings, then try again."
            }
        ), 500
    provider, model = stored["provider"], stored["model"]

    file_record = repository.get_file(filename)
    if not file_record:
        return jsonify({"error": "File not found"}), 404

    conversation_id = repository.get_conversation_id(file_record["id"])
    if not conversation_id:
        return jsonify({"error": "Conversation not found"}), 404

    # Retrieval happens up front: the user message and the sources must be
    # settled before the first token is streamed, so a provider failure can
    # never leave the turn half-recorded.
    try:
        repository.add_message(conversation_id, "user", query)
        query_embedding = AIService.get_embeddings(query)[0]
        sources = VectorService.query_vectors(
            query_embedding, filename, query_text=query
        )
        context = "\n\n".join(source["content"] for source in sources)
    except Exception as e:
        print(f"/response retrieval error: {e}")
        return jsonify({"error": str(e)}), 500

    def event(name, payload):
        """Do event."""
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        """Do generate."""
        fragments = []
        try:
            for token in AIService.stream_response(query, context, provider, model, api_key):
                fragments.append(token)
                yield event("token", {"text": token})
        except Exception as e:
            print(f"/response generation error: {e}")
            failure = (
                "Sorry — the language model is unavailable right now. Please try again."
            )
            repository.add_message(conversation_id, "bot", failure)
            yield event("error", {"error": failure})
            yield event("done", {"done": True, "sources": []})
            return

        answer = (
            "".join(fragments).strip() or "I don't know based on the given context."
        )
        repository.add_message(conversation_id, "bot", answer, sources)
        yield event("done", {"done": True, "sources": sources})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/messages", methods=["GET"])
@require_auth
def get_messages():
    """Do get messages."""
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
    """Do get files."""
    try:
        db_files = repository.list_files()
        storage_items = storage.list()
        storage_map = {item["name"]: item for item in storage_items}

        enriched_files = []
        for db_file in db_files:
            filename = db_file["filename"]
            storage_item = storage_map.get(filename)

            size = storage_item["size"] if storage_item else 0

            enriched_files.append(
                {
                    "id": db_file["id"],
                    "name": filename,
                    "url": file_url(filename),
                    "is_processed": db_file["is_processed"],
                    "metadata": {"size": size, "content_type": "application/pdf"},
                }
            )

        return jsonify({"files": enriched_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files/remove", methods=["DELETE"])
@require_auth
def remove_file():
    """Do remove file."""
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
    """Do delete embeddings."""
    VectorService.delete_all()
    return jsonify({"message": "Embeddings Deleted"}), 200


def _current_user():
    """
    Resolve the user whose settings this request touches.

    Demo mode has no token, so every request operates on the seeded demo
    user; authenticated requests use the token's subject email.
    """
    email = DEMO_EMAIL if is_demo_mode() else getattr(g, "user_email", None)
    if not email:
        return None
    return repository.get_user_by_email(email)


def _settings_payload(stored, plaintext_key=None):
    if not stored:
        return {
            "provider": None,
            "model": None,
            "masked_key": None,
            "supported_models": SUPPORTED_MODELS,
        }
    if plaintext_key is None:
        plaintext_key = decrypt_api_key(stored["encrypted_api_key"])
    return {
        "provider": stored["provider"],
        "model": stored["model"],
        "masked_key": mask_key(plaintext_key),
        "supported_models": SUPPORTED_MODELS,
    }


@app.route("/settings", methods=["GET"])
@require_auth
def get_settings():
    """Do get settings."""
    user = _current_user()
    if not user:
        return jsonify({"error": "User not found"}), 401

    stored = repository.get_user_settings(user["id"])
    try:
        payload = _settings_payload(stored)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(payload), 200


@app.route("/settings", methods=["PUT"])
@require_auth
def save_settings():
    """Do save settings."""
    user = _current_user()
    if not user:
        return jsonify({"error": "User not found"}), 401

    data = request.get_json() or {}
    provider = (data.get("provider") or "").strip().lower()
    model = (data.get("model") or "").strip()
    api_key = data.get("api_key") or ""

    if not api_key:
        return jsonify({"error": "API key is required"}), 400
    try:
        validate(provider, model)
    except SettingsError as e:
        return jsonify({"error": str(e)}), 400

    repository.upsert_user_settings(
        user["id"], provider, model, encrypt_api_key(api_key)
    )
    return jsonify(
        {
            "provider": provider,
            "model": model,
            "masked_key": mask_key(api_key),
            "supported_models": SUPPORTED_MODELS,
        }
    ), 200


@app.route("/settings/verify", methods=["POST"])
@require_auth
def verify_settings():
    """Do verify settings."""
    user = _current_user()
    if not user:
        return jsonify({"error": "User not found"}), 401

    stored = repository.get_user_settings(user["id"])
    if not stored:
        return jsonify({"error": "No chat settings saved yet"}), 400

    try:
        api_key = decrypt_api_key(stored["encrypted_api_key"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ok, error = verify_api_key(stored["provider"], stored["model"], api_key)
    # Provider errors can echo request context; the key must never reach
    # the client even if a provider client leaks it into the message.
    if error:
        error = error.replace(api_key, "••••")
    return jsonify({"ok": ok, "error": error}), 200


if __name__ == "__main__":
    app.run(debug=True, port=3000)
