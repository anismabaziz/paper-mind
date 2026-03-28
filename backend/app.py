from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import os

from config import BUCKET_NAME, supabase
from services.supabase_service import SupabaseService
from services.pdf_service import PDFService
from services.ai_service import AIService
from services.vector_service import VectorService

app = Flask(__name__)
CORS(app)

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
        # Upload to Storage
        public_url = SupabaseService.upload_to_storage(unique_filename, file_content)
        
        # Create DB record
        SupabaseService.create_file_record(unique_filename)
        
        return jsonify({
            "message": "File uploaded successfully",
            "file": {
                "name": unique_filename,
                "url": public_url
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/file/is-processed", methods=["POST"])
def check_processed():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    file = SupabaseService.get_file_by_name(filename)
    if not file:
        return jsonify({"error": "File not found"}), 404

    return jsonify({"is_processed": file["is_processed"]})

@app.route("/process-file", methods=["POST"])
def process_file():
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        # 1. Download & Extract
        file_content = SupabaseService.download_from_storage(filename)
        if not file_content:
            return jsonify({"error": "Failed to fetch file"}), 400
            
        text = PDFService.extract_text(file_content)
        chunks = PDFService.split_text(text)

        # 2. Embed & Vectorize
        embeddings = AIService.get_embeddings(chunks)
        VectorService.upsert_vectors(embeddings, chunks, filename)

        # 3. Create Conversation
        file_record = SupabaseService.get_file_by_name(filename)
        SupabaseService.create_conversation(file_record["id"])

        # 4. Mark as Processed
        SupabaseService.update_processing_status(filename, True)

        return jsonify({"message": "PDF processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/response", methods=["POST"])
def get_response():
    try:
        data = request.get_json()
        query = data.get("query")
        filename = data.get("filename")
        
        if not query or not filename:
            return jsonify({"error": "Query and Filename are required"}), 400

        # 1. Context Retrieval
        file_record = SupabaseService.get_file_by_name(filename)
        if not file_record:
            return jsonify({"error": "File not found"}), 404
            
        conversation_id = SupabaseService.get_conversation_by_file_id(file_record["id"])
        if not conversation_id:
            return jsonify({"error": "Conversation not found"}), 404

        # Store User Message
        SupabaseService.store_message(conversation_id, "user", query)

        # 2. Vector Search
        query_embedding = AIService.get_embeddings(query)[0]
        context_chunks = VectorService.query_vectors(query_embedding, filename)
        context = "\n".join(context_chunks)

        # 3. LLM Generation
        response_text = AIService.generate_response(query, context)

        # Store Bot Message
        SupabaseService.store_message(conversation_id, "bot", response_text)

        return jsonify({"results": response_text}), 200
    except Exception as e:
        print(f"/response error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/messages", methods=["GET"])
def get_messages():
    try:
        filename = request.args.get("filename")
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        file_record = SupabaseService.get_file_by_name(filename)
        if not file_record:
            return jsonify({"messages": []}), 200
            
        conversation_id = SupabaseService.get_conversation_by_file_id(file_record["id"])
        if not conversation_id:
            return jsonify({"messages": []}), 200

        messages = SupabaseService.get_messages(conversation_id)
        return jsonify({"messages": messages}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/files", methods=["GET"])
def get_files():
    try:
        db_files = SupabaseService.get_all_files()
        storage_items = SupabaseService.get_storage_list()
        storage_map = {item["name"]: item for item in storage_items}
        
        enriched_files = []
        for db_file in db_files:
            filename = db_file["filename"]
            storage_item = storage_map.get(filename)
            
            size = 0
            if storage_item:
                size = storage_item.get("metadata", {}).get("size") or storage_item.get("size", 0)
            
            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename) # Fallback if URL needed
            
            enriched_files.append({
                "id": db_file["id"],
                "name": filename,
                "url": public_url,
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
def remove_file():
    try:
        filename = request.args.get("path")
        if not filename:
            return jsonify({"error": "File path required"}), 400

        # Always remove embeddings for the target file, even if DB metadata is missing.
        VectorService.delete_by_filename(filename)

        file_record = SupabaseService.get_file_by_name(filename)
        if file_record:
            # 1. Delete Messages & Conversations
            conversation_id = SupabaseService.get_conversation_by_file_id(file_record["id"])
            if conversation_id:
                SupabaseService.delete_messages(conversation_id)
                SupabaseService.delete_conversation(conversation_id)

            # 2. Delete DB Record
            SupabaseService.delete_file_record(file_record["id"])

        # 3. Remove from Storage
        SupabaseService.remove_from_storage(filename)

        return jsonify({"message": "File and all its data deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete-embeddings", methods=["GET"])
def delete_embeddings():
    VectorService.delete_all()
    return jsonify({"message": "Embeddings Deleted"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=3000)
