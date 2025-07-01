from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import uuid
import io
import re
import pymupdf

load_dotenv()
app = Flask(__name__)

CORS(app)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supbase_service_role = os.getenv("SUPABASE_SERVICE_ROLE")

supabase: Client = create_client(supabase_url, supbase_service_role)

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
        # upload file
        supabase.storage.from_(BUCKET_NAME).upload(
            unique_filename,
            file_content,
            file_options={"content-type": "application/pdf"},
        )
        files = supabase.storage.from_(BUCKET_NAME).list()
        file = next((file for file in files if file["name"] == unique_filename), None)
        file["url"] = supabase.storage.from_(BUCKET_NAME).get_public_url(
            unique_filename
        )

        # create file record
        file_dict = {"filename": file["name"]}
        file["name"]
        supabase.table("paper-mind_files").insert(file_dict).execute()

        return jsonify(
            {
                "message": "File uploaded successfully",
                "file": file,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/file/is-processed", methods=["POST"])
def check_processed():
    # get filename from body
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    # fetch file
    response = (
        supabase.table("paper-mind_files")
        .select("*")
        .eq("filename", filename)
        .execute()
    )
    file = response.data[0]
    print(file)

    return {"is_processed": file["is_processed"]}


@app.route("/process-file", methods=["POST"])
def process_file():
    try:
        # getting filename
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        # getting file
        response = supabase.storage.from_(BUCKET_NAME).download(filename)
        if not response:
            return jsonify({"error": "Failed to fetch file"}), 400
        pdf_stream = io.BytesIO(response)

        print("hello world")

        # extracting text
        extracted_text = []
        with pymupdf.open("pdf", pdf_stream) as doc:
            for page in doc:
                text = page.get_text()
                text = re.sub(r"\n+", " ", text)
                text = re.sub(r"\s{2,}", " ", text)
                extracted_text.append(text.strip())
        cleaned_text = " ".join(extracted_text)

        # splitting text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([cleaned_text])
        text_chunks = [doc.page_content for doc in texts]

        # embedding text
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        batch_size = 100
        vector_embeddings = []

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i : i + batch_size]
            result = client.models.embed_content(
                model="models/text-embedding-004",
                contents=batch,
            )
            vector_embeddings.extend(
                [embedding.values for embedding in result.embeddings]
            )

        # mapping vectors
        vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "content": text_chunks[i],
                    "pdf_name": filename,
                    "chunk_index": i,
                },
            }
            for i, embedding in enumerate(vector_embeddings)
        ]

        # storing vectors in pinecone
        pc = Pinecone(os.getenv("PINECONE_API_KEY"))
        index_name = "pdf-index"
        index = pc.Index(index_name)
        index.upsert(vectors)

        # create conversation
        file_response = (
            supabase.table("paper-mind_files")
            .select("*")
            .eq("filename", filename)
            .execute()
        )
        file = file_response.data[0]
        conversation_dict = {"file_id": file["id"]}
        supabase.table("paper-mind_conversations").insert(conversation_dict).execute()

        # set processed to true
        supabase.table("paper-mind_files").update({"is_processed": True}).eq(
            "filename", filename
        ).execute()

        return jsonify({"message": "PDF processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete-embeddings", methods=["GET"])
def delete_embeddings():
    pc = Pinecone(os.getenv("PINECONE_API_KEY"))
    index_name = "pdf-index"
    index = pc.Index(index_name)
    index.delete(delete_all=True)
    return jsonify({"message": "Embeddings Deleted"}), 200


@app.route("/response", methods=["POST"])
def get_response():
    try:
        # get query
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query is required"}), 400

        # embed query
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=query,
        )
        embedding = [embedding.values for embedding in result.embeddings][0]

        # perform similarity search
        pc = Pinecone(os.getenv("PINECONE_API_KEY"))
        index_name = "pdf-index"
        index = pc.Index(index_name)
        search_results = index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True,
        )
        retrieved_texts = [
            match["metadata"]["content"] for match in search_results["matches"]
        ]

        context = "\n".join(retrieved_texts)

        # generate response
        system_instruction = (
            "You must only answer questions based on the provided context. "
            "If the context does not contain the answer, say 'I don't know based on the given context.' "
            "Do not use any outside knowledge."
        )
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            contents=[
                f"Context: {context}",
                query,
            ],
        )

        return jsonify({"results": result.text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files", methods=["GET"])
def get_files():
    try:
        response = supabase.storage.from_(BUCKET_NAME).list()
        filtered_files = [
            {
                **file,
                "url": supabase.storage.from_(BUCKET_NAME).get_public_url(file["name"]),
            }
            for file in response
            if file["name"] != ".emptyFolderPlaceholder"
        ]
        return jsonify({"files": filtered_files}), 200
    except Exception as e:
        return jsonify({"error": f"Error fetching files {str(e)}"}), 500


@app.route("/files/remove", methods=["DELETE"])
def removeFile():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "File path required"}), 400
    supabase.storage.from_(BUCKET_NAME).remove([path])
    return jsonify({"message": "File deleted successfully"}), 200


@app.route("/files/check-processing", methods=["GET"])
def check_processing():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "File name required"}), 400

    pc = Pinecone(os.getenv("PINECONE_API_KEY"))
    index_name = "pdf-index"
    index = pc.Index(index_name)
    query_results = index.query(
        vector=[0.0] * 768,
        top_k=1,
        include_metadata=True,
        filter={"filename": filename},
    )
    print(query_results)
    return jsonify({"processed": True}), 200


if __name__ == "__main__":
    app.run(debug=True, port=3000)
