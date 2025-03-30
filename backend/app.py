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
        supabase.storage.from_(BUCKET_NAME).upload(
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
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text_chunks,
        )
        vector_embeddings = [embedding.values for embedding in result.embeddings]

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
            file for file in response if file["name"] != ".emptyFolderPlaceholder"
        ]
        return jsonify({"files": filtered_files}), 200
    except Exception as e:
        return jsonify({"error": f"Error fetching files {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=3000)
