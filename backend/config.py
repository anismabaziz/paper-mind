import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone
from google import genai

load_dotenv()

# App Constants
BUCKET_NAME = "papermind-pdf"
INDEX_NAME = "pdf-index"

# Supabase Configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_secret_key = os.getenv("SUPABASE_SECRET_KEY")
supabase: Client = create_client(supabase_url, supabase_secret_key)

# Pinecone Configuration
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vector_index = pc.Index(INDEX_NAME)

# Google GenAI Configuration
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash"
