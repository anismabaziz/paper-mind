import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone
from google import genai
from groq import Groq

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

# Mode Selector
MODE = os.getenv("MODE", "google").lower()

# Google GenAI Configuration
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash"

# Groq Configuration
groq_client = None
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
