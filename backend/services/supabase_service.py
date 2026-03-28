from flask import jsonify
from config import supabase, BUCKET_NAME

class SupabaseService:
    @staticmethod
    def upload_to_storage(filename, content):
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            content,
            file_options={"content-type": "application/pdf"},
        )
        return supabase.storage.from_(BUCKET_NAME).get_public_url(filename)

    @staticmethod
    def get_storage_list():
        return supabase.storage.from_(BUCKET_NAME).list()

    @staticmethod
    def remove_from_storage(filename):
        return supabase.storage.from_(BUCKET_NAME).remove([filename])

    @staticmethod
    def download_from_storage(filename):
        return supabase.storage.from_(BUCKET_NAME).download(filename)

    @staticmethod
    def create_file_record(filename):
        file_dict = {"filename": filename}
        return supabase.table("paper-mind_files").insert(file_dict).execute()

    @staticmethod
    def get_file_by_name(filename):
        response = supabase.table("paper-mind_files").select("*").eq("filename", filename).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def get_all_files():
        response = supabase.table("paper-mind_files").select("*").execute()
        return response.data

    @staticmethod
    def update_processing_status(filename, status=True):
        return supabase.table("paper-mind_files").update({"is_processed": status}).eq("filename", filename).execute()

    @staticmethod
    def delete_file_record(file_id):
        return supabase.table("paper-mind_files").delete().eq("id", file_id).execute()

    @staticmethod
    def create_conversation(file_id):
        conversation_dict = {"file_id": file_id}
        return supabase.table("paper-mind_conversations").insert(conversation_dict).execute()

    @staticmethod
    def get_conversation_by_file_id(file_id):
        response = (
            supabase.table("paper-mind_conversations")
            .select("id")
            .eq("file_id", file_id)
            .execute()
        )
        return response.data[0]["id"] if response.data else None

    @staticmethod
    def delete_conversation(conversation_id):
        return supabase.table("paper-mind_conversations").delete().eq("id", conversation_id).execute()

    @staticmethod
    def store_message(conversation_id, sender, text):
        return supabase.table("paper-mind_messages").insert({
            "conversation_id": conversation_id,
            "sender": sender,
            "text": text
        }).execute()

    @staticmethod
    def get_messages(conversation_id):
        response = (
            supabase.table("paper-mind_messages")
            .select("id, text, sender, created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        return response.data

    @staticmethod
    def delete_messages(conversation_id):
        return supabase.table("paper-mind_messages").delete().eq("conversation_id", conversation_id).execute()
