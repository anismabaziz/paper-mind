from config import groq_client, GROQ_MODEL

class GroqService:
    @staticmethod
    def generate_response(query: str, context: str) -> str:
        if not groq_client:
            raise ValueError("Groq client is not initialized. Please ensure GROQ_API_KEY is configured in your .env file.")

        system_instruction = (
            "You must only answer questions based on the provided context. "
            "If the context does not contain the answer, say 'I don't know based on the given context.' "
            "Do not use any outside knowledge."
        )

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuery: {query}",
                }
            ],
            model=GROQ_MODEL,
        )

        result = chat_completion.choices[0].message.content
        return result or "I don't know based on the given context."
