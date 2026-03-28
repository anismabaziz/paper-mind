import pymupdf
import re
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFService:
    @staticmethod
    def extract_text(pdf_content):
        pdf_stream = io.BytesIO(pdf_content)
        extracted_text = []
        with pymupdf.open("pdf", pdf_stream) as doc:
            for page in doc:
                text = page.get_text()
                text = re.sub(r"\n+", " ", text)
                text = re.sub(r"\s{2,}", " ", text)
                extracted_text.append(text.strip())
        return " ".join(extracted_text)

    @staticmethod
    def split_text(text, chunk_size=600, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([text])
        return [doc.page_content for doc in texts]
