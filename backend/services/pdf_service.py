"""
    PDF parser: the first implementation behind the document parser seam.
"""

import io
import re

import pymupdf


class PDFParser:
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
