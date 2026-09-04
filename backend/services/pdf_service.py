"""
    PDF parser: the first implementation behind the document parser seam.
"""

import io
import re

import pymupdf


class PDFParser:
    @staticmethod
    def extract_text(pdf_content):
        # Flat text for backward compatibility: join pages with space and
        # normalize newlines to spaces so legacy callers see a single line.
        # Page-aware callers should use extract_pages which keeps row breaks.
        pages = PDFParser.extract_pages(pdf_content)
        flat = " ".join(pages)
        flat = flat.replace("\n", " ")
        flat = re.sub(r"\s{2,}", " ", flat)
        return flat.strip()

    @staticmethod
    def extract_pages(pdf_content) -> list[str]:
        """
            Extract one string per page.

            Table rows are kept as separate lines (newlines preserved) so a
            chunker can keep row boundaries. Repeating headers/footers are not
            stripped here; that is the Docling layer's job when opted in.
        """
        pdf_stream = io.BytesIO(pdf_content)
        pages: list[str] = []
        with pymupdf.open("pdf", pdf_stream) as doc:
            for page in doc:
                text = page.get_text()
                # Preserve line breaks for table rows: collapse only spaces/tabs
                # on each line, and collapse consecutive blank lines.
                # Keep single '\n' as row separator.
                lines = []
                for raw_line in text.split("\n"):
                    # collapse horizontal whitespace, keep empty lines as empty
                    cleaned = re.sub(r"[ \t]+", " ", raw_line).strip()
                    lines.append(cleaned)
                # Remove leading/trailing empty lines, collapse runs of empties
                normalized_lines: list[str] = []
                for line in lines:
                    if line == "" and (not normalized_lines or normalized_lines[-1] == ""):
                        continue
                    normalized_lines.append(line)
                # Strip trailing empty
                while normalized_lines and normalized_lines[-1] == "":
                    normalized_lines.pop()
                while normalized_lines and normalized_lines[0] == "":
                    normalized_lines.pop(0)
                page_text = "\n".join(normalized_lines)
                pages.append(page_text)
        return pages
