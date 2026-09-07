from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from docx import Document
from fastapi import UploadFile
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text_chunks(
    file: UploadFile,
    chunk_size: int = 2000,
) -> Iterator[tuple[str, int]]:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported file types are: PDF, DOCX and TXT")

    content = file.file.read()
    text = _extract_text(content, extension)

    if not text.strip():
        raise ValueError("The uploaded file does not contain readable text")

    for offset in range(0, len(text), chunk_size):
        chunk = text[offset:offset + chunk_size]
        if chunk.strip():
            yield chunk, offset


def _extract_text(content: bytes, extension: str) -> str:
    if extension == ".txt":
        return content.decode("utf-8")

    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    document = Document(BytesIO(content))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)