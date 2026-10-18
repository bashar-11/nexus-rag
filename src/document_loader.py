import os
import re
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, logger


def _clean_text(text: str) -> str:
    """
    Normalise raw extracted text before it is chunked and embedded.

    Whitespace collapsed here reduces token waste and stops the chunker from
    splitting on meaningless boundaries such as indentation or repeated
    newlines.
    """
    if not text:
        return ""
    # Collapse runs of 3+ newlines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces and tabs into one space
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _load_txt(file_path: str, file_name: str) -> List[Document]:
    """
    Read a text file, tolerating non-UTF-8 bytes.

    LangChain's TextLoader swallows UnicodeDecodeError and re-raises it as a
    RuntimeError, and its constructor has no `errors=` parameter. So a naive
    `except UnicodeDecodeError` around it never fires, and the documented
    fallback cannot be expressed through TextLoader at all. Read the bytes here
    instead and substitute the undecodable characters.
    """
    try:
        return TextLoader(file_path, encoding="utf-8").load()
    except (UnicodeDecodeError, RuntimeError, OSError):
        logger.warning(
            f"{file_name} is not valid UTF-8. Undecodable characters will be "
            "replaced, so a little text may be lost."
        )
        raw = Path(file_path).read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": file_name})]


def load_document(file_path: str) -> List[Document]:
    """
    Load a PDF or TXT file into LangChain Documents.

    Every Document carries its source file name and, for PDFs, its page
    number, so answers can be cited back to the exact place they came from.
    Chunks too small to contain meaning are discarded: they add noise to
    retrieval and cost embedding calls for no benefit.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = os.path.basename(file_path)
    docs: List[Document] = []

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        for i, doc in enumerate(raw_docs):
            cleaned_text = _clean_text(doc.page_content)
            # Skip pages that are effectively empty after cleaning
            # (scanned-image pages, blank pages, stray headers)
            if len(cleaned_text) > 10:
                doc.page_content = cleaned_text
                doc.metadata["source"] = file_name
                doc.metadata["page"] = doc.metadata.get("page", i) + 1
                docs.append(doc)

        if not docs:
            logger.warning(
                f"No usable text found in {file_name}. The PDF may be "
                "scanned images with no embedded text layer."
            )

    elif file_path.endswith(".txt"):
        raw_docs = _load_txt(file_path, file_name)

        for doc in raw_docs:
            cleaned_text = _clean_text(doc.page_content)
            if len(cleaned_text) > 10:
                doc.page_content = cleaned_text
                doc.metadata["source"] = file_name
                docs.append(doc)
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")

    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.

    The chunker recurses through the separators below from most structural to
    least, so it prefers to break on paragraph and sentence boundaries before
    it ever falls back to a space. Arabic punctuation is included because the
    app is bilingual: "؟", "،" and "؛" are the Arabic question mark, comma and
    semicolon, and splitting on those keeps Arabic sentences intact.

    Each chunk is stamped with a chunk_index (counted per source file) so a
    citation can point at the exact position in the original document.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "؟ ",  # Arabic question mark
            "؟",
            "؛ ",  # Arabic semicolon
            "؛",
            "، ",  # Arabic comma
            "،",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )
    chunks = text_splitter.split_documents(documents)

    # Stamp a per-source chunk counter so citations can identify position.
    counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk_index"] = counters[source]

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks