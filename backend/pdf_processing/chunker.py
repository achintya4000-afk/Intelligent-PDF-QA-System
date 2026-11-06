
def chunk_text(
    extracted_pages: list,
    chunk_size: int = 1000,
    overlap: int = 150
) -> list:
    """
    Divide extracted PDF pages into smaller text chunks.

    Each chunk retains its original PDF page number.
    Chunk size and overlap are measured in characters.
    """

    if not isinstance(extracted_pages, list):
        raise ValueError("Extracted pages must be a list.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    chunks = []
    chunk_id = 0

    for page_data in extracted_pages:
        page_number = page_data.get("page")
        text = page_data.get("text", "").strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Avoid cutting a word in the middle.
            if end < len(text):
                boundary = text.rfind(" ", start, end)

                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk
                })

                chunk_id += 1

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks
