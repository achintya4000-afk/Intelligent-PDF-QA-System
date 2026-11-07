def chunk_text(
    extracted_pages: list,
    chunk_size: int = 1000,
    overlap: int = 150
) -> list:
    """
    Divide extracted PDF pages into smaller text chunks.

    Each chunk retains its original PDF page number.
    Chunk size and overlap are measured in characters.

    The overlap is automatically reduced for small chunk sizes
    (at most one quarter of the chunk size), so every chunk size the
    frontend allows (100 to 4000) works.
    """

    if not isinstance(extracted_pages, list):
        raise ValueError("Extracted pages must be a list.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Overlap must not be negative.")

    # A fixed 150-character overlap is invalid for a 100-character chunk.
    # Limiting it also guarantees that every loop iteration moves forward.
    overlap = min(overlap, chunk_size // 4)

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
            # The boundary must be further than "overlap" characters from
            # the start, otherwise the next start position would not move
            # forward and the loop could repeat forever.
            if end < len(text):
                boundary = max(
                    text.rfind(" ", start, end),
                    text.rfind("\n", start, end)
                )

                if boundary > start + overlap:
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

            # Start the next chunk "overlap" characters back, then move
            # forward to the beginning of a word so chunks do not start
            # in the middle of a word.
            next_start = end - overlap

            while next_start < end and not text[next_start - 1].isspace():
                next_start += 1

            start = next_start

    return chunks