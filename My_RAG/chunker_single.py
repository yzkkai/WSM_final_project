def split(text, sep):
    marker = "<<<SPLIT_MARKER>>>"

    tmp = text
    for p in sep:
        tmp = tmp.replace(p, p + marker)

    parts = [s.strip() for s in tmp.split(marker) if s.strip()]
    return parts


def chunk_document(doc, language):
    SEPARATORS_EN = ["\n", ".", "?", "!"]
    SEPARATORS_ZH = ["\n", "。", "！", "？"]


    if doc['language'] != language:
        raise ValueError("language error")

    if language == 'en':
        sep = SEPARATORS_EN
    else:
        sep = SEPARATORS_ZH

    chunks = []
    if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
        text = doc['content']
    
        doc_chunks = split(text, sep)
        for idx, chunk_text in enumerate(doc_chunks):
            chunk = {
                'page_content': chunk_text,
                'metadata': {"index": idx}
            }
            chunks.append(chunk)
                    
    return chunks
