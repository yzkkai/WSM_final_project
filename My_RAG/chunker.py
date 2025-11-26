from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    # Define separators for mixed language support
    SEPARATORS_EN = ["\n\n", "\n", ".", "?", "!", " ", ""]
    SEPARATORS_ZH = ["\n\n", "\n", "。", "！", "？", " ", ""]

    # Pre-initialize splitters
    splitter_en = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS_EN,
        keep_separator="end",
        strip_whitespace=True
    )
    
    splitter_zh = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS_ZH,
        keep_separator="end",
        strip_whitespace=True
    )
    
    chunks = []
    
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            lang = doc['language']
            
            # Select appropriate splitter
            if lang == 'zh':
                splitter = splitter_zh
            else:
                splitter = splitter_en
            
            if lang == language:
                doc_chunks = splitter.split_text(text)
                for i, chunk_text in enumerate(doc_chunks):
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop('content', None)
                    chunk_metadata['chunk_index'] = i
                    chunk = {
                        'page_content': chunk_text,
                        'metadata': chunk_metadata
                    }
                    chunks.append(chunk)
                    
    return chunks
