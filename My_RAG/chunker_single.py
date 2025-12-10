import spacy

_NLP_CACHE = {}

def get_spacy_pipeline(lang):
    if lang not in _NLP_CACHE:
        if lang == "en":
            model_name = "en_core_web_trf"
        elif lang == "zh":
            model_name = "zh_core_web_trf"

        _NLP_CACHE[lang] = spacy.load(model_name)

    return _NLP_CACHE[lang]


def chunk_document(doc, language):
    text = doc["content"]

    nlp = get_spacy_pipeline(language)

    spacy_doc = nlp(text)

    chunks = []
    for idx, sent in enumerate(spacy_doc.sents):
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        chunk = {
            "page_content": sent_text,
            "metadata": {
                "index": idx,
                "start_char": sent.start_char,
                "end_char": sent.end_char,
            },
        }
        chunks.append(chunk)

    return chunks
