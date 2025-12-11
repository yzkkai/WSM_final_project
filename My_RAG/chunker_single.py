import spacy
import re

_NLP_CACHE = {}


def get_spacy_pipeline(lang):
    if lang not in _NLP_CACHE:
        if lang == "en":
            model_name = "en_core_web_trf"
        elif lang == "zh":
            model_name = "zh_core_web_trf"

        _NLP_CACHE[lang] = spacy.load(model_name)

    return _NLP_CACHE[lang]


def is_low_info_sentence(language, text, nlp, min_chars=6, min_content_tokens=4):
    if not text:
        return True

    sent = text.strip()
    if not sent:
        return True

    if len(sent) < min_chars:
        return True

    doc = nlp(sent)

    total_tokens = 0
    content_tokens = 0

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue

        total_tokens += 1

        if tok.is_stop:
            continue

        content_tokens += 1

    if total_tokens == 0:
        return True

    if content_tokens < min_content_tokens:
        return True

    return False


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
