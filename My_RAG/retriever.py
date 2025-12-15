from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

import os
import re
import spacy

from utils import load_ollama_config

_NLP_CACHE = {}


def get_spacy_pipeline(lang):
    if lang not in _NLP_CACHE:
        if lang == "en":
            model_name = "en_core_web_trf"
        elif lang == "zh":
            model_name = "zh_core_web_trf"
        _NLP_CACHE[lang] = spacy.load(model_name)
    return _NLP_CACHE[lang]


def make_spacy_preprocess_func(lang):
    def preprocess_text(text):
        nlp = get_spacy_pipeline(lang)

        doc = nlp(text)

        if lang == "en":
            tokens = []
            for tok in doc:
                if tok.is_space or tok.is_punct or tok.is_stop:
                    continue

                lemma = tok.lemma_.lower().strip()
                if not lemma:
                    continue

                tokens.append(lemma)

            return tokens

        else:
            tokens = []
            for tok in doc:
                txt = tok.text.strip()
                if not txt:
                    continue
                if tok.is_space or tok.is_punct or tok.is_stop:
                    continue
                tokens.append(txt)

            return tokens

    return preprocess_text


def create_retriever(chunks, language):
    # Convert chunks to LangChain Documents
    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"],
            )
        )

    # 1. BM25 Retriever
    bm25_preprocess_func = make_spacy_preprocess_func(language)
    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=bm25_preprocess_func,
    )
    bm25_retriever.k = 100

    # 2. Dense Retriever (FAISS + Ollama)
    ollama_conf = load_ollama_config()
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url=ollama_conf["host"],
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 100})

    # 3. Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7],
    )

    # Wrapper to maintain existing interface: retrieve(query, top_k)
    class RetrieverWrapper:
        def __init__(self, langchain_retriever):
            self.retriever = langchain_retriever

        def retrieve(self, query, top_k=5):
            docs = self.retriever.invoke(query)

            # Convert back to dict format expected by the app
            results = []
            for doc in docs[:top_k]:
                results.append(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )
            return results

    return RetrieverWrapper(ensemble_retriever)
