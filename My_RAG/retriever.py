from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
import os
import re
import jieba
from utils import load_ollama_config


def load_stopwords(filename):
    stopwords_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    return set()

STOPWORDS_EN = load_stopwords('stopwords_en.txt')
STOPWORDS_ZH = load_stopwords('stopwords_zh.txt')

def preprocess_text(text):
    # Determine if text is likely Chinese (simple heuristic)
    is_chinese = any(u'\u4e00' <= c <= u'\u9fff' for c in text)
    
    if is_chinese:
        tokens = jieba.cut(text)
        stopwords = STOPWORDS_ZH
    else:
        tokens = re.findall(r'\w+', text.lower())
        stopwords = STOPWORDS_EN
    
    return [token for token in tokens if token not in stopwords and token.strip()]

def preprocess_text_str(text):
    return " ".join(preprocess_text(text))

def create_retriever(chunks, language):
    """Creates a LangChain Ensemble retriever with BGE re-ranker from document chunks."""
    
    # Convert chunks to LangChain Documents
    documents = []
    for chunk in chunks:
        documents.append(Document(
            page_content=chunk['page_content'],
            metadata=chunk.get('metadata', {})
        ))
    
    # 1. BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(
        documents, 
        preprocess_func=preprocess_text
    )
    bm25_retriever.k = 100
    
    # 2. Dense Retriever (FAISS + Ollama)
    embeddings = OllamaEmbeddings(model="embeddinggemma:300m", base_url=load_ollama_config()["host"])
    vectorstore = FAISS.from_documents(documents, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    
    # 3. Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    
    # Wrapper to maintain existing interface: retrieve(query, top_k)
    class RetrieverWrapper:
        def __init__(self, langchain_retriever):
            self.retriever = langchain_retriever
            
        def retrieve(self, query, top_k=1):
            docs = self.retriever.invoke(query)
            
            # Convert back to dict format expected by the app
            results = []
            for doc in docs[:top_k]:
                results.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                })
            return results

    return RetrieverWrapper(ensemble_retriever)
