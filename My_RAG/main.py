from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker_single import chunk_document
from retriever import create_retriever
from generator import generate_answer
import argparse


def build_per_doc_retrievers(docs, language):
    per_doc = []

    for doc in docs:
        if doc["language"] != language:
            continue

        chunks = chunk_document(doc, language)
        retriever = create_retriever(chunks, language)

        per_doc.append(retriever)

    return per_doc


def build_finance_fallback_retriever(docs, language):
    finance_chunks = []
    for doc in docs:
        if doc["domain"] != "Finance" or doc["language"] != language:
            continue
        finance_chunks.extend(chunk_document(doc, language))

    return create_retriever(finance_chunks, language)


def main(query_path, docs_path, language, output_path):
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    print("Building retrievers...")
    per_doc_retrievers = build_per_doc_retrievers(docs_for_chunking, language)
    finance_retriever = build_finance_fallback_retriever(docs_for_chunking, language)

    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]

        cnt = 0
        retrieved_chunks = []
        for doc, retriever in zip(docs_for_chunking, per_doc_retrievers):
            domain = doc["domain"]
            if domain == "Finance":
                if doc["company_name"] in query_text:
                    cnt += 1
                    retrieved_chunks.append((f"Company: {doc['company_name']}", retriever.retrieve(query_text, 25)))
            elif domain == "Law":
                tmp = doc["court_name"]
                if tmp.replace(",", "") in query_text.replace(",", ""):
                    cnt += 1
                    retrieved_chunks.append((f"Court: {doc['court_name']}", retriever.retrieve(query_text, 25)))
            elif domain == "Medical":
                tmp = doc["hospital_patient_name"]
                hospital, patient = tmp.split("_", 1)
                if hospital in query_text or patient in query_text:
                    cnt += 1
                    retrieved_chunks.append((f"Hospital: {hospital}, Patient: {patient}", retriever.retrieve(query_text, 25)))
            else:
                raise ValueError("domain error")

        if cnt > 2:
            raise ValueError("?")

        if not retrieved_chunks:
            retrieved_chunks = [("Topic: Finance", finance_retriever.retrieve(query_text, 50))]

        answer = generate_answer(query_text, retrieved_chunks, language)

        query["prediction"]["content"] = answer

        for chunk in retrieved_chunks:
            for sentence in chunk[1]:
                query["prediction"]["references"].append(sentence["page_content"])

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
