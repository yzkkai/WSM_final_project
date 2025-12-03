from ollama import Client
import time
from utils import load_ollama_config

def detect_language(text):
    """Detect if text is primarily Chinese or English."""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len([c for c in text if c.strip()])
    return "zh" if chinese_chars / max(total_chars, 1) > 0.3 else "en"


def deduplicate_context(chunks):
    """Remove duplicate chunks based on content."""
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        content = chunk['page_content'].strip()
        if content and content not in seen:
            seen.add(content)
            unique_chunks.append(chunk)
    return unique_chunks


def optimize_context(chunks, max_length=None):
    """Optimize context by deduplication and length limiting."""
    # Deduplicate
    chunks = deduplicate_context(chunks)
    
    # Limit total length
    context_parts = []
    total_length = 0
    for chunk in chunks:
        content = chunk['page_content']
        if max_length is not None and total_length + len(content) > max_length:
            # Truncate last chunk if needed
            remaining = max_length - total_length
            if remaining > 100:  # Only add if meaningful
                context_parts.append(content[:remaining] + "...")
            break
        context_parts.append(content)
        total_length += len(content)
    
    return context_parts


def create_messages(query, context_parts, language):
    context = "\n\n".join(context_parts)

    if language == "zh":
        system_content = (
            "你是一个问答助手，只能根据提供的 RAG 检索结果回答问题。"
        )

        user_content = f"""【问题】
{query}

【RAG 检索结果】
{context}

【回答规则】
1. 只能使用“RAG 检索结果”中的信息，不得加入外部知识。
2. 如果检索结果中没有答案，请回答：“无法回答”。
3. 回答要简洁，不超过 350 个字。
4. 不要复述上下文中无关的内容，只回答问题本身。

【回答】请在此开始回答："""
    else:
        system_content = (
            "You are a question-answering assistant. You must answer strictly based on the provided RAG retrieval results."
        )

        user_content = f"""[Question]
{query}

[RAG Retrieval Results]
{context}

[Answering Rules]
1. You may only use information found in the "RAG Retrieval Results". Do not use any external knowledge.
2. If the retrieval results do not contain the answer, respond with: "Unable to answer".
3. Keep the answer concise, no more than 150 words.
4. Do not repeat irrelevant parts of the retrieved text. Only answer the question directly.

[Answer] Please begin your answer here:"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_answer(query, context_chunks, max_retries=3):
    """Generate answer with improved prompting, context handling, and error handling."""
    
    # Validate inputs
    if not query or not query.strip():
        return "Error: Empty query provided."
    
    if not context_chunks:
        return "Unable to answer."
    
    # Detect language
    language = detect_language(query)
    
    # Optimize context
    context_parts = optimize_context(context_chunks)
    
    if not context_parts:
        return "Unable to answer."
    
    # Create prompt
    messages = create_messages(query, context_parts, language)
    
    # Load config
    try:
        ollama_config = load_ollama_config()
    except Exception as e:
        return f"Error: Failed to load configuration - {str(e)}"
    
    # Generate with retry logic
    client = Client(host=ollama_config["host"])
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = client.chat(
                model=ollama_config["model"],
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 131072,
                    "num_predict": -1,
                },
            )
            
            answer = response["message"]["content"].strip()
            
            # Post-process answer
            if not answer:
                raise ValueError("Empty response from Ollama")
            
            return answer
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
    
    # All retries failed
    return f"Error: Failed to generate answer after {max_retries} attempts - {str(last_error)}"


if __name__ == "__main__":
    # Test the function
    print("=== Test 1: English Query ===")
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Query:", query)
    print("Answer:", answer)
    
    print("\n=== Test 2: Chinese Query ===")
    query_zh = "法国的首都是什么？"
    answer_zh = generate_answer(query_zh, context_chunks)
    print("Query:", query_zh)
    print("Answer:", answer_zh)
    
    print("\n=== Test 3: No Context ===")
    answer_no_ctx = generate_answer("What is quantum physics?", [])
    print("Query: What is quantum physics?")
    print("Answer:", answer_no_ctx)
