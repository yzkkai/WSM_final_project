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


def create_prompt(query, context_parts, language):
    """Create an optimized prompt based on detected language."""
    context = "\n\n".join(context_parts)
    
    if language == "zh":
        prompt = f"""你是一个问答助手。请根据以下检索到的上下文来回答问题。
如果你不知道答案，请直接说"我不知道"，不要编造答案。
请保持回答简洁，最多使用三句话。

问题: {query}

上下文:
{context}

回答:"""
    else:
        prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say "I don't know" - do not make up an answer.
Keep the answer concise, using three sentences maximum.

Question: {query}

Context:
{context}

Answer:"""
    
    return prompt


def generate_answer(query, context_chunks, max_retries=3):
    """Generate answer with improved prompting, context handling, and error handling."""
    
    # Validate inputs
    if not query or not query.strip():
        return "Error: Empty query provided."
    
    if not context_chunks:
        return "I don't know. No relevant context was found to answer this question."
    
    # Detect language
    language = detect_language(query)
    
    # Optimize context
    context_parts = optimize_context(context_chunks)
    
    if not context_parts:
        return "I don't know. The retrieved context was empty or invalid."
    
    # Create prompt
    prompt = create_prompt(query, context_parts, language)
    
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
            response = client.generate(
                model=ollama_config["model"], 
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 131072,
                    "num_predict": -1
                }
            )
            
            # Validate response
            if not response or "response" not in response:
                raise ValueError("Invalid response format from Ollama")
            
            answer = response["response"].strip()
            
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
