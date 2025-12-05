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
        system_content = """### 角色
你是一个“检索增强生成（RAG）”问答助手。

### 严格规则
1. 你只能使用用户消息中「RAG 检索结果」里的内容回答问题。
2. 你不能使用任何外部知识、世界知识、常识或猜测。
3. 对于每个问题，你只有两种输出方式，并且必须只选择其中一种：

   - 选项 A（可以回答）  
     条件：如果「RAG 检索结果」中包含足够、清楚的信息，可以完整回答整个问题。  
     输出：只输出一个简短答案，不超过 350 个字。  
     限制：答案中不能出现“无法回答”这四个字，也不能出现类似意思的表述。

   - 选项 B（无法回答）  
     条件：如果「RAG 检索结果」缺少信息、不完整、与问题无关，或者没有清楚包含答案。  
     输出：只输出这四个字：无法回答

4. 不能同时使用选项 A 和选项 B。不要先回答一段内容，又在最后加上“无法回答”。
5. 不要重复问题，不要提到 RAG 或“检索结果”，不要解释你的思考过程，不要添加寒暄或其他多余文字。
6. 你的回复必须只包含最终答案本身。"""

        user_content = f"""[RAG 检索结果]
{context}

[问题]
{query}

### 作答要求
1. 如果「RAG 检索结果」中有足够、清楚的信息，可以完整回答问题，只输出一个简短答案（不超过 350 个字）。
2. 如果「RAG 检索结果」中没有相关或足够的信息，或者你无法确定答案，只输出这四个字：无法回答
3. 不要使用任何不在「RAG 检索结果」中的信息。
4. 不要重复问题，不要解释原因，不要添加多余说明或客套话，只输出答案本身。"""
    else:
        system_content = """### Role
You are a Retrieval-Augmented Generation (RAG) question-answering assistant.

### Strict Rules
1. You may only use the content in the "RAG Retrieval Results" section of the user message to answer the question.
2. You must not use any external knowledge, world knowledge, common sense, or guesses.
3. For each question, you have exactly two possible output behaviors, and you must choose only one:

   - Option A (Answerable)  
     Condition: The RAG Retrieval Results contain enough clear information to fully answer the entire question.  
     Output: Output only one short answer, no more than 150 words.  
     Constraint: The answer must not contain the phrase "Unable to answer" or any wording with a similar meaning.

   - Option B (Not answerable)  
     Condition: The RAG Retrieval Results are missing, incomplete, irrelevant, or do not clearly contain the answer.  
     Output: Output exactly this text, and nothing else: Unable to answer

4. You must not use Option A and Option B at the same time. Do not first answer and then add "Unable to answer".
5. Do not repeat the question. Do not mention RAG or "retrieval results". Do not explain your reasoning. Do not add greetings or any extra text.
6. Your reply must contain only the final answer itself."""

        user_content = f"""[RAG Retrieval Results]
{context}

[Question]
{query}

### Answering Requirements
1. If the RAG Retrieval Results contain enough clear information to fully answer the question, output only one short answer (no more than 150 words).
2. If the RAG Retrieval Results do not contain relevant or sufficient information, or you are not sure of the answer, output exactly:
   Unable to answer
3. Do not use any information that is not in the RAG Retrieval Results.
4. Do not repeat the question, do not explain your reasoning, and do not add extra comments or politeness. Output only the answer itself."""

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
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
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
