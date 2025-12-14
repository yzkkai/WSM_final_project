from ollama import Client
import time
from utils import load_ollama_config


def build_context(context_chunks, language):
    blocks = []

    for idx, (topic, sentences) in enumerate(context_chunks, start=1):
        if isinstance(sentences, list):
            content = "\n".join([f"{sentence['page_content']}" for sentence in sentences])
        else:
            content = sentences

        if language == "zh":
            block = (
                f"【检索结果 {idx}】\n"
                f"{topic}\n"
                f"{content}"
            )
        else:
            block = (
                f"[Result {idx}]\n"
                f"{topic}\n"
                f"{content}"
            )

        blocks.append(block)

    return "\n\n".join(blocks)


def create_messages(query, context, language):
    if language == "zh":
        system_content = "你是一个问答助手，只能根据提供的 RAG 检索结果回答问题。"

        user_content = f"""【RAG 检索结果】
{context}

【问题】
{query}

【回答规则】
1. 只能使用“RAG 检索结果”中的信息，不得加入外部知识。
2. 如果检索结果中没有答案，请回答：“无法回答”。
3. 回答要简洁，不超过 350 个字。
4. 不要复述上下文中无关的内容，只回答问题本身。

【回答】请在此开始回答："""
    else:
        system_content = "You are a question-answering assistant. You must answer strictly based on the provided RAG retrieval results."

        user_content = f"""[RAG Retrieval Results]
{context}

[Question]
{query}

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


def generate_answer(query, context_chunks, language, max_retries=3):
    """Generate answer with improved prompting, context handling, and error handling."""
    
    if not context_chunks:
        return "Unable to answer."
    
    # Optimize context
    context = build_context(context_chunks, language)
    
    # Create prompt
    messages = create_messages(query, context, language)
    print(messages)

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
