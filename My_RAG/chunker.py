import json
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(
    docs: List[Dict[str, Any]], 
    language: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 100,
    use_vllm: bool = False,
    vllm_model_name: str = "openai/gpt-oss-20b"
) -> List[Dict[str, Any]]:
    
    # Define separators for mixed language support
    separators_config = {
        "en": ["\n\n", "\n", ".", "?", "!", " ", ""],
        "zh": ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", " "]
    }

    # Pre-initialize standard splitters
    splitters = {
        "en": RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators_config["en"],
            keep_separator="end",
            strip_whitespace=True
        ),
        "zh": RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators_config["zh"],
            keep_separator="end",
            strip_whitespace=True
        )
    }

    # Container for semantic chunks if VLLM is used
    semantic_chunks_map = {}

    # 1. Batch Process with VLLM if enabled
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
            
            batch_indices = []
            prompts = []
            
            for i, doc in enumerate(docs):
                content = doc.get('content')
                # Filter for valid content and matching language
                if (content and isinstance(content, str) and 
                    doc.get('language') == language):
                    
                    batch_indices.append(i)
                    
                    system_prompt = "You are a document processing assistant."
                    user_prompt = (
                        f"Split the following text into logical semantic segments. "
                        f"Return strictly a valid JSON list of strings. "
                        f"Do not alter the text content.\n\nText:\n{content}"
                    )
                    prompts.append(f"{system_prompt}\n{user_prompt}")

            if prompts:
                # Initialize VLLM Engine
                llm = LLM(model=vllm_model_name)
                sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
                outputs = llm.generate(prompts, sampling_params)

                # Parse outputs
                for idx, output in zip(batch_indices, outputs):
                    generated_text = output.outputs[0].text.strip()
                    try:
                        clean_text = generated_text.replace("```json", "").replace("```", "").strip()
                        parsed_chunks = json.loads(clean_text)
                        
                        if isinstance(parsed_chunks, list):
                            semantic_chunks_map[idx] = parsed_chunks
                    except (json.JSONDecodeError, ValueError):
                        pass

        except ImportError:
            print("Warning: 'vllm' library not found. Falling back to standard splitters.")
        except Exception as e:
            print(f"Warning: VLLM processing failed ({e}). Falling back to standard splitters.")

    chunks = []
    
    # 2. Main Processing Loop
    for doc_index, doc in enumerate(docs):
        content = doc.get('content')
        doc_lang = doc.get('language')

        # Basic Validation
        if not (content and isinstance(content, str) and doc_lang):
            continue

        # Select Splitter
        splitter = splitters["zh"] if doc_lang == 'zh' else splitters["en"]

        if doc_lang == language:
            final_doc_chunks = []
            
            # Step A: Try VLLM Chunks
            if doc_index in semantic_chunks_map:
                candidate_chunks = semantic_chunks_map[doc_index]
                
                # VALIDATION CHECK: Ensure every chunk actually exists in the source text
                # We check if the chunk is a substring of the content.
                all_chunks_valid = True
                for chunk_str in candidate_chunks:
                    if chunk_str not in content:
                        all_chunks_valid = False
                        break
                
                if all_chunks_valid:
                    final_doc_chunks = candidate_chunks
            
            # Step B: Fallback to Standard Splitter if VLLM failed or was invalid
            if not final_doc_chunks:
                candidate_chunks = splitter.split_text(content)
                
                # VALIDATION CHECK: Strictly filter standard chunks as requested
                # (Standard splitters rarely fail this, but we enforce the rule)
                final_doc_chunks = [c for c in candidate_chunks if c in content]

            # Output Formatting
            for i, chunk_text in enumerate(final_doc_chunks):
                chunk_metadata = doc.copy()
                chunk_metadata.pop('content', None)
                chunk_metadata['chunk_index'] = i
                
                chunk = {
                    'page_content': chunk_text,
                    'metadata': chunk_metadata
                }
                chunks.append(chunk)
                    
    return chunks
