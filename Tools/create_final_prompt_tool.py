from typing import Optional, List, Any

def create_final_prompt(
    user_query: str,
    support_docs: List[Any], # Changed to Any to accept Objects or Dicts
    conversation_summary: Optional[str] = None,
    max_support_docs: int = 4,
    max_total_support_chars: int = 3000
) -> list:
    """
    Builds the final prompt. Handles both Objects and Dictionaries for docs.
    Adapts instructions based on whether docs are present or not.
    """

    # 1. تجهيز الملخص
    formatted_summary = ""
    if conversation_summary:
        formatted_summary = f"## Previous Conversation Summary:\n{conversation_summary.strip()}\n"

    # 2. تجهيز المستندات (مع معالجة الأخطاء)
    formatted_support = ""
    has_docs = False # Flag to change system prompt logic

    if support_docs:
        formatted_support = "## Provided Information (Use as primary evidence *if relevant*):\n"
        total_chars_added = 0
        docs_added_count = 0 

        for i, doc in enumerate(support_docs[:max_support_docs]):
            # Handling both Object and Dictionary
            if isinstance(doc, dict):
                content = doc.get('text', '') or doc.get('page_content', '')
                source = doc.get('source', 'Unknown') or doc.get('metadata', {}).get('source', 'Unknown')
            else:
                content = getattr(doc, 'page_content', '') or getattr(doc, 'text', '')
                source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
            
            if not content: continue 
            
            cleaned_content = ' '.join(str(content).split()) 
            current_doc_string = f"{i+1}. [Source: {source}]: {cleaned_content}\n"
            
            if total_chars_added + len(current_doc_string) > max_total_support_chars:
                break
            
            formatted_support += current_doc_string
            total_chars_added += len(current_doc_string)
            docs_added_count += 1
        
        if docs_added_count > 0:
            has_docs = True
        else:
            formatted_support = "No relevant support documents found."

    else:
        formatted_support = "No relevant support documents found. Rely on your internal knowledge."

    # 3. تعديل الـ System Prompt بناءً على وجود Docs
    if has_docs:
        # RAG Mode Instructions
        evidence_instruction = """
        EVIDENCE INTEGRATION:
        You contain 'Provided Information'. Treat this as primary evidence.
        **CRITICAL:** Validate this information first. If irrelevant/contradictory, ignore it and use internal knowledge.
        Chain-of-thought: Cite the 'Provided Information' explicitly if used.
        """
    else:
        # Memory Mode Instructions (Cleaner)
        evidence_instruction = """
        KNOWLEDGE SOURCE:
        Rely SOLELY on your extensive internal medical knowledge. 
        Do not ask for external documents. Provide a comprehensive analysis based on the query.
        """

    system_prompt = f"""You are an expert-level Medical Analyst AI.
    Your purpose is to provide a detailed, analytical, and diagnostic response to medical queries.
    
    CORE KNOWLEDGE:
    Leverage your internal medical expertise to form conclusions.
    
    {evidence_instruction}

    YOUR TASK:
    1. Assess the situation.
    2. Formulate a Differential Diagnosis (if applicable).
    3. Explain your reasoning clearly.
    
    LANGUAGE:
    Respond ONLY in the same language as the User Query.
    """

    user_prompt = f"""{formatted_summary}
    ## Current User Query:
    {user_query}

    {formatted_support}

    ## Task:
    Provide an expert medical analysis following the system instructions.
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]