import re
import asyncio
from typing import List, Tuple


def translate_rewrite_expand_query(Optimization_Model, user_query: str) -> Tuple[str, List[str]]:
    """
    Optimizes the user query by:
    1. Providing a literal English translation.
    2. Rewriting it into a primary English medical question.
    3. Generating additional related query variations.
    Uses a few-shot prompt for the optimizer LLM (via API).

    Args:
        Optimization_Model: The callable model function.
        user_query: The original query string (can be Arabic or English).

    Returns:
        A tuple containing:
        - translated_query (str): The literal translation (or original if failed).
        - expanded_queries (List[str]): A list of expanded/rewritten queries.
    """

    prompt = f"""You are an expert medical query translator, rewriter, and expander. Handle colloquial language well. Your tasks for the 'Original Query' below are:
1.  **Translate:** Provide ONE literal, accurate translation of the Original Query into standard English. Preserve ALL details and symptoms. **Do NOT rewrite or interpret yet.**
2.  **Rewrite Primary:** THEN, formulate ONE primary, clear, specific medical question in ENGLISH based on ALL details from the Original Query. Suitable for a medical database.
3.  **Expand Variations:** Generate 2 additional distinct query variations in ENGLISH exploring different facets based on ALL details.
**Output Instruction:** Return ONLY these 4 query strings. Format them EXACTLY as a numbered list (4 items total, each on a new line: "1. Literal Translation\\n2. Primary Rewritten\\n3. Expansion 1\\n4. Expansion 2"). Do absolutely NOT include any explanations, reasoning, tags, titles, or any other text.

--- Examples ---

Original Query (Arabic): "طفل عنده سخونية وكحة بقاله يومين"
1. Child has fever and cough for two days.
2. What are the differential diagnoses and treatment for a child presenting with fever and cough for two days?
3. Pediatric fever cough differential diagnosis
4. Common causes of acute cough and fever in children

Original Query (English): "knee pain when walking"
1. knee pain when walking
2. What are the common causes and diagnostic approaches for knee pain that occurs during ambulation?
3. Differential diagnosis for exertional knee pain
4. Knee pain aggravated by walking causes

Original Query (Arabic): "عندي صداع نصفي ومش بيروح بالمسكنات العادية"
1. I have a migraine headache that does not respond to regular painkillers.
2. What are the management options for migraine headaches refractory to standard analgesics?
3. Treatment-resistant migraine alternative therapies
4. Chronic migraine management strategies

--- Task ---

Original Query: "{user_query}"
"""

    try:
        raw_response_content = Optimization_Model(prompt)

        # حالة 1: الموديل مرجعش حاجة
        if not raw_response_content:
            print("Warning: Optimization_Model returned None. Using original query.")
            return user_query, [user_query]

        generated_queries = []
        lines = raw_response_content.strip().split('\n')

        for line in lines:
            cleaned_line = line.strip()
            # تنظيف الترقيم (1. أو 1- أو 1))
            cleaned_line = re.sub(r"^\d+[\.\-\)]\s*", "", cleaned_line)
            cleaned_line = cleaned_line.strip('"').strip("'").strip()

            if len(cleaned_line) > 5:
                generated_queries.append(cleaned_line)

        # حالة 2: الموديل رجع ليست فاضية
        if len(generated_queries) < 1:
            return user_query, [user_query]

        # حالة 3: الموديل رجع إجابات (بنقسمهم لـ Translated والباقي Expansions)
        else:
            # أول واحدة هي الترجمة الحرفية
            translated_query = generated_queries[0]

            # لو مفيش غير واحدة، الليستة هتبقى فاضية وده أمان
            expanded_queries = generated_queries[1:]

            if not expanded_queries:
                expanded_queries = [translated_query]

            return translated_query, expanded_queries

    except Exception as e:
        print(f"Error during query translate/rewrite/expand process: {e}. Returning original query.")
        return user_query, [user_query]


async def async_translate_rewrite_expand_query(Optimization_Model, user_query: str) -> Tuple[str, List[str]]:
    """
    Async version — offloads the sync model call to a thread pool.
    The prompt building and parsing are lightweight; only the model call blocks.
    """

    prompt = f"""You are an expert medical query translator, rewriter, and expander. Handle colloquial language well. Your tasks for the 'Original Query' below are:
1.  **Translate:** Provide ONE literal, accurate translation of the Original Query into standard English. Preserve ALL details and symptoms. **Do NOT rewrite or interpret yet.**
2.  **Rewrite Primary:** THEN, formulate ONE primary, clear, specific medical question in ENGLISH based on ALL details from the Original Query. Suitable for a medical database.
3.  **Expand Variations:** Generate 2 additional distinct query variations in ENGLISH exploring different facets based on ALL details.
**Output Instruction:** Return ONLY these 4 query strings. Format them EXACTLY as a numbered list (4 items total, each on a new line: "1. Literal Translation\\n2. Primary Rewritten\\n3. Expansion 1\\n4. Expansion 2"). Do absolutely NOT include any explanations, reasoning, tags, titles, or any other text.

--- Examples ---

Original Query (Arabic): "طفل عنده سخونية وكحة بقاله يومين"
1. Child has fever and cough for two days.
2. What are the differential diagnoses and treatment for a child presenting with fever and cough for two days?
3. Pediatric fever cough differential diagnosis
4. Common causes of acute cough and fever in children

Original Query (English): "knee pain when walking"
1. knee pain when walking
2. What are the common causes and diagnostic approaches for knee pain that occurs during ambulation?
3. Differential diagnosis for exertional knee pain
4. Knee pain aggravated by walking causes

Original Query (Arabic): "عندي صداع نصفي ومش بيروح بالمسكنات العادية"
1. I have a migraine headache that does not respond to regular painkillers.
2. What are the management options for migraine headaches refractory to standard analgesics?
3. Treatment-resistant migraine alternative therapies
4. Chronic migraine management strategies

--- Task ---

Original Query: "{user_query}"
"""

    try:
        raw_response_content = await asyncio.to_thread(Optimization_Model, prompt)

        if not raw_response_content:
            print("Warning: Optimization_Model returned None. Using original query.")
            return user_query, [user_query]

        generated_queries = []
        lines = raw_response_content.strip().split('\n')

        for line in lines:
            cleaned_line = line.strip()
            cleaned_line = re.sub(r"^\d+[\.\-\)]\s*", "", cleaned_line)
            cleaned_line = cleaned_line.strip('"').strip("'").strip()

            if len(cleaned_line) > 5:
                generated_queries.append(cleaned_line)

        if len(generated_queries) < 1:
            return user_query, [user_query]
        else:
            translated_query = generated_queries[0]
            expanded_queries = generated_queries[1:]
            if not expanded_queries:
                expanded_queries = [translated_query]
            return translated_query, expanded_queries

    except Exception as e:
        print(f"Error during async query translate/rewrite/expand process: {e}. Returning original query.")
        return user_query, [user_query]