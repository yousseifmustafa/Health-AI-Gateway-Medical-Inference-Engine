import re
import asyncio
import logging
from typing import List, Tuple

logger = logging.getLogger("sehatech.tools.query_optimization")

_FEW_SHOT_PROMPT = """You are an expert medical query translator, rewriter, and expander. Handle colloquial language well. Your tasks for the 'Original Query' below are:
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


def _parse_generated_queries(raw: str, fallback: str) -> Tuple[str, List[str]]:
    """Shared parser for both sync and async paths."""
    if not raw:
        logger.warning("Optimization model returned None. Using original query.")
        return fallback, [fallback]

    generated_queries = []
    for line in raw.strip().split("\n"):
        cleaned = re.sub(r"^\d+[.\-\)]\s*", "", line.strip())
        cleaned = cleaned.strip('"').strip("'").strip()
        if len(cleaned) > 5:
            generated_queries.append(cleaned)

    if len(generated_queries) < 1:
        return fallback, [fallback]

    translated_query = generated_queries[0]
    expanded_queries = generated_queries[1:] or [translated_query]
    return translated_query, expanded_queries


def translate_rewrite_expand_query(optimization_model, user_query: str) -> Tuple[str, List[str]]:
    """
    Optimizes the user query by translating, rewriting, and expanding it.

    Args:
        optimization_model: The callable model function.
        user_query: The original query string (can be Arabic or English).

    Returns:
        A tuple of (translated_query, expanded_queries).
    """
    prompt = _FEW_SHOT_PROMPT.format(user_query=user_query)
    try:
        raw = optimization_model(prompt)
        return _parse_generated_queries(raw, user_query)
    except Exception as e:
        logger.error("Error during query translate/rewrite/expand: %s. Returning original.", e)
        return user_query, [user_query]


async def async_translate_rewrite_expand_query(optimization_model, user_query: str) -> Tuple[str, List[str]]:
    """Async version — offloads the sync model call to a thread pool."""
    return await asyncio.to_thread(translate_rewrite_expand_query, optimization_model, user_query)