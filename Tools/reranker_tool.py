import asyncio
from typing import List, Any
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def rerank_contexts(
    query: str,
    contexts: List[str],
    reranker_model: Any,
    top_n: int = 3
    ) -> List[str]:
    """
    Reranks a list of contexts based on their relevance to a query using a cross-encoder model.

    Args:
        query: The user's original query string.
        contexts: A list of context strings retrieved initially (e.g., top 10 from vector search).
        reranker_model: The loaded reranker model instance (e.g., FlagReranker('BAAI/bge-reranker-v2-m3')).
        top_n: The number of top contexts to return after reranking.

    Returns:
        A list containing the top_n most relevant context strings according to the reranker.
    """
    if not query or not contexts or not reranker_model:
        return contexts[:top_n] if contexts else []

    pairs = [[query, ctx] for ctx in contexts]

    try:
        scores = reranker_model.compute_score(pairs, normalize=True)

        scored_contexts = list(zip(scores, contexts))

        reranked_contexts = sorted(scored_contexts, key=lambda x: x[0], reverse=True)

        top_contexts = [ctx for score, ctx in reranked_contexts[:top_n]]

        return top_contexts

    except Exception as e:
        print(f"Error during reranking: {e}. Returning original contexts truncated.")
        return contexts[:top_n]


async def async_rerank_contexts(
    query: str,
    contexts: List[str],
    reranker_model: Any,
    top_n: int = 3
    ) -> List[str]:
    """Async version — offloads the CPU-bound reranker computation to a thread pool."""
    return await asyncio.to_thread(rerank_contexts, query, contexts, reranker_model, top_n)
