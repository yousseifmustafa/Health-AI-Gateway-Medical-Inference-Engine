import asyncio
import logging
from typing import List

logger = logging.getLogger("sehatech.tools.retrieval")


async def parallel_retrieval(retriever, queries: List[str]):
    """
    Runs similarity search for all expanded queries concurrently using asyncio.
    Deduplicates results by page_content to avoid redundant documents.
    """
    logger.info("Running %d retrievals concurrently using asyncio...", len(queries))

    tasks = [retriever.ainvoke(query) for query in queries]
    results_list = await asyncio.gather(*tasks)

    seen_content = set()
    unique_results = []
    for result in results_list:
        for doc in result:
            content = getattr(doc, "page_content", "")
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append(doc)

    logger.info("Retrieved %d unique documents from %d queries.", len(unique_results), len(queries))
    return unique_results