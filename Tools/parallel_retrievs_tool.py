import asyncio
from typing import List

async def parallel_retrieval(retriever, queries: List[str]):
    """
    Runs similarity search for all expanded queries concurrently using asyncio.
    """
    print(f"INFO: Running {len(queries)} retrievals concurrently using asyncio...")
   
    tasks = [retriever.ainvoke(query) for query in queries]
    results_list = await asyncio.gather(*tasks)
    
    all_results = []
    for result in results_list:
        all_results.extend(result)

    return all_results