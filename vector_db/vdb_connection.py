import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import Zilliz  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever 
from typing import Optional 

load_dotenv()

logger = logging.getLogger("sehatech.vectordb")


def create_retriever(embedding_function: HuggingFaceEmbeddings, k: int = 3) -> Optional[VectorStoreRetriever]:
    """
    Creates a retriever by "connecting" to the ZILLIZ CLOUD vector store.
    """
    try:
        ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_URI")
        ZILLIZ_CLOUD_API_KEY = os.getenv("ZILLIZ_TOKEN")
        COLLECTION_NAME = os.getenv("ZILLIZ_COLLECTION", "seha_rag_collection") 
        
        if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_API_KEY:
            logger.critical("ZILLIZ_CLOUD_URI or ZILLIZ_CLOUD_API_KEY not found in .env")
            return None
            
    except Exception as e:
        logger.error("Error reading environment variables: %s", e)
        return None

    try:
        vector_store = Zilliz(
            embedding_function=embedding_function, 
            collection_name=COLLECTION_NAME,
            connection_args={
                'uri': ZILLIZ_CLOUD_URI,
                'token': ZILLIZ_CLOUD_API_KEY,
            },
        )
        logger.info("Connected to Zilliz successfully.")
    
    except Exception as e:
        logger.critical("Error connecting to Zilliz vector store: %s", e)
        return None
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        logger.info("Retriever created successfully from Zilliz with k=%d.", k)
        return retriever
    except Exception as e:
        logger.error("Error creating retriever from Zilliz store: %s", e)
        return None