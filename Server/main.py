from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from Server.config import logger, limiter
from Server.routers.chat import router as chat_router
from Server.routers.analyzers import router as analyzers_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the LangGraph supervisor once at startup, pre-warm models, and start background tasks."""
    
    try:
        from Langgraphs.supervisor_graph import make_graph
        app_graph = await make_graph()
        app.state.app_graph = app_graph
        logger.info("Supervisor graph compiled and cached at startup.")
    except Exception as e:
        logger.error("Failed to compile supervisor graph at startup: %s", e)
        app.state.app_graph = None

    # Pre-warm heavy models so first request doesn't pay cold-start penalty
    try:
        from Models.Model_Manager import get_model_manager
        mm = get_model_manager()
        _ = mm.embedding_model  
        logger.info("Models pre-warmed: embedding model loaded.")
    except Exception as e:
        logger.warning("Model pre-warming failed (non-fatal): %s", e)

    # Start background WAL drain loop
    try:
        from Database_Manager import start_wal_drain_loop
        start_wal_drain_loop()
    except Exception as e:
        logger.warning("WAL drain loop start failed (non-fatal): %s", e)

    yield #server is running 

    # shutdown
    logger.info("Server shutting down — cleaning up resources...")
    try:
        from Database_Manager import shutdown as db_shutdown
        await db_shutdown()
    except Exception as e:
        logger.warning("Shutdown cleanup error: %s", e)
    logger.info("Server shutdown complete.")


app = FastAPI(lifespan=lifespan, title="SehaTech AI Supervisor")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429, content={"detail": "Rate limit exceeded. Try again in a minute."}
))

@app.get("/health")
async def health_check():
    """Health probe for load balancers."""
    return {"status": "ok", "graph_loaded": getattr(app.state, "app_graph", None) is not None}

app.include_router(chat_router)
app.include_router(analyzers_router)
