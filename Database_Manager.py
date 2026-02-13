"""
Database_Manager.py — PostgreSQL Connection & Thread Management for SehaTech.

Provides:
1. AsyncPostgresSaver checkpointer for LangGraph persistent chat history.
2. user_threads table linking MongoDB user IDs to LangGraph thread IDs.
3. get_or_create_thread(mongo_id) helper for thread management.

Requires:
    pip install "psycopg[binary,pool]" langgraph-checkpoint-postgres
    DATABASE_URL=postgresql://user:password@host:port/database  (in .env)
"""

import os
import uuid
import asyncio
from dotenv import load_dotenv

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

# --- SQL for the user_threads mapping table ---
CREATE_USER_THREADS_TABLE = """
CREATE TABLE IF NOT EXISTS user_threads (
    id              SERIAL PRIMARY KEY,
    mongo_user_id   VARCHAR(255) NOT NULL,
    thread_id       UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_user_threads_mongo_id
    ON user_threads (mongo_user_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_threads_active_user
    ON user_threads (mongo_user_id) WHERE is_active = TRUE;
"""

# --- Singleton Pool & Checkpointer ---
_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
_lock = asyncio.Lock()
_setup_done = False


def _get_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return url


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Returns the shared AsyncPostgresSaver instance.
    On first call: creates the connection pool, runs saver.setup()
    to create LangGraph internal tables, and creates the user_threads table.
    """
    global _pool, _checkpointer, _setup_done

    if _checkpointer is not None and _setup_done:
        return _checkpointer

    async with _lock:
        if _checkpointer is not None and _setup_done:
            return _checkpointer

        db_url = _get_db_url()

        # Create the async connection pool
        _pool = AsyncConnectionPool(
            conninfo=db_url,
            min_size=2,
            max_size=10,
            open=False,
        )
        await _pool.open()

        # Create the LangGraph checkpointer
        _checkpointer = AsyncPostgresSaver(conn=_pool)
        await _checkpointer.setup()

        # Create the user_threads mapping table
        async with _pool.connection() as conn:
            await conn.execute(CREATE_USER_THREADS_TABLE)
            await conn.commit()

        _setup_done = True
        print("INFO: PostgreSQL checkpointer & user_threads table initialized.")

    return _checkpointer


async def get_or_create_thread(mongo_id: str) -> str:
    """
    Looks up an active thread for the given MongoDB user ID.
    If one exists, returns its thread_id.
    If not, generates a new UUID, inserts a row, and returns it.

    Args:
        mongo_id: The MongoDB user ID string.

    Returns:
        thread_id as a string (UUID format).
    """
    if _pool is None:
        await get_checkpointer()  # ensure pool is ready

    async with _pool.connection() as conn:
        # Atomic upsert — prevents TOCTOU race on concurrent requests
        new_thread_id = str(uuid.uuid4())
        row = await conn.execute(
            """
            INSERT INTO user_threads (mongo_user_id, thread_id)
            VALUES (%s, %s)
            ON CONFLICT (mongo_user_id) WHERE is_active = TRUE
            DO UPDATE SET mongo_user_id = EXCLUDED.mongo_user_id
            RETURNING thread_id
            """,
            (mongo_id, new_thread_id)
        )
        result = await row.fetchone()
        await conn.commit()
        return str(result[0])


async def deactivate_thread(mongo_id: str) -> bool:
    """
    Marks the active thread for a user as inactive (soft delete).
    Useful when starting a new conversation.

    Returns:
        True if a thread was deactivated, False if none found.
    """
    if _pool is None:
        await get_checkpointer()

    async with _pool.connection() as conn:
        result = await conn.execute(
            "UPDATE user_threads SET is_active = FALSE WHERE mongo_user_id = %s AND is_active = TRUE",
            (mongo_id,)
        )
        await conn.commit()
        return result.rowcount > 0
