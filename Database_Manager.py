"""
Database_Manager.py — PostgreSQL Connection, Thread Management & Backend API Sync for SehaTech.

Provides:
1. AsyncPostgresSaver checkpointer for LangGraph persistent chat history (Ephemeral Layer).
2. user_threads table linking MongoDB user IDs to LangGraph thread IDs.
3. get_or_create_thread(mongo_id) helper for thread management.
4. fetch_user_permanent_records(user_id) — Retrieves life-long patient data from Backend API (Permanent Layer).
5. modify_user_permanent_records(user_id, action, fact) — Syncs patient data changes back to Backend API.

Requires:
    pip install "psycopg[binary,pool]" langgraph-checkpoint-postgres httpx
    DATABASE_URL=postgresql://user:password@host:port/database  (in .env)
    SEHATECH_API_BASE=https://api.sehatech.com/v1  (in .env, optional — has default)
"""

import os
import uuid
import asyncio
import logging
from dotenv import load_dotenv

import httpx
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

logger = logging.getLogger("sehatech.database")

# --- Backend API Configuration ---
SEHATECH_API_BASE = os.getenv("SEHATECH_API_BASE", "https://api.sehatech.com/v1")

# --- Mock Mode: set MOCK_BACKEND_API=true when the real backend is not ready ---
MOCK_BACKEND_API = os.getenv("MOCK_BACKEND_API", "true").lower() in ("true", "1", "yes")

_mock_records_store: dict[str, dict] = {}


def _get_mock_records(user_id: str) -> dict:
    """Returns (or initializes) mock patient records for a given user_id."""
    if user_id not in _mock_records_store:
        _mock_records_store[user_id] = {
            "user_id": user_id,
            "allergies": ["Penicillin"],
            "chronic_diseases": ["Type 2 Diabetes"],
            "current_medications": ["Metformin 500mg"],
            "surgical_history": [],
            "notes": "Mock data — real backend API not yet integrated.",
        }
    return _mock_records_store[user_id]

# --- SQL for the user_threads mapping table ---
CREATE_USER_THREADS_TABLE = """
CREATE TABLE IF NOT EXISTS user_threads (
    id              SERIAL PRIMARY KEY,
    mongo_user_id   VARCHAR(255) NOT NULL,
    thread_id       TEXT NOT NULL UNIQUE,  -- Changed from UUID to TEXT for dynamic IDs
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_user_threads_mongo_id
    ON user_threads (mongo_user_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_threads_active_user
    ON user_threads (mongo_user_id) WHERE is_active = TRUE;
"""

# --- SQL for the Write-Ahead Log (WAL) table ---
CREATE_WAL_TABLE = """
CREATE TABLE IF NOT EXISTS pending_record_modifications (
    id          SERIAL PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    action      VARCHAR(10) NOT NULL,
    fact        TEXT NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    retry_count INTEGER DEFAULT 0,
    status      VARCHAR(20) DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_wal_pending
    ON pending_record_modifications (status) WHERE status = 'pending';
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

        # Create the async connection pool with health checks
        _pool = AsyncConnectionPool(
            conninfo=db_url,
            min_size=2,
            max_size=10,
            open=False,
            reconnect_timeout=300,
        )
        await _pool.open()

        # Create the LangGraph checkpointer tables (requires autocommit for CREATE INDEX CONCURRENTLY)
        async with _pool.connection() as setup_conn:
            try:
                await setup_conn.set_autocommit(True)
            except AttributeError:
                setup_conn.autocommit = True
                
            temp_saver = AsyncPostgresSaver(conn=setup_conn)
            await temp_saver.setup()

        # Initialize the persistent checkpointer with the pool
        _checkpointer = AsyncPostgresSaver(conn=_pool)
        
        # Create the user_threads mapping table + WAL table
        async with _pool.connection() as conn:
            # Migration: Ensure thread_id is TEXT (if it was UUID from old schema)
            try:
                await conn.execute("ALTER TABLE user_threads ALTER COLUMN thread_id TYPE TEXT USING thread_id::text")
            except Exception:
                pass  # Ignore if already TEXT or table doesn't exist

            await conn.execute(CREATE_USER_THREADS_TABLE)
            await conn.execute(CREATE_WAL_TABLE)
            await conn.commit()

        _setup_done = True
        logger.info("PostgreSQL checkpointer, user_threads, and WAL table initialized.")

        # Drain any pending WAL entries from previous runs
        await retry_pending_modifications()

    return _checkpointer


async def get_or_create_thread(user_uid: str, force_new_session: bool = False) -> str:
    """
    Implements Dynamic Session-Based Threading.
    Format: "{user_uid}_{sequence_number}" (e.g., "USER123_1", "USER123_2")

    Args:
        user_uid: The user's unique identifier.
        force_new_session: If True, deactivates current thread and creates a new one.

    Returns:
        The active thread_id string.
    """
    if _pool is None:
        await get_checkpointer()

    async with _pool.connection() as conn:
        # 1. Count total threads for this user to determine sequence number
        # We count ALL threads (active + inactive) to ensure strictly increasing sequence
        row = await conn.execute(
            "SELECT COUNT(*) FROM user_threads WHERE mongo_user_id = %s",
            (user_uid,)
        )
        total_count = (await row.fetchone())[0]

        # 2. Check for an existing ACTIVE thread
        if not force_new_session:
            row = await conn.execute(
                "SELECT thread_id FROM user_threads WHERE mongo_user_id = %s AND is_active = TRUE ORDER BY created_at DESC LIMIT 1",
                (user_uid,)
            )
            existing = await row.fetchone()
            if existing:
                return existing[0]

        # 3. Create NEW thread (Sequence = Total + 1)
        new_sequence = total_count + 1
        new_thread_id = f"{user_uid}_{new_sequence}"

        # Deactivate any old active threads first (to be safe)
        if force_new_session:
            await conn.execute(
                "UPDATE user_threads SET is_active = FALSE WHERE mongo_user_id = %s",
                (user_uid,)
            )

        # Insert the new thread
        try:
            await conn.execute(
                """
                INSERT INTO user_threads (mongo_user_id, thread_id, is_active)
                VALUES (%s, %s, TRUE)
                """,
                (user_uid, new_thread_id)
            )
            await conn.commit()
            logger.info("Session initialized: %s", new_thread_id)
            return new_thread_id
        except Exception as e:
            # Handle race condition or duplicate key error gracefully
            logger.warning("Race condition creating thread %s: %s. Retrying fetch...", new_thread_id, e)
            return await get_or_create_thread(user_uid, force_new_session=False)


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


# ============================================================================
# PERMANENT MEMORY LAYER — Backend API Sync (Patient Longitudinal Safety)
# ============================================================================

# --- Persistent HTTP Client (reuses TCP connections across calls) ---
_http_client: httpx.AsyncClient | None = None
_http_lock = asyncio.Lock()

_RETRY_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 2.0


async def _get_http_client() -> httpx.AsyncClient:
    """Returns a shared, persistent httpx.AsyncClient (lazy-initialized)."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        return _http_client
    async with _http_lock:
        if _http_client is not None and not _http_client.is_closed:
            return _http_client
        _http_client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )
        logger.info("Persistent httpx.AsyncClient initialized.")
    return _http_client


async def fetch_user_permanent_records(user_id: str) -> dict:
    """
    Retrieves a patient's permanent medical records from the SehaTech Backend API.

    This function is the READ side of the Permanent Memory Layer. It performs an
    async GET request to the centralized patient record service to fetch life-long
    medical data that persists across all sessions and devices.

    **Patient Longitudinal Safety:**
    By proactively loading a patient's permanent profile (Allergies, Chronic
    Diseases, Current Medications, Surgical History) at the START of every triage
    session, the Supervisor Agent can:
      - Avoid prescribing contraindicated medications (e.g., NSAIDs for a patient
        with Chronic Kidney Disease).
      - Ground its questions in known context instead of re-asking what it should
        already know ("I see you have Type 2 Diabetes — let's check your glucose first").
      - Reduce Medical Hallucination by cross-referencing new symptoms against
        established conditions.

    Args:
        user_id: The unique patient identifier (MongoDB ObjectId string).

    Returns:
        A dictionary containing the patient's permanent records with keys such as
        'allergies', 'chronic_diseases', 'current_medications', 'surgical_history'.
        Returns a neutral empty-records dict on any failure (API down, timeout,
        user not found) so the system can still operate without permanent data.
    """
    _empty_fallback = {
        "user_id": user_id,
        "allergies": [],
        "chronic_diseases": [],
        "current_medications": [],
        "surgical_history": [],
        "notes": "No permanent records available — proceeding with session data only.",
    }

    # --- Mock mode: return instant demo data ---
    if MOCK_BACKEND_API:
        records = _get_mock_records(user_id)
        logger.info("[MOCK] Permanent records returned for user %s.", user_id)
        return records

    client = await _get_http_client()

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            response = await client.get(f"{SEHATECH_API_BASE}/get-records/{user_id}")

            if response.status_code == 200:
                records = response.json()
                logger.info("Permanent records fetched for user %s.", user_id)
                return records
            elif response.status_code == 404:
                logger.info("No permanent records found for user %s. New patient profile.", user_id)
                return _empty_fallback
            else:
                logger.warning(
                    "Backend API returned status %d for user %s (attempt %d/%d).",
                    response.status_code, user_id, attempt, _RETRY_ATTEMPTS,
                )

        except httpx.TimeoutException:
            logger.warning(
                "Backend API timed out fetching records for user %s (attempt %d/%d).",
                user_id, attempt, _RETRY_ATTEMPTS,
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch permanent records for user %s (attempt %d/%d): %s",
                user_id, attempt, _RETRY_ATTEMPTS, e,
            )

        # Backoff before retry (skip on last attempt)
        if attempt < _RETRY_ATTEMPTS:
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

    return _empty_fallback


async def modify_user_permanent_records(user_id: str, action: str, medical_fact: str) -> str:
    """
    Syncs a change to a patient's permanent medical record back to the SehaTech Backend API.

    This function is the WRITE side of the Permanent Memory Layer. It sends an async
    POST request to update the centralized patient record service, ensuring that the
    backend stays fresh and accurate based on real-time patient interactions.

    **Patient Longitudinal Safety:**
    Medical records are living documents. This tool ensures that:
      - **ADD**: New diagnoses, allergies, or medications discovered during triage are
        immediately persisted (e.g., a newly identified Penicillin allergy).
      - **REMOVE**: Conditions that a patient has recovered from or treatments that
        have concluded are cleared (e.g., "I finished my antibiotics course last week").
      - **UPDATE**: Existing facts are corrected or refined (e.g., updating dosage
        of a chronic medication).

    The 'source' field in the payload is always set to 'AI_Triage_Supervisor' to
    maintain a clear audit trail of which system made the modification.

    Args:
        user_id: The unique patient identifier (MongoDB ObjectId string).
        action:  One of 'ADD', 'REMOVE', or 'UPDATE' — the type of modification.
        medical_fact: The specific medical fact being modified (e.g.,
                      "Penicillin allergy", "Type 2 Diabetes", "Metformin 500mg").

    Returns:
        A success or failure message string describing the outcome.
    """
    valid_actions = {"ADD", "REMOVE", "UPDATE"}
    action_upper = action.strip().upper()

    if action_upper not in valid_actions:
        return f"Error: Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}."

    # --- Mock mode: apply change to in-memory store ---
    if MOCK_BACKEND_API:
        records = _get_mock_records(user_id)
        fact = medical_fact.strip()
        # Determine which list to modify based on simple keyword heuristics
        target_key = None
        fact_lower = fact.lower()
        if any(kw in fact_lower for kw in ["allergy", "allergic", "penicillin", "sulfa"]):
            target_key = "allergies"
        elif any(kw in fact_lower for kw in ["diabetes", "hypertension", "asthma", "chronic", "disease"]):
            target_key = "chronic_diseases"
        elif any(kw in fact_lower for kw in ["surgery", "surgical", "appendectomy", "operation"]):
            target_key = "surgical_history"
        else:
            target_key = "current_medications"

        if action_upper == "ADD":
            if fact not in records[target_key]:
                records[target_key].append(fact)
        elif action_upper == "REMOVE":
            records[target_key] = [item for item in records[target_key] if item.lower() != fact_lower]
        elif action_upper == "UPDATE":
            # For update, just add it (the old value should be removed separately)
            if fact not in records[target_key]:
                records[target_key].append(fact)

        logger.info("[MOCK] Record updated — %s '%s' in %s for user %s.", action_upper, fact, target_key, user_id)
        return f"Success: Patient record updated — {action_upper} '{fact}'."

    payload = {
        "userId": user_id,
        "action": action_upper,
        "fact": medical_fact.strip(),
        "source": "AI_Triage_Supervisor",
    }

    client = await _get_http_client()

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            response = await client.post(
                f"{SEHATECH_API_BASE}/update-records",
                json=payload,
            )

            if response.status_code in (200, 201):
                logger.info("Permanent record updated — %s '%s' for user %s.", action_upper, medical_fact, user_id)
                return f"Success: Patient record updated — {action_upper} '{medical_fact}'."
            else:
                logger.warning(
                    "Backend API returned %d on record update for %s (attempt %d/%d).",
                    response.status_code, user_id, attempt, _RETRY_ATTEMPTS,
                )

        except httpx.TimeoutException:
            logger.warning(
                "Backend API timed out updating records for user %s (attempt %d/%d).",
                user_id, attempt, _RETRY_ATTEMPTS,
            )
        except Exception as e:
            logger.warning(
                "Failed to modify permanent records for user %s (attempt %d/%d): %s",
                user_id, attempt, _RETRY_ATTEMPTS, e,
            )

        if attempt < _RETRY_ATTEMPTS:
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS)

    # --- WAL Fallback: queue the failed modification for later retry ---
    await _wal_enqueue(user_id, action_upper, medical_fact.strip())

    return "Warning: Backend API unreachable after retries. The change was not saved — please retry."


# ============================================================================
# WRITE-AHEAD LOG (WAL) — Guarantees no patient data is silently lost
# ============================================================================

_WAL_MAX_RETRIES = 5


async def _wal_enqueue(user_id: str, action: str, fact: str) -> None:
    """Insert a failed modification into the WAL for later retry."""
    if _pool is None:
        logger.error("WAL enqueue failed — DB pool not initialized.")
        return
    try:
        async with _pool.connection() as conn:
            await conn.execute(
                "INSERT INTO pending_record_modifications (user_id, action, fact) VALUES (%s, %s, %s)",
                (user_id, action, fact),
            )
            await conn.commit()
        logger.info("WAL: Queued failed modification for user %s — %s '%s'.", user_id, action, fact)
    except Exception as e:
        logger.error("WAL enqueue error: %s", e)


async def retry_pending_modifications() -> None:
    """
    Drains the WAL table by retrying all 'pending' modifications.
    Called at startup and periodically via the background drain loop.
    """
    if _pool is None:
        return

    # --- Step 1: Mark entries that have exhausted retries as 'failed' ---
    try:
        async with _pool.connection() as conn:
            result = await conn.execute(
                "UPDATE pending_record_modifications SET status = 'failed' "
                "WHERE status = 'pending' AND retry_count >= %s",
                (_WAL_MAX_RETRIES,),
            )
            await conn.commit()
            if result.rowcount > 0:
                logger.warning("WAL: Marked %d entries as 'failed' (exceeded %d retries).", result.rowcount, _WAL_MAX_RETRIES)
    except Exception as e:
        logger.error("WAL: Failed to mark exhausted entries: %s", e)

    # --- Step 2: Retry remaining pending entries ---
    try:
        async with _pool.connection() as conn:
            rows = await conn.execute(
                "SELECT id, user_id, action, fact, retry_count "
                "FROM pending_record_modifications "
                "WHERE status = 'pending' AND retry_count < %s "
                "ORDER BY created_at ASC",
                (_WAL_MAX_RETRIES,),
            )
            pending = await rows.fetchall()
    except Exception as e:
        logger.error("WAL: Failed to query pending modifications: %s", e)
        return

    if not pending:
        logger.info("WAL: No pending modifications to retry.")
        return

    logger.info("WAL: Retrying %d pending modification(s)...", len(pending))
    client = await _get_http_client()

    for row_id, user_id, action, fact, retry_count in pending:
        try:
            payload = {
                "userId": user_id,
                "action": action,
                "fact": fact,
                "source": "AI_Triage_Supervisor_WAL_Retry",
            }
            response = await client.post(
                f"{SEHATECH_API_BASE}/update-records",
                json=payload,
            )

            if response.status_code in (200, 201):
                async with _pool.connection() as conn:
                    await conn.execute(
                        "UPDATE pending_record_modifications SET status = 'synced' WHERE id = %s",
                        (row_id,),
                    )
                    await conn.commit()
                logger.info("WAL: Successfully synced row %d — %s '%s' for user %s.", row_id, action, fact, user_id)
            else:
                async with _pool.connection() as conn:
                    await conn.execute(
                        "UPDATE pending_record_modifications SET retry_count = retry_count + 1 WHERE id = %s",
                        (row_id,),
                    )
                    await conn.commit()
                logger.warning("WAL: API returned %d for row %d (retry %d/%d).", response.status_code, row_id, retry_count + 1, _WAL_MAX_RETRIES)

        except Exception as e:
            async with _pool.connection() as conn:
                await conn.execute(
                    "UPDATE pending_record_modifications SET retry_count = retry_count + 1 WHERE id = %s",
                    (row_id,),
                )
                await conn.commit()
            logger.warning("WAL: Retry failed for row %d: %s", row_id, e)


# ============================================================================
# BACKGROUND WAL DRAIN LOOP — retries pending modifications periodically
# ============================================================================

_WAL_DRAIN_INTERVAL = 60  # seconds
_wal_drain_task: asyncio.Task | None = None


async def _wal_drain_loop() -> None:
    """Background coroutine that drains the WAL every 60 seconds."""
    while True:
        await asyncio.sleep(_WAL_DRAIN_INTERVAL)
        try:
            await retry_pending_modifications()
        except Exception as e:
            logger.error("WAL drain loop error: %s", e)


def start_wal_drain_loop() -> None:
    """Start the background WAL drain task. Call once after event loop is running."""
    global _wal_drain_task
    if _wal_drain_task is None or _wal_drain_task.done():
        _wal_drain_task = asyncio.create_task(_wal_drain_loop())
        logger.info("Background WAL drain loop started (interval=%ds).", _WAL_DRAIN_INTERVAL)


# ============================================================================
# GRACEFUL SHUTDOWN — close HTTP client and DB pool
# ============================================================================

async def shutdown() -> None:
    """Cleanly close the HTTP client and DB connection pool."""
    global _http_client, _pool, _checkpointer, _setup_done, _wal_drain_task

    # Cancel WAL drain loop
    if _wal_drain_task and not _wal_drain_task.done():
        _wal_drain_task.cancel()
        try:
            await _wal_drain_task
        except asyncio.CancelledError:
            pass
        _wal_drain_task = None
        logger.info("WAL drain loop stopped.")

    # Close HTTP client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
        logger.info("HTTP client closed.")

    # Close DB pool
    if _pool:
        await _pool.close()
        _pool = None
        _checkpointer = None
        _setup_done = False
        logger.info("DB connection pool closed.")

