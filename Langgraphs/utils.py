import re
import threading
import logging
from Models.Model_Manager import get_model_manager
from vector_db.vdb_connection import create_retriever

logger = logging.getLogger("sehatech.utils")

# CONFLICT DETECTOR UTILITIES

# Negation patterns for detecting contradictions
_NEGATION_PATTERNS = [
    r"\bno\s+(allergies|allergy|medications?|chronic|conditions?|surgeries|surgery|history)\b",
    r"\bi\s+(don'?t|do\s+not)\s+(take|have|use)\b",
    r"\bi\s+(stopped|quit|finished|completed)\b",
    r"\bnot\s+(taking|on|using)\b",
    r"\b(مفيش|ماعنديش|مش|ماباخدش|مابخدش|مابستخدمش|لا|مش بآخد|وقفت|خلصت)\b",
    r"\bno\s+medical\s+history\b",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)

def _detect_conflicts(user_text: str, records: dict) -> list[str]:
    """
    Deterministic conflict detection — zero LLM cost, zero latency.
    Compares user's latest message against fetched permanent records.
    Returns a list of conflict description strings (empty if no conflicts).
    """
    conflicts = []
    if not records or not user_text:
        return conflicts

    text_lower = user_text.lower()
    has_negation = bool(_NEGATION_RE.search(user_text))

    if not has_negation:
        return conflicts

    # Check: "no allergies" vs non-empty allergy list
    allergies = records.get("allergies", [])
    if allergies and re.search(r"(no\s+allerg|مفيش\s*حساسي|ماعنديش\s*حساسي)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO allergies, but records show: {', '.join(allergies)}"
        )

    # Check: "no medications" vs non-empty medication list
    meds = records.get("current_medications", [])
    if meds and re.search(r"(no\s+medic|don'?t\s+take|not\s+taking|مش\s*بآخد|ماباخدش|مابخدش)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO medications, but records show active prescriptions: {', '.join(meds)}"
        )

    # Check: "no chronic conditions" vs non-empty chronic list
    chronic = records.get("chronic_diseases", [])
    if chronic and re.search(r"(no\s+chronic|no\s+condition|don'?t\s+have\s+any|مفيش\s*أمراض|ماعنديش)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO chronic conditions, but records show: {', '.join(chronic)}"
        )

    # Check: "no surgeries" vs non-empty surgical history
    surgeries = records.get("surgical_history", [])
    if surgeries and re.search(r"(no\s+surg|never\s+had\s+surg|مفيش\s*عملي|ماعملتش)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO surgical history, but records show: {', '.join(surgeries)}"
        )

    return conflicts

# DIAGNOSIS RETRIEVER UTILITIES
class _RetrieverHolder:
    """Lazy singleton for the vector DB retriever only. 
    ModelManager comes from the shared singleton."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._retriever = None
        return cls._instance

    @property
    def context_retriever(self):
        """Lazy-load the retriever on first access (requires embedding model)."""
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    logger.info("Lazy-initializing Zilliz Retriever...")
                    mm = get_model_manager()
                    retriever = create_retriever(mm.embedding_model)
                    if retriever is None:
                        raise RuntimeError(
                            "Failed to create Zilliz retriever — check ZILLIZ_URI and ZILLIZ_TOKEN."
                        )
                    self._retriever = retriever
        return self._retriever

def _get_retriever():
    return _RetrieverHolder().context_retriever
