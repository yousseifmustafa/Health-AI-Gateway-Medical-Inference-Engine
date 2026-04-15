from typing import Annotated, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# SUPERVISOR GRAPH STATES
class SupervisorInputState(BaseModel):
    """Schema for data accepted FROM the client (server.py / Flutter app)."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    messages: Annotated[list[BaseMessage], add_messages] = []
    image_url: Optional[str] = None
    conversation_summary: str = ""
    user_id: Optional[str] = None

class SupervisorOutputState(BaseModel):
    """Schema for data returned TO the client — no internal flags or records."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    messages: Annotated[list[BaseMessage], add_messages] = []
    conversation_summary: str = ""

class AgentState(BaseModel):
    """Internal superset — contains every field nodes may read or write."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    messages: Annotated[list[BaseMessage], add_messages] = []
    image_url: Optional[str] = None
    conversation_summary: str = ""
    user_id: Optional[str] = None
    patient_records: Optional[dict] = None
    conflict_flag: bool = False
    conflict_details: Optional[str] = None


# DIAGNOSE GRAPH STATES
class DiagnoseInputState(BaseModel):
    """Schema for data accepted FROM the supervisor's consult_doctor_tool."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    user_query: str = ""
    conversation_summary: str = ""
    translated_query: str = ""
    expanded_queries: list[str] = []

class DiagnoseOutputState(BaseModel):
    """Schema for data returned TO the caller — no RAG chunks or scores."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    final_answer: Optional[str] = None

class DiagnoseState(BaseModel):
    """Internal superset — contains every field nodes may read or write."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    user_query: str = ""
    conversation_summary: str = ""

    # Pre-optimized by the Supervisor — passed as initial inputs
    translated_query: str = ""
    expanded_queries: list[str] = []

    # Unified Answer Field
    english_medical_answer: str = ""
    confidence_score: float = 0.0

    # RAG Fields (strings only — no heavy Document objects in state)
    support_contents: list[str] = []
    final_docs: list[str] = []

    final_answer: Optional[str] = None

class GradeOutput(BaseModel):
    score: float = Field(description="Confidence score between 0.0 and 1.0 regarding medical accuracy and safety.")
