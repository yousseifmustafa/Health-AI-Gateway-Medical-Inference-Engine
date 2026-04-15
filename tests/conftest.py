import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from langchain_core.documents import Document


@pytest.fixture
def mock_gemini_diagnose():
    """Mocks the LLM calls inside Diagnose_graph.py to avoid API costs."""
    with patch("Langgraphs.Diagnose_graph.ChatGoogleGenerativeAI") as MockLLM:
        mock_instance = MockLLM.return_value
        
        # 1. Mock standard ainvoke (generate_node)
        mock_instance.ainvoke = AsyncMock(return_value=AIMessage(content="[Mocked Medical Diagnosis]"))
        
        # 2. Mock structured output (grade_node)
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=MagicMock(score=0.95)) 
        mock_instance.with_structured_output.return_value = mock_structured
        
        yield mock_instance


@pytest.fixture
def mock_gemini_supervisor():
    """Mocks the LLM calls inside supervisor_graph.py to control agent routing behavior."""
    with patch("Langgraphs.supervisor_graph.ChatGoogleGenerativeAI") as MockLLM:
        mock_instance = MockLLM.return_value
        
        # Mock standard bind_tools.ainvoke (agent_node)
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.ainvoke = AsyncMock(
            return_value=AIMessage(content="ألف سلامة عليك، هل هناك أي تاريخ مرضي؟")
        )
        mock_instance.bind_tools.return_value = mock_llm_with_tools
        
        yield mock_instance


@pytest.fixture
def mock_retriever():
    """Mocks the Zilliz vector database string retrieval."""
    with patch("Langgraphs.Diagnose_graph._get_retriever") as mock_get_r:
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.ainvoke = AsyncMock(return_value=[
            Document(page_content="Paracetamol is used for mild to moderate pain.")
        ])
        
        mock_get_r.return_value = mock_retriever_instance
        yield mock_get_r


@pytest.fixture
def mock_model_manager():
    """Mocks the global Model_Manager singleton logic (like API key fetch, translation, etc)."""
    with patch("Models.Model_Manager.get_model_manager") as mock_get_mm:
        mock_mm = MagicMock()
        mock_mm.google_key_manager.get_next_api_key.return_value = "fake-mock-api-key"
        
        # Mocking query optimization
        mock_mm.optimize_query.ainvoke = AsyncMock(return_value="mock_translated_query")
        
        mock_get_mm.return_value = mock_mm
        yield mock_mm
