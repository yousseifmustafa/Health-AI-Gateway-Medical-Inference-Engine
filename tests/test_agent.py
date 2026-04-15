import pytest
from langchain_core.messages import HumanMessage, AIMessage
from Langgraphs.supervisor_graph import make_graph

# ----------------------------------------------------------------------------
# Scenario TS-01: Happy Path - Complete Golden Triangle
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ts01_happy_path(mock_gemini_supervisor, mock_model_manager):
    app = await make_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="عندي صداع شديد بقاله 3 أيام ومعنديش أمراض مزمنة أو حساسية")],
        "image_url": None,
        "conversation_summary": "",
        "user_id": "test_user_1",
        "patient_records": {}
    }
    
    # Mock LLM generating ToolCall for consult_doctor_tool
    mock_llm = mock_gemini_supervisor.bind_tools.return_value
    mock_llm.ainvoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "consult_doctor_tool",
            "args": {"symptom_description": "صداع شديد", "medical_history": "لا يوجد"},
            "id": "call_doctor_123"
        }]
    )
    
    result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": "ts01"}})
    
    # Verify no flag was tripped
    full_state = await app.aget_state({"configurable": {"thread_id": "ts01"}})
    assert full_state.values.get("conflict_flag") is False
    
    # Verify tools were invoked
    final_messages = result.get("messages", [])
    assert len(final_messages) > 1
    assert any("tool_calls" in dict(msg) or hasattr(msg, "tool_calls") for msg in final_messages)

# ----------------------------------------------------------------------------
# Scenario TS-02: Incomplete Data - Missing Duration/History
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ts02_incomplete_data(mock_gemini_supervisor, mock_model_manager):
    app = await make_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="بطني بتوجعني جداً")],
        "image_url": None,
        "conversation_summary": "",
        "user_id": "test_user_2",
        "patient_records": {}
    }
    
    # Mock LLM to NOT call tools, but ask a question
    mock_llm = mock_gemini_supervisor.bind_tools.return_value
    mock_llm.ainvoke.return_value = AIMessage(content="منذ متى بدأ هذا الألم؟ وهل تعاني من أمراض مزمنة؟")
    
    result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": "ts02"}})
    
    # Strictly assert internal state
    full_state = await app.aget_state({"configurable": {"thread_id": "ts02"}})
    assert full_state.values.get("conflict_flag") is False
    
    # Strictly assert final message
    final_msg = result["messages"][-1]
    assert not hasattr(final_msg, "tool_calls") or not final_msg.tool_calls
    assert "متى بدأ" in final_msg.content or "أمراض مزمنة" in final_msg.content


# ----------------------------------------------------------------------------
# Scenario TS-04: Vision Target - Image Upload Flow
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ts04_vision_target(mock_gemini_supervisor, mock_model_manager):
    app = await make_graph()
    initial_state = {
        "messages": [HumanMessage(content="إيه الدوا ده؟")],
        "image_url": "https://fake-cloudinary.com/box.jpg",
        "conversation_summary": "",
        "user_id": "test_user_4",
        "patient_records": {}
    }
    
    mock_llm = mock_gemini_supervisor.bind_tools.return_value
    mock_llm.ainvoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "analyze_medical_image_tool",
            "args": {"query": "إيه الدوا ده؟"},
            "id": "call_vision_123"
        }]
    )
    
    result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": "ts04"}})
    final_msgs = result.get("messages", [])
    
    assert any(
        hasattr(msg, "tool_calls") and any(tc["name"] == "analyze_medical_image_tool" for tc in msg.tool_calls)
        for msg in final_msgs
    )

# ----------------------------------------------------------------------------
# Scenario TS-05: Emergency Risk - Notify Family
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ts05_emergency_risk(mock_gemini_supervisor, mock_model_manager):
    app = await make_graph()
    initial_state = {
        "messages": [HumanMessage(content="أشعر بألم شديد بالصدر، كلم أهلي فوراً")],
        "image_url": None,
        "conversation_summary": "",
        "user_id": "test_user_5",
        "patient_records": {}
    }
    
    mock_llm = mock_gemini_supervisor.bind_tools.return_value
    mock_llm.ainvoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "notify_family_tool",
            "args": {"message": "المريض يحتاج لتدخل، يبلغ عن ألم بالصدر", "urgency_level": "High"},
            "id": "call_notify_123"
        }]
    )
    
    result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": "ts05"}})
    
    tool_called = False
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "notify_family_tool":
                    tool_called = True
                    assert tc["args"]["urgency_level"] == "High"
    
    assert tool_called is True, "notify_family_tool was not called for emergency risk."

# ----------------------------------------------------------------------------
# Scenario TS-06: Out of Scope / General Search
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ts06_out_of_scope(mock_gemini_supervisor, mock_model_manager):
    app = await make_graph()
    initial_state = {
        "messages": [HumanMessage(content="أقدر ألاقي مستشفى قريبة فين؟")],
        "image_url": None,
        "conversation_summary": "",
        "user_id": "test_user_6",
        "patient_records": {}
    }
    
    mock_llm = mock_gemini_supervisor.bind_tools.return_value
    mock_llm.ainvoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "web_search_tool",
            "args": {"query": "مستشفيات قريبة"},
            "id": "call_search_123"
        }]
    )
    
    result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": "ts06"}})
    assert any(
        hasattr(msg, "tool_calls") and any(tc["name"] == "web_search_tool" for tc in msg.tool_calls)
        for msg in result.get("messages", [])
    )
