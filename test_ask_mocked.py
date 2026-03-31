"""
Test ask() mode with mocked LLM to verify placeholder fix works.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pylogtracer import LogTracer
from langchain_core.messages import AIMessage

# Create sample log file with INC1000004
sample_log = """
2026-04-01 08:15:30 ERROR: INC1000004 - MySQL connection failed: [Errno 111] Connection refused
2026-04-01 08:16:45 INFO: INC1000004 - Retrying MySQL connection...
2026-04-01 08:17:12 ERROR: INC1000004 - MySQL query timeout: SELECT * FROM users timed out after 30s
2026-04-01 08:18:00 CRITICAL: INC1000004 - Failed to establish connection after 3 retries
2026-04-01 08:18:30 INFO: INC1000004 - Root cause identified: Database server was down for maintenance
2026-04-01 08:20:00 INFO: INC1000004 - Incident resolved - MySQL service restored
2026-04-01 08:20:30 INFO: Application started
"""

# Write sample log
log_file = Path("application_logs.txt")
log_file.write_text(sample_log)
print(f"[INFO] Created sample log file: {log_file}")

# Initialize tracer
tracer = LogTracer(
    file_path="application_logs.txt",
    llm_config={
        "provider": "ollama",
        "model": "qwen2.5:3b",
        "base_url": "http://localhost:11434"
    }
)

print("\n" + "="*70)
print("TEST: ask() with mocked LLM responses")
print("="*70)

# Mock the LLM to return predefined responses
def mock_llm_invoke(messages):
    """Simulate LLM responses for the agent loop."""
    # Find the last user message to determine what to respond with
    last_user_msg = None
    for msg in reversed(messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            last_user_msg = msg.content
            break
    
    # First turn: decide to search for INC1000004
    if "what is the prediction result" in (last_user_msg or "").lower():
        response = AIMessage(content="TOOL: get_related_logs\nARGS: {\"identifier\": \"INC1000004\"}\nREASON: To find all logs related to incident INC1000004")
    # Second turn: provide the final answer
    elif "TOOL_RESULT" in (last_user_msg or ""):
        response = AIMessage(content="FINAL_ANSWER: INC1000004 was a MySQL connection failure that occurred on 2026-04-01 from 08:15:30 to 08:20:00. The root cause was a database server maintenance window. The incident contained 2 ConnectionError occurrences and 1 TimeoutError. It was successfully resolved when MySQL service was restored.")
    else:
        response = AIMessage(content="FINAL_ANSWER: Unable to determine incident details")
    
    return response

# Patch the LLM factory
with patch.object(tracer._factory, 'get_llm') as mock_get_llm:
    mock_llm = Mock()
    mock_llm.invoke = mock_llm_invoke
    mock_get_llm.return_value = mock_llm
    
    try:
        answer = tracer.ask("what is the prediction result of INC1000004?")
        print("\n[RESULT]")
        print("------------>", answer)
        
        # Check results
        if "<write your answer here" in answer.lower() or "<specific answer" in answer.lower():
            print("\n❌ FAILED: Answer still contains placeholder template!")
        elif "No answer generated" in answer:
            print("\n⚠️  WARNING: No answer generated")
        elif "INC1000004" in answer:
            print("\n✅ SUCCESS: Real answer with INC1000004 details extracted properly!")
        else:
            print("\n✅ SUCCESS: Answer extracted (no placeholder)")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
