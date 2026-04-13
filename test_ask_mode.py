"""
End-to-end test for tracer.ask() to verify placeholder fix.
"""
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pylogtracer import LogTracer

# Create sample log file with MySQL errors and INC1000004
sample_log = """
2026-04-01 08:15:30 ERROR: INC1000004 - MySQL connection failed: [Errno 111] Connection refused
2026-04-01 08:16:45 INFO: INC1000004 - Retrying MySQL connection...
2026-04-01 08:17:12 ERROR: INC1000004 - MySQL query timeout: SELECT * FROM users timed out after 30s
2026-04-01 08:18:00 CRITICAL: INC1000004 - Failed to establish connection after 3 retries
2026-04-01 08:18:30 INFO: INC1000004 - Root cause identified: Database server was down for maintenance
2026-04-01 08:20:00 INFO: INC1000004 - Incident resolved - MySQL service restored
2026-04-01 08:20:30 INFO: Application started
2026-04-01 09:05:22 ERROR: INC1000005 - MySQL syntax error: SELCT * FROM orders (typo in query)
2026-04-01 09:10:15 WARNING: INC1000005 - Slow MySQL query detected: 5.2s for report_generation
2026-04-01 10:30:45 ERROR: INC1000006 - MySQL access denied: User 'app'@'localhost' has no permission
2026-04-01 11:00:00 INFO: Database maintenance started
2026-04-01 11:05:10 ERROR: INC1000007 - MySQL deadlock detected: INSERT/UPDATE conflict on users table
2026-04-01 12:30:00 INFO: Task completed successfully
2026-04-01 13:45:22 ERROR: INC1000008 - Connection pool exhausted: No available MySQL connections
2026-04-01 13:45:22 INFO:  incident no INC1000004 prediction is {"problem_context":"1234567890","problem_behavior":"abc"}
2026-04-01 13:46:22 INFO: prediction completed for INC1000004 
2026-04-01 13:45:22 INFO: incident no INC1000004 prediction is {"problem_context":"1234567890","problem_behavior":"abc"}

"""

# Write sample log
log_file = Path("application_logs.txt")
log_file.write_text(sample_log)
print(f"[INFO] Created sample log file: {log_file}")

# Initialize tracer with ollama
tracer = LogTracer(
    file_path="application_logs.txt",
    llm_config={
        "provider": "ollama",
        "model": "qwen3:latest",
        "base_url": "http://localhost:11434"
    }
)

print("\n" + "="*70)
print("TESTING: tracer.ask() - Prediction result for INC1000004")
print("="*70)

try:
    answer = tracer.ask("what is the prediction result of INC1000004? and list out all the  logs related to this incident")
    print("\n[RESULT]")
    print("------------>", answer)
    
    # Check if answer is placeholder
    if "<write your answer here" in answer.lower():
        print("\n❌ FAILED: Answer still contains placeholder template!")
    elif "No answer generated" in answer:
        print("\n⚠️  WARNING: No answer generated (LLM might not be running)")
    else:
        print("\n✅ SUCCESS: Answer extracted properly (no placeholder)")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nNote: This test requires Ollama running at http://localhost:11434")
    print("If Ollama is not available, the test will fail.")
