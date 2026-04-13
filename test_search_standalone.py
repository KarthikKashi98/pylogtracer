#!/usr/bin/env python
"""
Standalone test to verify search tool prints
"""
from pathlib import Path
from pylogtracer import LogTracer

# Create sample log
sample_log = """2026-04-01 08:15:30 ERROR: INC1000004 - MySQL connection failed
2026-04-01 08:16:45 INFO: INC1000004 - Retrying MySQL connection
2026-04-01 13:45:22 INFO: incident no INC1000004 prediction is {"problem_context":"1234567890"}
"""

log_file = Path("test_search.txt")
log_file.write_text(sample_log)

tracer = LogTracer(file_path=str(log_file))

print("\n" + "="*70)
print("CALLING tracer.search('INC1000004')")
print("="*70 + "\n")

result = tracer.search("INC1000004")

print("\n" + "="*70)
print(f"RESULT: Found {result['total_found']} entries")
print("="*70)

log_file.unlink()
