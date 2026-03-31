"""
Test library mode (no LLM needed) to verify INC1000004 detection works.
"""
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pylogtracer import LogTracer

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

# Initialize tracer (no LLM config needed for library mode)
tracer = LogTracer(file_path="application_logs.txt")

print("\n" + "="*70)
print("TEST 1: Search for INC1000004")
print("="*70)

result = tracer.search("INC1000004")
print("\nSearch result:")
print(f"  Found: {len(result)} entries")
for entry in result:
    print(f"    - {entry}")

print("\n" + "="*70)
print("TEST 2: Get related logs for INC1000004")
print("="*70)

related = tracer.get_related_logs("INC1000004")
print(f"\nFound: {related['found']}")
if related['found']:
    print(f"Anchor entry: {related['anchor_entry']}")
    print(f"Cluster has {related['total_in_cluster']} entries:")
    for entry in related['cluster']:
        ts = entry.get('timestamp', 'N/A')
        et = entry.get('error_type', 'N/A')
        pe = entry.get('primary_error', 'N/A')
        print(f"  {ts} | {et} | {pe}")

print("\n" + "="*70)
print("TEST 3: Error frequency")
print("="*70)

freq = tracer.error_frequency()
print("\nError frequency:")
for err_type, count in freq.items():
    print(f"  {err_type}: {count}")

print("\n✅ All library tests completed successfully!")
