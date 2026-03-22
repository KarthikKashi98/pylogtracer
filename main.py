from pylogtracer import LogTracer

# ── Initialize LogTracer ──────────────────────────────────────────
tracer = LogTracer(
    file_path  = "application_logs.log",
    llm_config = {
        "provider": "ollama",
        "model":    "qwen2.5:3b",            # change to your model
        "base_url": "http://localhost:11434"  # ollama default
    }
)

# ══════════════════════════════════════════════════════════════════
# LIBRARY MODE — no LLM needed
# ══════════════════════════════════════════════════════════════════

# print(tracer.summary())
print(tracer.error_frequency())
# print(tracer.error_frequency(date="2026-03-01"))
# print(tracer.errors_by_date("2026-03-01"))
# print(tracer.errors_in_range("2026-03-01 09:00:00", "2026-03-01 11:00:00"))
# print(tracer.last_incident())
# print(tracer.health_check())
# print(tracer.incident_duration())
# print(tracer.search("INC1000004"))
# print(tracer.get_related_logs("INC1000004"))
# print(tracer.get_entry_details("INC1000004"))

# ══════════════════════════════════════════════════════════════════
# AGENT MODE — LLM required
# ══════════════════════════════════════════════════════════════════

print("-------------->", tracer.ask("what is the last error?"))
# print("-------------->", tracer.ask("is the system healthy?"))
# print("-------------->", tracer.ask("what caused the crash?"))
# print("-------------->", tracer.ask("how many errors happened?"))
print("-------------->", tracer.ask("show me INC1000004 related logs"))
# print("-------------->", tracer.ask("how long did the last incident last?"))
# print("-------------->", tracer.ask("show errors between 9am and 10am"))
# print("-------------->", tracer.ask("what caused the crash and how long did it last?"))
# print("-------------->", tracer.ask("compare errors on March 1 vs March 2"))

# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE — LLM required
# ══════════════════════════════════════════════════════════════════

# print(tracer.root_cause_analysis())
# print(tracer.root_cause_analysis(date="2026-03-01"))
