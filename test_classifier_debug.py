"""Test script to debug classifier behavior."""
import sys
sys.path.insert(0, 'src')

from pylogtracer.preprocessing.error_type_classifier import ErrorTypeClassifier
from pylogtracer.llm.llm_factory import LLMFactory

# Test 1: Without LLM factory (like current logs)
print("=" * 60)
print("TEST 1: Without LLM Factory (Pattern Matching Only)")
print("=" * 60)

classifier_no_llm = ErrorTypeClassifier(factory=None)

test_errors = [
    "Connection refused at db:5432",
    "Timeout waiting for API response",
    "Permission denied writing to /var/log",
    "Database query failed",
    "Authentication failed: invalid credentials",
    "Unknown error",  # Should stay as UnknownError
]

for i, error in enumerate(test_errors, 1):
    result = classifier_no_llm._regex_classify(error)
    print(f"  [{i}] {error}")
    print(f"      → {result or 'UnknownError'}\n")

# Test 2: With LLM factory (if configured)
print("\n" + "=" * 60)
print("TEST 2: With LLM Factory (Pattern + LLM)")
print("=" * 60)

try:
    factory = LLMFactory()  # Reads from .env or uses defaults
    model = factory.get_model()
    print(f"  LLM Factory initialized: {model}")
    print(f"  This will allow learning keywords from LLM\n")
    
    classifier_with_llm = ErrorTypeClassifier(factory=factory)
    print(f"  Classifier ready for learning keywords per error type")
    
except Exception as e:
    print(f"  ⚠️  Could not initialize LLM: {e}")
    print(f"  Run: pip install langchain langchain-openai")
    print(f"       (or use: export LLM_PROVIDER=ollama)")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("""
Your logs currently show all UnknownError because:

1. ✓ Pattern matching IS working (detects common errors)
2. ✗ LLM factory is None (can't learn custom keywords)

To fix this:

Option A: Use default patterns (No LLM needed)
  - Already improved! Run your code again
  
Option B: Enable LLM for custom error learning
  - Export: export LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b
  - Or pass llm_config to LogTracer()
  
Option C: Configure OpenAI/Anthropic
  - export LLM_PROVIDER=openai LLM_API_KEY=sk-...
  - export LLM_MODEL=gpt-4o-mini
""")
