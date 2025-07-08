#!/usr/bin/env python3
"""Debug Anthropic batch job"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from instructor.batch import BatchProcessor
from pydantic import BaseModel
import anthropic

class User(BaseModel):
    name: str
    age: int

# Read the batch ID
with open("anthropic_batch_id.txt", "r") as f:
    batch_id = f.read().strip()

print(f"Debugging Anthropic batch: {batch_id}")

client = anthropic.Anthropic()
batch = client.messages.batches.retrieve(batch_id)

print(f"Status: {batch.processing_status}")
print(f"Created at: {batch.created_at}")
print(f"Request counts: {getattr(batch, 'request_counts', 'Not available')}")

# Get raw results
print("\n=== RAW RESULTS ===")
results = client.messages.batches.results(batch_id)
for i, result in enumerate(results):
    print(f"\nResult {i}:")
    print(result.model_dump_json(indent=2))

# Test parsing with BatchProcessor
print("\n=== BATCHPROCESSOR PARSING ===")
processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)
results_lines = []
for result in results:
    results_lines.append(result.model_dump_json())

results_content = '\n'.join(results_lines)
all_results = processor.parse_results(results_content)

print(f"Total results: {len(all_results)}")
for result in all_results:
    if result.success:
        print(f"✅ Success ({result.custom_id}): {result.result}")
    else:
        print(f"❌ Error ({result.custom_id}): {result.error_message}")
        print(f"   Raw data: {result.raw_data}")