#!/usr/bin/env python3
"""Wait for OpenAI batch to complete and then fetch results"""

import time
from openai import OpenAI

# Read the batch ID
with open("openai_batch_id.txt", "r") as f:
    batch_id = f.read().strip()

print(f"Monitoring OpenAI batch: {batch_id}")

client = OpenAI()

# Check status every 30 seconds for up to 10 minutes
max_attempts = 20
attempt = 0

while attempt < max_attempts:
    batch = client.batches.retrieve(batch_id)
    print(f"Attempt {attempt + 1}: Status = {batch.status}")
    
    if batch.status == "completed":
        print("✅ Batch completed! Fetching results...")
        break
    elif batch.status in ["failed", "expired", "cancelled"]:
        print(f"❌ Batch {batch.status}")
        exit(1)
    
    attempt += 1
    if attempt < max_attempts:
        print("⏳ Waiting 30 seconds...")
        time.sleep(30)

if attempt >= max_attempts:
    print("⏰ Timeout waiting for batch to complete")
    exit(1)

# Fetch results using our CLI
import subprocess
result = subprocess.run(["uv", "run", "python", "run_batch_test.py", "fetch", "--provider", "openai"], 
                       capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)