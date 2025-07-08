#!/usr/bin/env python3
"""Debug OpenAI batch job"""

import os
from openai import OpenAI

# Read the batch ID
with open("openai_batch_id.txt", "r") as f:
    batch_id = f.read().strip()

print(f"Debugging OpenAI batch: {batch_id}")

client = OpenAI()

# Get detailed batch information
batch = client.batches.retrieve(batch_id)
print(f"Status: {batch.status}")
print(f"Created at: {batch.created_at}")
print(f"Completed at: {batch.completed_at}")
print(f"Total requests: {batch.request_counts.total}")
print(f"Completed requests: {batch.request_counts.completed}")
print(f"Failed requests: {batch.request_counts.failed}")
print(f"Input file ID: {batch.input_file_id}")
print(f"Output file ID: {batch.output_file_id}")
print(f"Error file ID: {batch.error_file_id}")

# Check if there's an error file
if batch.error_file_id:
    print("\n=== ERROR FILE CONTENT ===")
    error_content = client.files.content(batch.error_file_id)
    print(error_content.read().decode('utf-8'))

# Check if there's an output file
if batch.output_file_id:
    print("\n=== OUTPUT FILE CONTENT ===")
    output_content = client.files.content(batch.output_file_id)
    print(output_content.read().decode('utf-8'))
else:
    print("\n❌ No output file available")