#!/usr/bin/env python3
"""Debug script to check how batch files are created"""

from instructor.batch import BatchProcessor
from pydantic import BaseModel
from typing import List

class User(BaseModel):
    name: str
    age: int

def create_test_messages() -> List[List[dict]]:
    """Create 4 test message conversations for user extraction"""
    test_prompts = [
        "Hi there! My name is Alice and I'm 28 years old. I work as a software engineer.",
        "Hello, I'm Bob, 35 years old, and I love hiking and photography.",
        "This is Sarah speaking. I'm 42 and I'm a graphic designer.",
        "Hey! I'm Mike, 31, and I'm a teacher at the local high school.",
    ]

    messages_list = []
    for prompt in test_prompts:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at extracting structured user information from text. Extract the person's name and age.",
            },
            {"role": "user", "content": prompt},
        ]
        messages_list.append(messages)

    return messages_list

def debug_batch_creation():
    """Debug batch file creation"""
    model = "anthropic/claude-3-5-sonnet-20241022"
    messages_list = create_test_messages()
    
    print(f"Number of message conversations: {len(messages_list)}")
    
    processor = BatchProcessor(model, User)
    
    # Debug: let's check what happens when we create the batch
    batch_filename = "debug_batch.jsonl"
    processor.create_batch_from_messages(
        file_path=batch_filename,
        messages_list=messages_list,
        max_tokens=200,
        temperature=0.1,
    )
    
    # Read the created file and check its contents
    with open(batch_filename, 'r') as f:
        lines = f.readlines()
    
    print(f"Number of lines in batch file: {len(lines)}")
    
    for i, line in enumerate(lines):
        print(f"\n--- Line {i+1} ---")
        print(line.strip())

if __name__ == "__main__":
    debug_batch_creation()
