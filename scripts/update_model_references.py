import os
import re
from pathlib import Path
from typing import List, Tuple, Union

def update_model_reference(content: str) -> str:
    """Update model references to use gpt-4-turbo-preview."""
    # Define patterns to replace
    patterns = [
        (r'model="gpt-3\.5-turbo[^"]*"', 'model="gpt-4-turbo-preview"'),
        (r'model="gpt-4"', 'model="gpt-4-turbo-preview"'),
        (r'model="gpt-4o[^"]*"', 'model="gpt-4-turbo-preview"'),
        # Add more patterns if needed
    ]

    updated_content = content
    for old_pattern, new_value in patterns:
        updated_content = re.sub(old_pattern, new_value, updated_content)

    return updated_content

def process_file(file_path: Union[str, Path]) -> None:
    """Process a single file and update model references."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    updated_content = update_model_reference(content)

    if content != updated_content:
        print(f"Updating {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

def main() -> None:
    # Define directories to process
    dirs_to_process = [
        "docs/examples",
        "docs/prompting",
        "docs"
    ]

    # Process each directory
    for dir_path in dirs_to_process:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    process_file(file_path)

if __name__ == "__main__":
    main()
