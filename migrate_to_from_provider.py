#!/usr/bin/env python3
"""
Script to migrate documentation from provider-specific functions to from_provider().

Usage: python migrate_to_from_provider.py [--dry-run] [file1] [file2] ...
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "openai/gpt-5-nano",
    "anthropic": "anthropic/claude-3-haiku-20240307",
    "google": "google/gemini-2.5-flash",
    "groq": "groq/llama3-70b-8192",
    "cohere": "cohere/command-r-plus",
    "cerebras": "cerebras/llama3.1-70b",
    "fireworks": "fireworks/llama-v3p2-1b-instruct",
}


def migrate_provider_calls(content: str, filename: str) -> Tuple[str, int]:
    """Migrate provider-specific calls to from_provider()."""
    changes = 0
    original = content

    # Pattern 1: from_openai with OpenAI() - sync
    pattern = r'instructor\.from_openai\(\s*(?:openai\.)?OpenAI\([^)]*\)\s*(?:,\s*mode\s*=\s*(instructor\.Mode\.\w+))?\s*\)'
    def replace_openai_sync(match):
        mode = match.group(1)
        if mode:
            return f'instructor.from_provider("{DEFAULT_MODELS["openai"]}", mode={mode})'
        return f'instructor.from_provider("{DEFAULT_MODELS["openai"]}")'
    content = re.sub(pattern, replace_openai_sync, content)

    # Pattern 2: from_openai with AsyncOpenAI() - async
    pattern = r'instructor\.from_openai\(\s*(?:openai\.)?AsyncOpenAI\([^)]*\)\s*(?:,\s*mode\s*=\s*(instructor\.Mode\.\w+))?\s*\)'
    def replace_openai_async(match):
        mode = match.group(1)
        if mode:
            return f'instructor.from_provider("{DEFAULT_MODELS["openai"]}", async_client=True, mode={mode})'
        return f'instructor.from_provider("{DEFAULT_MODELS["openai"]}", async_client=True)'
    content = re.sub(pattern, replace_openai_async, content)

    # Pattern 3: from_anthropic with Anthropic() - sync
    pattern = r'instructor\.from_anthropic\(\s*(?:anthropic\.)?Anthropic\([^)]*\)\s*(?:,\s*mode\s*=\s*(instructor\.Mode\.\w+))?\s*\)'
    def replace_anthropic_sync(match):
        mode = match.group(1)
        if mode:
            return f'instructor.from_provider("{DEFAULT_MODELS["anthropic"]}", mode={mode})'
        return f'instructor.from_provider("{DEFAULT_MODELS["anthropic"]}")'
    content = re.sub(pattern, replace_anthropic_sync, content)

    # Pattern 4: from_anthropic with AsyncAnthropic() - async
    pattern = r'instructor\.from_anthropic\(\s*(?:anthropic\.)?AsyncAnthropic\([^)]*\)\s*(?:,\s*mode\s*=\s*(instructor\.Mode\.\w+))?\s*\)'
    def replace_anthropic_async(match):
        mode = match.group(1)
        if mode:
            return f'instructor.from_provider("{DEFAULT_MODELS["anthropic"]}", async_client=True, mode={mode})'
        return f'instructor.from_provider("{DEFAULT_MODELS["anthropic"]}", async_client=True)'
    content = re.sub(pattern, replace_anthropic_async, content)

    # Pattern 5: from_gemini
    pattern = r'instructor\.from_gemini\([^)]+\)'
    content = re.sub(pattern, f'instructor.from_provider("{DEFAULT_MODELS["google"]}")', content)

    # Pattern 6: from_groq
    pattern = r'instructor\.from_groq\(\s*(?:groq\.)?(?:Async)?Groq\([^)]*\)\s*\)'
    content = re.sub(pattern, f'instructor.from_provider("{DEFAULT_MODELS["groq"]}")', content)

    # Pattern 7: from_cohere
    pattern = r'instructor\.from_cohere\(\s*(?:cohere\.)?(?:Async)?ClientV?2?\([^)]*\)\s*\)'
    content = re.sub(pattern, f'instructor.from_provider("{DEFAULT_MODELS["cohere"]}")', content)

    # Pattern 8: from_cerebras
    pattern = r'instructor\.from_cerebras\(\s*(?:Async)?Cerebras\([^)]*\)\s*\)'
    content = re.sub(pattern, f'instructor.from_provider("{DEFAULT_MODELS["cerebras"]}")', content)

    # Pattern 9: from_fireworks
    pattern = r'instructor\.from_fireworks\(\s*(?:Async)?Fireworks\([^)]*\)\s*\)'
    content = re.sub(pattern, f'instructor.from_provider("{DEFAULT_MODELS["fireworks"]}")', content)

    # Pattern 10: from_vertexai - special case, needs vertexai=True
    pattern = r'instructor\.from_vertexai\([^)]+\)'
    content = re.sub(pattern, f'instructor.from_provider("{DEFAULT_MODELS["google"]}", vertexai=True)', content)

    # Pattern 11: from_litellm - keep as is for now since it's a proxy
    # We'll skip this one as it needs to pass through model names

    # Clean up common import patterns
    # Remove provider SDK imports when no longer needed
    lines = content.split('\n')
    new_lines = []
    skip_next_blank = False

    for i, line in enumerate(lines):
        # Check if this import is no longer needed
        if re.match(r'^\s*from\s+(openai|anthropic|groq|cohere)\s+import\s+', line):
            # Check if the provider name appears elsewhere in the file (not in from_provider calls)
            provider = re.search(r'from\s+(\w+)\s+import', line).group(1)
            # Look ahead to see if it's actually used
            rest_of_file = '\n'.join(lines[i+1:])
            # If the provider class isn't used directly, we can remove the import
            if provider == 'openai' and not re.search(r'\bOpenAI\((?!.*from_provider)', rest_of_file):
                skip_next_blank = True
                continue
            elif provider == 'anthropic' and not re.search(r'\bAnthropic\((?!.*from_provider)', rest_of_file):
                skip_next_blank = True
                continue
            elif provider == 'groq' and not re.search(r'\bGroq\((?!.*from_provider)', rest_of_file):
                skip_next_blank = True
                continue
            elif provider == 'cohere' and not re.search(r'\bClient(?:V2)?\((?!.*from_provider)', rest_of_file):
                skip_next_blank = True
                continue

        # Skip blank line after removed import
        if skip_next_blank and line.strip() == '':
            skip_next_blank = False
            continue

        new_lines.append(line)

    content = '\n'.join(new_lines)

    if content != original:
        changes = len(re.findall(r'from_provider', content)) - len(re.findall(r'from_provider', original))

    return content, changes


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single file."""
    try:
        content = filepath.read_text(encoding='utf-8')

        # Check if file needs migration
        if not re.search(r'from_(openai|anthropic|gemini|groq|cohere|cerebras|fireworks|vertexai)\(', content):
            return False

        new_content, changes = migrate_provider_calls(content, str(filepath))

        if changes > 0:
            if dry_run:
                print(f"Would update {filepath} ({changes} provider calls migrated)")
            else:
                filepath.write_text(new_content, encoding='utf-8')
                print(f"Updated {filepath} ({changes} provider calls migrated)")
            return True

        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Migrate docs to from_provider()')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without modifying files')
    parser.add_argument('files', nargs='*', help='Specific files to process (default: all docs/*.md)')

    args = parser.parse_args()

    # Get files to process
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        docs_dir = Path('/home/user/instructor/docs')
        files = list(docs_dir.rglob('*.md'))

    # Process files
    updated = 0
    for filepath in sorted(files):
        if process_file(filepath, args.dry_run):
            updated += 1

    action = "Would update" if args.dry_run else "Updated"
    print(f"\n{action} {updated} file(s)")


if __name__ == '__main__':
    main()
