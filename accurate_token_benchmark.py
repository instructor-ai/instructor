#!/usr/bin/env python3
"""
Accurate token efficiency benchmark using tiktoken for actual token counts.
Compares JSON vs YAML formats for structured LLM outputs.
"""

import json
import yaml
from typing import Dict, Any, List


def get_actual_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Get actual token count using tiktoken (OpenAI tokenizer)."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to character-based approximation if tiktoken not available
        print("⚠️  tiktoken not installed, using character approximation")
        return max(1, len(text.strip()) // 4)


def create_test_cases() -> Dict[str, Dict[str, Any]]:
    """Create diverse test cases representing typical LLM extraction scenarios."""

    return {
        "Simple Person": {
            "name": "John Smith",
            "age": 35,
            "email": "john.smith@email.com",
            "occupation": "software engineer",
            "interests": ["hiking", "photography", "cooking"],
            "address": "123 Main Street, Anytown, USA",
        },
        "Company Info": {
            "name": "Microsoft Corporation",
            "industry": "technology",
            "founded_year": 1975,
            "employees": 220000,
            "headquarters": "Redmond, Washington",
            "revenue_millions": 211000.0,
            "is_public": True,
            "subsidiaries": ["LinkedIn", "GitHub", "Skype", "Xbox Game Studios"],
        },
        "Nested Project": {
            "project_name": "AI Assistant",
            "version": "2.1.0",
            "description": "An intelligent assistant for automated tasks",
            "team_members": [
                {"name": "Alice Johnson", "role": "Lead Developer", "experience": 8},
                {"name": "Bob Wilson", "role": "Data Scientist", "experience": 5},
                {"name": "Carol Davis", "role": "Product Manager", "experience": 12},
            ],
            "technologies": {
                "backend": ["Python", "FastAPI", "PostgreSQL"],
                "frontend": ["React", "TypeScript", "TailwindCSS"],
                "ai_models": ["GPT-4", "BERT", "ResNet"],
            },
            "deployment": {
                "environment": "production",
                "platform": "AWS",
                "containers": True,
                "scaling": {"min_instances": 2, "max_instances": 10},
            },
        },
        "Product Catalog": {
            "products": [
                {
                    "id": "PROD001",
                    "name": "Laptop",
                    "price": 999.99,
                    "specs": {"cpu": "Intel i7", "ram": "16GB", "storage": "512GB SSD"},
                    "tags": ["electronics", "computers", "portable"],
                },
                {
                    "id": "PROD002",
                    "name": "Mouse",
                    "price": 29.99,
                    "specs": {"type": "wireless", "dpi": 1600, "buttons": 5},
                    "tags": ["electronics", "peripherals", "accessories"],
                },
            ],
            "total_items": 2,
            "last_updated": "2025-10-07",
        },
    }


def benchmark_with_tiktoken():
    """Run comprehensive benchmark using actual token counts."""

    print("=" * 70)
    print("ACCURATE TOKEN EFFICIENCY BENCHMARK (Using tiktoken)")
    print("=" * 70)
    print()

    try:
        import tiktoken

        print("✅ Using tiktoken (cl100k_base encoding) for accurate token counts\n")
    except ImportError:
        print("⚠️  tiktoken not installed - using character approximation")
        print("   Install with: uv pip install tiktoken\n")

    test_cases = create_test_cases()
    all_results = []

    for case_name, data in test_cases.items():
        print(f"📊 Testing: {case_name}")
        print("-" * 70)

        # Generate different formats
        json_pretty = json.dumps(data, indent=2, ensure_ascii=False)
        json_compact = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        yaml_standard = yaml.dump(
            data, default_flow_style=False, allow_unicode=True, width=80
        )
        yaml_compact = yaml.dump(data, default_flow_style=True, allow_unicode=True)

        # Get actual token counts
        tokens_json_pretty = get_actual_token_count(json_pretty)
        tokens_json_compact = get_actual_token_count(json_compact)
        tokens_yaml_standard = get_actual_token_count(yaml_standard)
        tokens_yaml_compact = get_actual_token_count(yaml_compact)

        # Calculate efficiency percentages
        yaml_vs_json_pretty = (
            (tokens_json_pretty - tokens_yaml_standard) / tokens_json_pretty
        ) * 100
        yaml_compact_vs_json_pretty = (
            (tokens_json_pretty - tokens_yaml_compact) / tokens_json_pretty
        ) * 100
        json_compact_vs_pretty = (
            (tokens_json_pretty - tokens_json_compact) / tokens_json_pretty
        ) * 100

        # Store results
        result = {
            "name": case_name,
            "json_pretty": tokens_json_pretty,
            "json_compact": tokens_json_compact,
            "yaml_standard": tokens_yaml_standard,
            "yaml_compact": tokens_yaml_compact,
            "yaml_efficiency": yaml_vs_json_pretty,
            "yaml_compact_efficiency": yaml_compact_vs_json_pretty,
            "json_compact_efficiency": json_compact_vs_pretty,
        }
        all_results.append(result)

        # Display results
        print(f"  JSON Pretty:    {tokens_json_pretty:4d} tokens  (baseline)")
        print(
            f"  JSON Compact:   {tokens_json_compact:4d} tokens  ({json_compact_vs_pretty:+6.1f}%)"
        )
        print(
            f"  YAML Standard:  {tokens_yaml_standard:4d} tokens  ({yaml_vs_json_pretty:+6.1f}%)"
        )
        print(
            f"  YAML Compact:   {tokens_yaml_compact:4d} tokens  ({yaml_compact_vs_json_pretty:+6.1f}%)"
        )
        print()

        # Show example for first case
        if case_name == "Simple Person":
            print("  📝 Format Examples:")
            print(f"\n  JSON Pretty ({tokens_json_pretty} tokens):")
            for line in json_pretty.split("\n")[:6]:
                print(f"    {line}")
            print("    ...")

            print(f"\n  YAML Standard ({tokens_yaml_standard} tokens):")
            for line in yaml_standard.split("\n")[:6]:
                print(f"    {line}")
            print("    ...")
            print()

    # Calculate averages
    print("=" * 70)
    print("📈 SUMMARY STATISTICS")
    print("=" * 70)

    avg_yaml_efficiency = sum(r["yaml_efficiency"] for r in all_results) / len(
        all_results
    )
    avg_yaml_compact_efficiency = sum(
        r["yaml_compact_efficiency"] for r in all_results
    ) / len(all_results)
    avg_json_compact_efficiency = sum(
        r["json_compact_efficiency"] for r in all_results
    ) / len(all_results)

    print(f"\nAverage Token Savings vs JSON Pretty:")
    print(f"  YAML Standard:  {avg_yaml_efficiency:+6.1f}%")
    print(f"  YAML Compact:   {avg_yaml_compact_efficiency:+6.1f}%")
    print(f"  JSON Compact:   {avg_json_compact_efficiency:+6.1f}%")

    # Detailed comparison table
    print(f"\n{'Test Case':<20} {'JSON':<8} {'J.Cmp':<8} {'YAML':<8} {'Y.Cmp':<8}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['name']:<20} "
            f"{r['json_pretty']:>7d}  "
            f"{r['json_compact']:>7d}  "
            f"{r['yaml_standard']:>7d}  "
            f"{r['yaml_compact']:>7d}"
        )

    # Analysis
    print("\n" + "=" * 70)
    print("🔍 ANALYSIS")
    print("=" * 70)

    if avg_yaml_efficiency > 0:
        print(
            f"✅ YAML Standard is {avg_yaml_efficiency:.1f}% more token-efficient than JSON Pretty"
        )
    else:
        print(
            f"❌ YAML Standard uses {abs(avg_yaml_efficiency):.1f}% MORE tokens than JSON Pretty"
        )

    if avg_yaml_efficiency > avg_json_compact_efficiency:
        diff = avg_yaml_efficiency - avg_json_compact_efficiency
        print(f"✅ YAML Standard is {diff:.1f}% more efficient than JSON Compact")
    else:
        diff = avg_json_compact_efficiency - avg_yaml_efficiency
        print(f"⚠️  JSON Compact is {diff:.1f}% more efficient than YAML Standard")

    # Best format
    formats = [
        ("JSON Compact", avg_json_compact_efficiency),
        ("YAML Standard", avg_yaml_efficiency),
        ("YAML Compact", avg_yaml_compact_efficiency),
    ]
    best = max(formats, key=lambda x: x[1])

    print(f"\n🏆 Most Token-Efficient Format: {best[0]} ({best[1]:+.1f}% savings)")

    # Key insights
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)
    print('• YAML eliminates quotes around keys (e.g., name: vs "name":)')
    print("• YAML eliminates quotes around string values in many cases")
    print("• YAML uses indentation instead of braces/brackets for nesting")
    print('• YAML has cleaner list syntax (- item vs "item",)')
    print("• Benefits increase with nested structures and string-heavy data")
    print("• Trade-off: readability vs maximum compression")

    return all_results


if __name__ == "__main__":
    results = benchmark_with_tiktoken()

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
