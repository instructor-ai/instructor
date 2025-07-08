"""Anthropic Claude Batch API Example

This example demonstrates how to use Anthropic's Message Batches API to process large volumes
of requests asynchronously at 50% cost savings with sub-hour processing for most batches.

Features:
- Create batch requests with custom IDs
- Monitor batch processing status
- Retrieve and process results
- Handle both successful and failed requests

Usage:
    python run_anthropic.py
"""

import json
import time
import os
from typing import List, Dict, Any
import anthropic
from pydantic import BaseModel


class PoemRequest(BaseModel):
    theme: str
    style: str = "haiku"
    mood: str = "contemplative"


class BatchResult(BaseModel):
    custom_id: str
    poem: str
    error: str = None


def create_batch_requests(poems: List[PoemRequest]) -> List[Dict[str, Any]]:
    """Create batch requests from poem themes"""
    requests = []
    for i, poem in enumerate(poems):
        request = {
            "custom_id": f"poem-{i}",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 150,
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Write a {poem.style} poem about '{poem.theme}' "
                                 f"with a {poem.mood} mood. Be creative and expressive."
                    }
                ]
            }
        }
        requests.append(request)
    return requests


def monitor_batch_job(client: anthropic.Anthropic, batch_id: str) -> str:
    """Monitor batch job until completion"""
    print(f"Monitoring batch job: {batch_id}")
    
    while True:
        batch_status = client.messages.batches.retrieve(batch_id)
        print(f"Status: {batch_status.processing_status}")
        
        if batch_status.processing_status == "ended":
            print("✅ Batch completed successfully!")
            return batch_status.processing_status
        elif batch_status.processing_status in ["failed", "expired"]:
            print(f"❌ Batch failed with status: {batch_status.processing_status}")
            return None
        
        time.sleep(60)  # Check every minute


def process_batch_results(client: anthropic.Anthropic, batch_id: str) -> List[BatchResult]:
    """Process batch results from completed batch"""
    results_iter = client.messages.batches.results(batch_id)
    
    batch_results = []
    for result in results_iter:
        custom_id = result.custom_id
        
        if result.result.type == "succeeded":
            # Extract text from Claude's response
            poem_text = result.result.message.content[0].text
            batch_results.append(BatchResult(
                custom_id=custom_id,
                poem=poem_text
            ))
        else:
            # Handle errors
            error_msg = str(result.result.error) if hasattr(result.result, 'error') else "Unknown error"
            batch_results.append(BatchResult(
                custom_id=custom_id,
                poem="",
                error=error_msg
            ))
    
    return batch_results


def main():
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create sample poem requests
    poem_requests = [
        PoemRequest(theme="cherry blossoms", style="haiku", mood="peaceful"),
        PoemRequest(theme="city lights", style="free verse", mood="energetic"),
        PoemRequest(theme="ocean waves", style="sonnet", mood="contemplative"),
        PoemRequest(theme="mountain peaks", style="haiku", mood="majestic"),
        PoemRequest(theme="autumn leaves", style="limerick", mood="playful"),
    ]
    
    # Add more themes to demonstrate batch processing
    themes = [
        "starlit sky", "morning dew", "thunder storm", "quiet forest",
        "bustling market", "lonely lighthouse", "dancing flames", "winter snow",
        "spring rain", "desert sunset", "flowing river", "ancient tree",
        "butterfly garden", "moonlit path", "golden wheat", "misty mountain",
        "coral reef", "prairie wind", "crystal cave", "burning candle",
        "hidden valley", "soaring eagle", "gentle breeze", "midnight hour",
        "rainbow bridge", "silver lake", "crimson rose", "azure sky"
    ]
    
    styles = ["haiku", "free verse", "limerick", "sonnet", "acrostic"]
    moods = ["peaceful", "energetic", "contemplative", "playful", "mysterious"]
    
    for i, theme in enumerate(themes):
        style = styles[i % len(styles)]
        mood = moods[i % len(moods)]
        poem_requests.append(PoemRequest(theme=theme, style=style, mood=mood))
    
    print(f"Creating batch job with {len(poem_requests)} poem requests...")
    
    # 1. Create batch requests
    batch_requests = create_batch_requests(poem_requests)
    
    # 2. Create batch job
    print("Creating batch job...")
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Batch created with ID: {batch.id}")
    print(f"Request count: {batch.request_counts}")
    
    # 3. Monitor batch job
    status = monitor_batch_job(client, batch.id)
    
    if status == "ended":
        # 4. Process results
        print("Processing results...")
        batch_results = process_batch_results(client, batch.id)
        
        # Display results
        print(f"\n📝 Generated {len(batch_results)} poems:")
        print("=" * 60)
        
        successful_results = [r for r in batch_results if not r.error]
        failed_results = [r for r in batch_results if r.error]
        
        print(f"✅ Successful: {len(successful_results)}")
        print(f"❌ Failed: {len(failed_results)}")
        print()
        
        # Show first few successful results
        for i, result in enumerate(successful_results[:5]):
            print(f"🎭 {result.custom_id}:")
            print(f"   {result.poem}")
            print("-" * 40)
        
        if len(successful_results) > 5:
            print(f"... and {len(successful_results) - 5} more poems")
            print()
        
        # Show any failures
        if failed_results:
            print("❌ Failed requests:")
            for result in failed_results:
                print(f"   {result.custom_id}: {result.error}")
            print()
        
        # Save results to file
        results_filename = "anthropic_batch_results.json"
        with open(results_filename, 'w') as f:
            json.dump([result.dict() for result in batch_results], f, indent=2)
        print(f"Results saved to {results_filename}")
        
        # Display batch statistics
        final_batch = client.messages.batches.retrieve(batch.id)
        print(f"\n📊 Batch Statistics:")
        print(f"   Total requests: {final_batch.request_counts.processing}")
        print(f"   Processing time: {final_batch.created_at} to {final_batch.ended_at}")
        print(f"   Cost savings: 50% vs regular API")
    
    print("🎉 Batch processing complete!")


if __name__ == "__main__":
    main()