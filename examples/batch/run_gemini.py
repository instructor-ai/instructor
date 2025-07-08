"""Google Gemini Batch API Example

This example demonstrates how to use Google Gemini's Batch API to process large volumes
of requests asynchronously at 50% cost savings with 24-hour processing guarantee.

Features:
- Create batch requests for Cloud Storage input
- Monitor batch job status
- Process results from Cloud Storage output
- Handle large-scale batch processing (25,000+ requests recommended)

Note: This example requires Google Cloud Storage setup and appropriate credentials.

Usage:
    python run_gemini.py
"""

import json
import time
import os
from typing import List, Dict, Any
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from pydantic import BaseModel


class ArticleRequest(BaseModel):
    topic: str
    word_count: int = 200
    tone: str = "informative"
    audience: str = "general"


class BatchResult(BaseModel):
    request_id: str
    article: str
    error: str = None


def create_batch_requests(articles: List[ArticleRequest]) -> List[Dict[str, Any]]:
    """Create batch requests for Gemini API"""
    requests = []
    for i, article in enumerate(articles):
        request = {
            "contents": [{
                "parts": [{
                    "text": f"Write a {article.word_count}-word {article.tone} article about '{article.topic}' "
                           f"for {article.audience} audience. Make it engaging and well-structured."
                }],
                "role": "user"
            }],
            "generationConfig": {
                "maxOutputTokens": article.word_count * 2,
                "temperature": 0.7,
                "topP": 0.9
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }
        requests.append(request)
    return requests


def save_batch_to_jsonl(requests: List[Dict[str, Any]], filename: str) -> None:
    """Save batch requests to JSONL format for Cloud Storage"""
    with open(filename, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    print(f"Saved {len(requests)} requests to {filename}")


def monitor_batch_job(client: genai.Client, job_name: str) -> str:
    """Monitor batch job until completion"""
    print(f"Monitoring batch job: {job_name}")
    
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED
    }
    
    while True:
        job = client.batches.get(job_name)
        print(f"Status: {job.state}")
        
        if job.state == JobState.JOB_STATE_SUCCEEDED:
            print("✅ Batch completed successfully!")
            return job.state
        elif job.state in [JobState.JOB_STATE_FAILED, JobState.JOB_STATE_CANCELLED]:
            print(f"❌ Batch failed with status: {job.state}")
            if hasattr(job, 'error') and job.error:
                print(f"Error: {job.error}")
            return None
        
        time.sleep(300)  # Check every 5 minutes (Gemini batches can take longer)


def process_batch_results(results_location: str) -> List[BatchResult]:
    """Process batch results from Cloud Storage location
    
    Note: In a real implementation, you would download the results from
    Cloud Storage and process them. This is a simplified example.
    """
    print(f"Results would be available at: {results_location}")
    
    # This is a mock implementation - in reality you would:
    # 1. Download the results file from Cloud Storage
    # 2. Parse the JSONL results
    # 3. Process each response
    
    batch_results = []
    # Mock results for demonstration
    for i in range(5):
        batch_results.append(BatchResult(
            request_id=f"article-{i}",
            article=f"[Mock article {i} would be here - download from {results_location}]"
        ))
    
    return batch_results


def main():
    # Initialize Gemini client
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    
    # Create sample article requests
    article_requests = [
        ArticleRequest(topic="Artificial Intelligence in Healthcare", tone="informative", audience="professionals"),
        ArticleRequest(topic="Climate Change Solutions", tone="optimistic", audience="general"),
        ArticleRequest(topic="Space Exploration Technologies", tone="exciting", audience="students"),
        ArticleRequest(topic="Renewable Energy Trends", tone="analytical", audience="investors"),
        ArticleRequest(topic="Digital Privacy Rights", tone="educational", audience="consumers"),
    ]
    
    # Generate more requests to reach the recommended 25,000+ minimum
    topics = [
        "Machine Learning Applications", "Quantum Computing Breakthroughs", "Biotechnology Advances",
        "Sustainable Agriculture", "Smart City Development", "Blockchain Technology", "Robotics Innovation",
        "Cybersecurity Trends", "Virtual Reality Gaming", "Electric Vehicle Market", "Gene Therapy Progress",
        "Solar Energy Efficiency", "Ocean Conservation", "Mental Health Technology", "Food Security Issues"
    ]
    
    tones = ["informative", "analytical", "optimistic", "educational", "exciting"]
    audiences = ["professionals", "general", "students", "investors", "consumers"]
    
    # Create enough requests to demonstrate batch processing
    for i in range(100):  # In practice, you'd want 25,000+ requests
        topic = topics[i % len(topics)]
        tone = tones[i % len(tones)]
        audience = audiences[i % len(audiences)]
        article_requests.append(ArticleRequest(
            topic=f"{topic} - Part {i//len(topics) + 1}",
            tone=tone,
            audience=audience,
            word_count=150 + (i % 100)  # Vary word count
        ))
    
    print(f"Creating batch job with {len(article_requests)} article requests...")
    
    # 1. Create batch requests
    batch_requests = create_batch_requests(article_requests)
    
    # 2. Save to JSONL file (would normally upload to Cloud Storage)
    batch_filename = "gemini_batch_requests.jsonl"
    save_batch_to_jsonl(batch_requests, batch_filename)
    
    # Important: In a real implementation, you would upload this file to Cloud Storage
    # For example: gs://your-bucket/batch_requests.jsonl
    input_uri = f"gs://your-batch-bucket/{batch_filename}"
    output_uri = "gs://your-batch-bucket/batch_results/"
    
    print(f"⚠️  Note: You need to upload {batch_filename} to Cloud Storage: {input_uri}")
    print(f"⚠️  Make sure your Cloud Storage bucket exists and you have proper permissions.")
    
    # 3. Create batch job (this would work with actual Cloud Storage URIs)
    try:
        print("Creating batch job...")
        job = client.batches.create(
            model="gemini-2.0-flash-001",
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_uri)
        )
        
        print(f"Batch job created:")
        print(f"  Job name: {job.name}")
        print(f"  Job state: {job.state}")
        print(f"  Input: {input_uri}")
        print(f"  Output: {output_uri}")
        
        # 4. Monitor batch job
        final_state = monitor_batch_job(client, job.name)
        
        if final_state == JobState.JOB_STATE_SUCCEEDED:
            # 5. Process results
            print("Processing results...")
            batch_results = process_batch_results(output_uri)
            
            # Display results
            print(f"\n📰 Generated {len(batch_results)} articles:")
            print("=" * 60)
            
            for result in batch_results:
                if result.error:
                    print(f"❌ {result.request_id}: ERROR - {result.error}")
                else:
                    print(f"✅ {result.request_id}:")
                    print(f"   {result.article[:100]}...")
                    print()
            
            # Save results summary
            results_filename = "gemini_batch_results.json"
            with open(results_filename, 'w') as f:
                json.dump([result.dict() for result in batch_results], f, indent=2)
            print(f"Results summary saved to {results_filename}")
            
            print(f"\n📊 Batch Statistics:")
            print(f"   Total requests: {len(batch_requests)}")
            print(f"   Cost savings: 50% vs regular API")
            print(f"   Results location: {output_uri}")
        
        # Cleanup local file
        os.remove(batch_filename)
        print(f"Cleaned up {batch_filename}")
        
    except Exception as e:
        print(f"❌ Error creating batch job: {e}")
        print("\nThis example requires:")
        print("1. Google Cloud Storage bucket setup")
        print("2. Proper authentication (Application Default Credentials)")
        print("3. Cloud Storage permissions")
        print("4. Gemini API access")
        print("\nFor testing purposes, the batch file has been created locally.")
        
        # Show what the batch file contains
        print(f"\n📄 Sample batch request (first 3 lines of {batch_filename}):")
        with open(batch_filename, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    request = json.loads(line)
                    print(f"  Request {i+1}: {request['contents'][0]['parts'][0]['text'][:80]}...")
                else:
                    break
        
        os.remove(batch_filename)
    
    print("🎉 Batch processing example complete!")


if __name__ == "__main__":
    main()