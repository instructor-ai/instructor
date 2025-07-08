"""OpenAI Batch API CLI with Structured Data Extraction

A command-line tool for managing OpenAI batch jobs with structured data extraction.
Extracts User objects (name, age) from text with 50% cost savings.

Features:
- Create batch jobs for user extraction
- Check batch job status
- Process completed batch results
- List existing batch jobs

Usage:
    python run_openai.py create     # Create new batch job
    python run_openai.py status BATCH_ID  # Check status
    python run_openai.py process BATCH_ID # Process results
    python run_openai.py list      # List batch jobs
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel
import typer

app = typer.Typer(
    help="OpenAI Batch API CLI for User Extraction",
    epilog="Example usage: python run_openai.py create --count 5"
)


class User(BaseModel):
    name: str
    age: int


class UserRequest(BaseModel):
    prompt: str
    description: str = "Extract user information"


class BatchResult(BaseModel):
    custom_id: str
    user: Optional[User] = None
    error: Optional[str] = None


def create_batch_requests(user_requests: List[UserRequest]) -> List[Dict[str, Any]]:
    """Create batch requests for user extraction with JSON schema"""
    
    # Define the JSON schema for User extraction
    user_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string", 
                "description": "The person's full name (first and last name if available)"
            },
            "age": {
                "type": "integer", 
                "description": "The person's age in years as a number"
            }
        },
        "required": ["name", "age"],
        "additionalProperties": False
    }
    
    # Create function call tool for structured extraction
    tool = {
        "type": "function",
        "function": {
            "name": "extract_user_info",
            "description": "Extract the person's name and age from the given text",
            "parameters": user_schema
        }
    }
    
    batch_requests = []
    for i, request in enumerate(user_requests):
        batch_request = {
            "custom_id": f"user-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data extraction specialist. Your job is to extract the person's name and age from the user's message. You MUST use the extract_user_info function to provide the structured response."
                    },
                    {
                        "role": "user",
                        "content": f"Please extract the name and age from this text: {request.prompt}"
                    }
                ],
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": "extract_user_info"}},
                "max_tokens": 200,
                "temperature": 0.0
            }
        }
        batch_requests.append(batch_request)
    
    return batch_requests


def save_batch_file(batch_requests: List[Dict[str, Any]], filename: str) -> None:
    """Save batch requests to JSONL file"""
    with open(filename, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
    print(f"Saved {len(batch_requests)} requests to {filename}")


def get_client() -> OpenAI:
    """Get OpenAI client with API key validation"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.echo("❌ Error: OPENAI_API_KEY environment variable is not set", err=True)
        typer.echo("Please set your OpenAI API key:", err=True)
        typer.echo("export OPENAI_API_KEY='your-api-key-here'", err=True)
        raise typer.Exit(1)
    return OpenAI(api_key=api_key)


def list_batch_jobs_internal(client: OpenAI, limit: int = 10) -> None:
    """List existing batch jobs"""
    try:
        typer.echo(f"📋 Listing {limit} most recent batch jobs:")
        typer.echo("-" * 80)
        
        batches = client.batches.list(limit=limit)
        
        if not batches.data:
            typer.echo("No batch jobs found.")
            return
            
        for batch in batches.data:
            typer.echo(f"🔸 Batch ID: {batch.id}")
            typer.echo(f"   Status: {batch.status}")
            typer.echo(f"   Created: {batch.created_at}")
            if hasattr(batch, 'request_counts') and batch.request_counts:
                typer.echo(f"   Requests: {batch.request_counts.total} total, {batch.request_counts.completed} completed, {batch.request_counts.failed} failed")
            typer.echo()
            
    except Exception as e:
        typer.echo(f"Error listing batch jobs: {e}", err=True)


def check_batch_status(client: OpenAI, batch_id: str) -> Optional[str]:
    """Check batch job status once"""
    try:
        batch_status = client.batches.retrieve(batch_id)
        print(f"🔍 Batch {batch_id} status: {batch_status.status}")
        
        if batch_status.status == "completed":
            print("✅ Batch completed successfully!")
            return batch_status.output_file_id
        elif batch_status.status in ["failed", "expired", "cancelled"]:
            print(f"❌ Batch failed with status: {batch_status.status}")
            if hasattr(batch_status, 'errors') and batch_status.errors:
                print(f"Errors: {batch_status.errors}")
            return None
        else:
            print(f"⏳ Batch is still {batch_status.status}")
            return "pending"
            
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return None


def process_batch_results(client: OpenAI, output_file_id: str) -> List[BatchResult]:
    """Process batch results from output file with structured extraction"""
    try:
        result_file = client.files.content(output_file_id)
        content = result_file.content.decode('utf-8')
        
        batch_results = []
        
        # Parse each line of the JSONL output
        for line in content.strip().split('\n'):
            if not line.strip():
                continue
                
            result = json.loads(line)
            custom_id = result.get('custom_id', f"unknown-{len(batch_results)}")
            
            # Check if there's an error
            if result.get('error'):
                batch_results.append(BatchResult(
                    custom_id=custom_id,
                    error=str(result['error'])
                ))
                continue
            
            # Extract user data from function call response
            try:
                response = result['response']['body']
                message = response['choices'][0]['message']
                
                if 'tool_calls' in message and message['tool_calls']:
                    # Extract from function call
                    tool_call = message['tool_calls'][0]
                    if tool_call['function']['name'] == 'extract_user_info':
                        user_data = json.loads(tool_call['function']['arguments'])
                        user = User(**user_data)
                        batch_results.append(BatchResult(
                            custom_id=custom_id,
                            user=user
                        ))
                    else:
                        batch_results.append(BatchResult(
                            custom_id=custom_id,
                            error=f"Unexpected function call: {tool_call['function']['name']}"
                        ))
                else:
                    batch_results.append(BatchResult(
                        custom_id=custom_id,
                        error="No function call found in response"
                    ))
                    
            except Exception as e:
                batch_results.append(BatchResult(
                    custom_id=custom_id,
                    error=f"Error parsing response: {str(e)}"
                ))
        
        return batch_results
        
    except Exception as e:
        print(f"Error processing batch results: {e}")
        return []


@app.command()
def create(
    count: int = typer.Option(10, help="Number of user extraction requests to create"),
    preview: bool = typer.Option(False, help="Preview the batch requests without submitting")
) -> None:
    """Create a new batch job for user extraction"""
    client = get_client()
    create_new_batch_job(client, count, preview)


@app.command()
def status(batch_id: str) -> None:
    """Check the status of a batch job"""
    client = get_client()
    check_batch_status(client, batch_id)


@app.command()
def process(batch_id: str) -> None:
    """Process results from a completed batch job"""
    client = get_client()
    process_completed_batch_internal(client, batch_id)


@app.command()
def list(
    limit: int = typer.Option(10, help="Number of batch jobs to show")
) -> None:
    """List existing batch jobs"""
    client = get_client()
    list_batch_jobs_internal(client, limit)


def create_new_batch_job(client: OpenAI, count: int = 10, preview: bool = False) -> str:
    """Create a new batch job and return the batch ID"""
    typer.echo("🚀 Creating new batch job...")
    
    # Sample user data pool
    user_data_pool = [
        "My name is Alice and I'm 25 years old. I work as a software engineer.",
        "Hi, I'm Bob, 32 years old, and I love hiking and reading.",
        "This is Sarah speaking. I'm 28 and I'm a graphic designer.",
        "Hello! John here, I'm 45 years old and I teach mathematics.",
        "I'm Emma, 22 years old, currently studying psychology.",
        "My name is Michael and I'm 35 years old. I'm a chef at a local restaurant.",
        "I'm Lisa, 29 years old, working as a marketing manager.",
        "This is David, 41 years old, I'm a freelance photographer.",
        "Hello, I'm Jessica, 26 years old, and I'm a nurse.",
        "I'm Ryan, 33 years old, working in software development.",
        "My name is Amanda and I'm 30 years old. I'm a teacher.",
        "I'm Kevin, 27 years old, and I work in finance.",
        "This is Rachel, 31 years old, I'm a project manager.",
        "Hello! I'm Chris, 38 years old, and I'm an architect.",
        "I'm Sophie, 24 years old, working as a graphic artist.",
        "I'm Tom, 42 years old, and I'm a veterinarian.",
        "My name is Maria and I'm 27 years old. I'm a journalist.",
        "Hello, I'm Alex, 31 years old, working as a pilot.",
        "I'm Rebecca, 29 years old, and I'm a psychologist.",
        "This is James, 36 years old, I'm a civil engineer."
    ]
    
    # Select the requested number of user requests
    user_requests = []
    for i in range(min(count, len(user_data_pool))):
        user_requests.append(UserRequest(prompt=user_data_pool[i]))
    
    typer.echo(f"Creating batch job with {len(user_requests)} user extraction requests...")
    
    # 1. Create batch requests with structured extraction
    batch_requests = create_batch_requests(user_requests)
    
    # 2. Save batch requests to JSONL file
    batch_filename = "openai_batch_requests.jsonl"
    save_batch_file(batch_requests, batch_filename)
    
    # Preview mode - show the first request and exit
    if preview:
        typer.echo("\n📄 Preview of first batch request:")
        typer.echo("-" * 60)
        first_request = batch_requests[0]
        typer.echo(f"Custom ID: {first_request['custom_id']}")
        typer.echo(f"Messages:")
        for msg in first_request['body']['messages']:
            typer.echo(f"  {msg['role']}: {msg['content']}")
        typer.echo(f"Tools: {first_request['body']['tools'][0]['function']['name']}")
        typer.echo(f"Tool Choice: {first_request['body']['tool_choice']}")
        typer.echo("-" * 60)
        
        # Clean up and exit
        if os.path.exists(batch_filename):
            os.remove(batch_filename)
        return None
    
    try:
        # 3. Upload batch file
        print("Uploading batch file...")
        with open(batch_filename, "rb") as f:
            batch_file = client.files.create(
                file=f,
                purpose="batch"
            )
        print(f"File uploaded with ID: {batch_file.id}")
        
        # 4. Create batch job
        print("Creating batch job...")
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Story generation batch job"}
        )
        print(f"Batch job created with ID: {batch_job.id}")
        
        # 5. Check status and return batch ID
        status_result = check_batch_status(client, batch_job.id)
        print(f"\n✅ Batch job submitted successfully!")
        print(f"🆔 Batch ID: {batch_job.id}")
        print(f"💡 Use this ID to check status later")
        print(f"💰 Cost savings: 50% vs regular API")
        
        return batch_job.id
        
    except Exception as e:
        typer.echo(f"❌ Error creating batch job: {e}", err=True)
        return None
    finally:
        # Cleanup
        if os.path.exists(batch_filename):
            os.remove(batch_filename)
            typer.echo(f"🧹 Cleaned up {batch_filename}")


def process_completed_batch_internal(client: OpenAI, batch_id: str) -> None:
    """Process results from a completed batch job"""
    typer.echo(f"📊 Processing completed batch job: {batch_id}")
    
    # Check if batch is actually completed
    status_result = check_batch_status(client, batch_id)
    
    if status_result and status_result != "pending":
        # Process results
        typer.echo("Processing results...")
        batch_results = process_batch_results(client, status_result)
        
        # Display results
        typer.echo(f"\n👥 Extracted {len(batch_results)} user objects:")
        typer.echo("=" * 50)
        
        successful_results = [r for r in batch_results if not r.error and r.user]
        failed_results = [r for r in batch_results if r.error or not r.user]
        
        typer.echo(f"✅ Successful extractions: {len(successful_results)}")
        typer.echo(f"❌ Failed extractions: {len(failed_results)}")
        typer.echo()
        
        # Show successful results
        for result in successful_results:
            typer.echo(f"✅ {result.custom_id}:")
            typer.echo(f"   Name: {result.user.name}")
            typer.echo(f"   Age: {result.user.age}")
            typer.echo()
        
        # Show failed results
        for result in failed_results:
            typer.echo(f"❌ {result.custom_id}: ERROR - {result.error}")
        
        # Save results to file
        results_filename = f"batch_results_{batch_id}.json"
        with open(results_filename, 'w') as f:
            json.dump([result.model_dump() for result in batch_results], f, indent=2)
        typer.echo(f"💾 Results saved to {results_filename}")
    else:
        typer.echo("❌ Batch not completed yet or failed to get status", err=True)


if __name__ == "__main__":
    app()