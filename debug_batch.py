#!/usr/bin/env python3
"""Debug script to check Anthropic batch results"""

import anthropic
import json

def debug_anthropic_batch():
    """Debug the current Anthropic batch results"""
    batch_id = "msgbatch_01CmrcYacn34ugzmLNPTu6wX"
    
    client = anthropic.Anthropic()
    
    # Use stable or beta API path
    if hasattr(client, "messages") and hasattr(client.messages, "batches"):
        batches_client = client.messages.batches
        print("Using stable API path")
    else:
        batches_client = client.beta.messages.batches
        print("Using beta API path")
    
    # Get batch status
    batch = batches_client.retrieve(batch_id)
    print(f"Batch status: {batch.processing_status}")
    
    if batch.processing_status == "ended":
        print("Getting results...")
        results = batches_client.results(batch_id)
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            result_json = result.model_dump_json()
            print(result_json)
            
            # Also parse and pretty print
            result_dict = json.loads(result_json)
            print(f"Custom ID: {result_dict.get('custom_id')}")
            print(f"Result type: {result_dict.get('result', {}).get('type')}")
            
            if result_dict.get('result', {}).get('type') == 'message':
                message = result_dict['result']['message']
                print(f"Message content: {message.get('content')}")
            elif result_dict.get('result', {}).get('type') == 'error':
                error = result_dict['result']['error']
                print(f"Error: {error}")

if __name__ == "__main__":
    debug_anthropic_batch()
