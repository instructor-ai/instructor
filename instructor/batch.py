"""
Unified Batch Processing API for Multiple Providers

This module provides a unified interface for batch processing across OpenAI, Anthropic, 
and Google GenAI providers. The API uses a Maybe/Result-like pattern with custom_id 
tracking for type-safe handling of batch results.

Supported Providers:
- OpenAI: 50% cost savings on batch requests
- Anthropic: 50% cost savings on batch requests (Message Batches API)
- Google GenAI: 50% cost savings on batch requests (Cloud Vertex AI)

Features:
- Type-safe Maybe/Result pattern for handling successes and errors
- Custom ID tracking for correlating results to original requests
- Unified interface across all providers
- Helper functions for filtering and extracting results

Example usage:
    from instructor.batch import BatchProcessor, filter_successful, extract_results
    from pydantic import BaseModel
    
    class User(BaseModel):
        name: str
        age: int
    
    processor = BatchProcessor("openai/gpt-4o-mini", User)
    batch_id = processor.submit_batch("requests.jsonl")
    
    # Results are BatchSuccess[T] | BatchError union types
    all_results = processor.retrieve_results(batch_id)
    successful_results = filter_successful(all_results)
    extracted_users = extract_results(all_results)

Documentation:
- OpenAI Batch API: https://platform.openai.com/docs/guides/batch
- Anthropic Message Batches: https://docs.anthropic.com/en/api/creating-message-batches
- Google Cloud Batch Prediction: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
"""
from __future__ import annotations
from typing import Any, Union, TypeVar, Optional, List, Dict, Type, Tuple, Generic
from collections.abc import Iterable
from pydantic import BaseModel, Field
from instructor.process_response import handle_response_model
import instructor
import uuid
import json
import os
from instructor.auto_client import from_provider

T = TypeVar("T", bound=BaseModel)


class BatchSuccess(BaseModel, Generic[T]):
    """Successful batch result with custom_id"""
    custom_id: str
    result: T
    success: bool = True
    
    class Config:
        arbitrary_types_allowed = True


class BatchError(BaseModel):
    """Error information for failed batch requests"""
    custom_id: str
    error_type: str
    error_message: str
    success: bool = False
    raw_data: Optional[Dict[str, Any]] = None


# Union type for batch results - like a Maybe/Result type
BatchResult = Union[BatchSuccess[T], BatchError]


def filter_successful(results: List[BatchResult]) -> List[BatchSuccess[T]]:
    """Filter to only successful results"""
    return [r for r in results if r.success]


def filter_errors(results: List[BatchResult]) -> List[BatchError]:
    """Filter to only error results"""
    return [r for r in results if not r.success]


def extract_results(results: List[BatchResult]) -> List[T]:
    """Extract just the result objects from successful results"""
    return [r.result for r in results if r.success]


def get_results_by_custom_id(results: List[BatchResult]) -> Dict[str, BatchResult]:
    """Create a dictionary mapping custom_id to results"""
    return {r.custom_id: r for r in results}


class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = Field(default=1000)
    temperature: Optional[float] = Field(default=1.0)
    tools: Optional[list[Tool]]
    tool_choice: Optional[dict[str, Any]]


class BatchModel(BaseModel):
    custom_id: str
    body: RequestBody
    url: str
    method: str


class BatchRequest(BaseModel, Generic[T]):
    """Unified batch request that works across all providers using JSON schema"""

    custom_id: str
    messages: List[Dict[str, Any]]
    response_model: Type[T]
    model: str
    max_tokens: Optional[int] = Field(default=1000)
    temperature: Optional[float] = Field(default=0.1)

    class Config:
        arbitrary_types_allowed = True

    def get_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema from response_model"""
        return self.response_model.model_json_schema()

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI batch format with JSON schema"""
        schema = self.get_json_schema()
        
        # OpenAI strict mode requires additionalProperties to be false
        def make_strict_schema(schema_dict):
            """Recursively add additionalProperties: false for OpenAI strict mode"""
            if isinstance(schema_dict, dict):
                if "type" in schema_dict:
                    if schema_dict["type"] == "object":
                        schema_dict["additionalProperties"] = False
                    elif schema_dict["type"] == "array" and "items" in schema_dict:
                        schema_dict["items"] = make_strict_schema(schema_dict["items"])
                
                # Recursively process properties
                if "properties" in schema_dict:
                    for prop_name, prop_schema in schema_dict["properties"].items():
                        schema_dict["properties"][prop_name] = make_strict_schema(prop_schema)
                
                # Process definitions/defs
                for key in ["definitions", "$defs"]:
                    if key in schema_dict:
                        for def_name, def_schema in schema_dict[key].items():
                            schema_dict[key][def_name] = make_strict_schema(def_schema)
            
            return schema_dict
        
        strict_schema = make_strict_schema(schema.copy())
        
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_model.__name__,
                        "strict": True,
                        "schema": strict_schema,
                    },
                },
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic batch format with JSON schema"""
        schema = self.get_json_schema()
        
        # Ensure schema has proper format for Anthropic
        if "type" not in schema:
            schema["type"] = "object"
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        
        # Extract system message and convert to system parameter
        system_message = None
        filtered_messages = []
        
        for message in self.messages:
            if message.get("role") == "system":
                system_message = message.get("content", "")
            else:
                filtered_messages.append(message)
        
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": filtered_messages,
            "tools": [
                {
                    "name": "extract_data",
                    "description": f"Extract data matching the {self.response_model.__name__} schema",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "extract_data"},
        }
        
        # Add system parameter if system message exists
        if system_message:
            params["system"] = system_message
        
        return {
            "custom_id": self.custom_id,
            "params": params,
        }

    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Gemini batch format with JSON schema"""
        schema = self.get_json_schema()
        return {
            "contents": [
                {"parts": [{"text": self._format_gemini_prompt()}], "role": "user"}
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
                "responseSchema": schema,
                "responseMimeType": "application/json",
            },
        }

    def to_genai_batch_format(self) -> Dict[str, Any]:
        """Convert to Google GenAI batch format for batch processing
        
        Based on Google Cloud documentation:
        https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#batch_prediction_inputs_and_outputs
        """
        schema = self.get_json_schema()
        return {
            "request": {
                "contents": [
                    {"parts": [{"text": self._format_gemini_prompt()}], "role": "user"}
                ],
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "responseSchema": schema,
                    "responseMimeType": "application/json",
                },
            }
        }

    def _format_gemini_prompt(self) -> str:
        """Format messages for Gemini as a single prompt"""
        prompt_parts = []
        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"Instructions: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        schema_desc = (
            f"Return a JSON object matching this schema: {self.get_json_schema()}"
        )
        prompt_parts.append(schema_desc)
        return "\n\n".join(prompt_parts)

    def save_to_file(self, file_path: str, provider: str) -> None:
        """Save batch request to file in provider-specific format"""
        if provider == "openai":
            data = self.to_openai_format()
        elif provider == "anthropic":
            data = self.to_anthropic_format()
        elif provider == "google":
            data = self.to_genai_batch_format()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        with open(file_path, "a") as f:
            f.write(json.dumps(data) + "\n")


class BatchProcessor(Generic[T]):
    """Unified batch processor that works across all providers"""

    def __init__(self, model: str, response_model: Type[T]):
        self.model = model
        self.response_model = response_model

        # Parse provider from model string
        try:
            self.provider_name, self.model_name = model.split("/", 1)
        except ValueError:
            raise ValueError(
                'Model string must be in format "provider/model-name" '
                '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
            )

        # Get the instructor client
        self.client = from_provider(model)

    def create_batch_from_messages(
        self,
        file_path: str,
        *,
        messages_list: List[List[Dict[str, Any]]],
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0.1,
    ) -> str:
        """Create batch file from list of message conversations"""
        # Remove existing file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        batch_requests = []
        for i, messages in enumerate(messages_list):
            batch_request = BatchRequest[self.response_model](
                custom_id=f"request-{i}",
                messages=messages,
                response_model=self.response_model,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            batch_request.save_to_file(file_path, self.provider_name)
            batch_requests.append(batch_request)

        print(f"Created batch file {file_path} with {len(batch_requests)} requests")
        return file_path

    def submit_batch(
        self,
        file_path: str,
        **kwargs
    ) -> str:
        """Submit batch job to the provider and return job ID"""
        if self.provider_name == "openai":
            return self._submit_openai_batch(file_path, **kwargs)
        elif self.provider_name == "anthropic":
            return self._submit_anthropic_batch(file_path, **kwargs)
        elif self.provider_name == "google":
            return self._submit_google_batch(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported provider for batch submission: {self.provider_name}")

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch job status from the provider"""
        if self.provider_name == "openai":
            return self._get_openai_status(batch_id)
        elif self.provider_name == "anthropic":
            return self._get_anthropic_status(batch_id)
        elif self.provider_name == "google":
            return self._get_google_status(batch_id)
        else:
            raise ValueError(f"Unsupported provider for batch status: {self.provider_name}")

    def retrieve_results(self, batch_id: str) -> List[BatchResult]:
        """Retrieve and parse batch results from the provider"""
        if self.provider_name == "openai":
            results_content = self._retrieve_openai_results(batch_id)
        elif self.provider_name == "anthropic":
            results_content = self._retrieve_anthropic_results(batch_id)
        elif self.provider_name == "google":
            results_content = self._retrieve_google_results(batch_id)
        else:
            raise ValueError(f"Unsupported provider for result retrieval: {self.provider_name}")
        
        return self.parse_results(results_content)

    def _submit_openai_batch(self, file_path: str, **kwargs) -> str:
        """Submit OpenAI batch job"""
        try:
            from openai import OpenAI
            client = OpenAI()
            
            with open(file_path, "rb") as f:
                batch_file = client.files.create(file=f, purpose="batch")

            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=kwargs.get("metadata", {"description": "Instructor batch job"}),
            )
            return batch_job.id
        except Exception as e:
            raise Exception(f"Failed to submit OpenAI batch: {e}")

    def _submit_anthropic_batch(self, file_path: str, **kwargs) -> str:
        """Submit Anthropic batch job"""
        try:
            import anthropic
            client = anthropic.Anthropic()
            
            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            
            with open(file_path, "r") as f:
                requests = [json.loads(line) for line in f if line.strip()]

            batch = batches_client.create(requests=requests)
            return batch.id
        except Exception as e:
            raise Exception(f"Failed to submit Anthropic batch: {e}")

    def _submit_google_batch(self, file_path: str, **kwargs) -> str:
        """Submit Google GenAI batch job"""
        try:
            from google import genai
            from google.genai.types import CreateBatchJobConfig, HttpOptions
            from google.cloud import storage
            import uuid
            
            # Required environment variables
            bucket_name = kwargs.get("bucket_name") or os.getenv("GCS_BUCKET")
            if not bucket_name:
                raise Exception("GCS_BUCKET environment variable or bucket_name parameter is required")
            
            # Remove gs:// prefix if present
            if bucket_name.startswith("gs://"):
                bucket_name = bucket_name[5:]
            
            # Upload to GCS
            blob_path = f"batch-inputs/{uuid.uuid4()}.jsonl"
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(file_path)
            gcs_input_uri = f"gs://{bucket_name}/{blob_path}"
            
            # Create GenAI client and submit batch
            client = genai.Client(http_options=HttpOptions(api_version="v1"))
            output_uri = f"gs://{bucket_name}/batch-outputs/"
            
            job = client.batches.create(
                model=self.model_name,
                src=gcs_input_uri,
                config=CreateBatchJobConfig(dest=output_uri),
            )
            return job.name
        except Exception as e:
            raise Exception(f"Failed to submit Google batch: {e}")

    def _get_openai_status(self, batch_id: str) -> Dict[str, Any]:
        """Get OpenAI batch status"""
        try:
            from openai import OpenAI
            client = OpenAI()
            batch = client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "request_counts": {
                    "total": getattr(batch.request_counts, "total", 0),
                    "completed": getattr(batch.request_counts, "completed", 0),
                    "failed": getattr(batch.request_counts, "failed", 0)
                }
            }
        except Exception as e:
            raise Exception(f"Failed to get OpenAI batch status: {e}")

    def _get_anthropic_status(self, batch_id: str) -> Dict[str, Any]:
        """Get Anthropic batch status"""
        try:
            import anthropic
            client = anthropic.Anthropic()
            
            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            
            batch = batches_client.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "created_at": batch.created_at,
                "request_counts": getattr(batch, "request_counts", {})
            }
        except Exception as e:
            raise Exception(f"Failed to get Anthropic batch status: {e}")

    def _get_google_status(self, batch_id: str) -> Dict[str, Any]:
        """Get Google GenAI batch status"""
        try:
            from google import genai
            from google.genai.types import HttpOptions
            client = genai.Client(http_options=HttpOptions(api_version="v1"))
            job = client.batches.get(name=batch_id)
            return {
                "id": job.name,
                "status": job.state.name,
                "created_at": getattr(job, "create_time", None),
                "output_uri": getattr(job, "output_uri", None)
            }
        except Exception as e:
            raise Exception(f"Failed to get Google batch status: {e}")

    def _retrieve_openai_results(self, batch_id: str) -> str:
        """Retrieve OpenAI batch results"""
        try:
            from openai import OpenAI
            client = OpenAI()
            batch = client.batches.retrieve(batch_id)
            
            if batch.status != "completed":
                raise Exception(f"Batch not completed, status: {batch.status}")
            
            if not batch.output_file_id:
                raise Exception("No output file available")
            
            file_response = client.files.content(batch.output_file_id)
            return file_response.text
        except Exception as e:
            raise Exception(f"Failed to retrieve OpenAI results: {e}")

    def _retrieve_anthropic_results(self, batch_id: str) -> str:
        """Retrieve Anthropic batch results"""
        try:
            import anthropic
            client = anthropic.Anthropic()
            
            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            
            batch = batches_client.retrieve(batch_id)
            
            # Check for various terminal states
            if batch.processing_status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Batch job failed with status: {batch.processing_status}")
            
            if batch.processing_status != "ended":
                raise Exception(f"Batch not completed, status: {batch.processing_status}")
            
            results = batches_client.results(batch_id)
            results_lines = []
            for result in results:
                results_lines.append(result.model_dump_json())
            
            return '\n'.join(results_lines)
        except Exception as e:
            raise Exception(f"Failed to retrieve Anthropic results: {e}")

    def _retrieve_google_results(self, batch_id: str) -> str:
        """Retrieve Google GenAI batch results"""
        try:
            from google import genai
            from google.genai.types import JobState, HttpOptions
            from google.cloud import storage
            
            # Get job status
            client = genai.Client(http_options=HttpOptions(api_version="v1"))
            job = client.batches.get(name=batch_id)
            
            if job.state != JobState.JOB_STATE_SUCCEEDED:
                raise Exception(f"Batch not completed successfully, status: {job.state.name}")
            
            if not job.output_uri:
                raise Exception("No output URI available")
            
            # Download results from GCS
            if not job.output_uri.startswith("gs://"):
                raise Exception("Invalid GCS URI format")
            
            gcs_path = job.output_uri[5:]  # Remove gs://
            bucket_name, prefix = gcs_path.split("/", 1)
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            results_lines = []
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith('.jsonl'):
                    content = blob.download_as_text()
                    for line in content.splitlines():
                        if line.strip():
                            results_lines.append(line)
            
            return '\n'.join(results_lines)
        except Exception as e:
            raise Exception(f"Failed to retrieve Google results: {e}")

    def parse_results(
        self, results_content: str
    ) -> List[BatchResult]:
        """Parse batch results from content string into Maybe-like results with custom_id tracking"""
        results: List[BatchResult] = []

        lines = results_content.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                custom_id = data.get("custom_id", "unknown")
                extracted_data = self._extract_from_response(data)

                if extracted_data:
                    try:
                        # Parse into response model
                        result = self.response_model(**extracted_data)
                        batch_result = BatchSuccess[T](custom_id=custom_id, result=result)
                        results.append(batch_result)
                    except Exception as e:
                        error_result = BatchError(
                            custom_id=custom_id,
                            error_type="parsing_error",
                            error_message=f"Failed to parse into {self.response_model.__name__}: {e}",
                            raw_data=extracted_data,
                        )
                        results.append(error_result)
                else:
                    # Check if this is a provider error response
                    error_message = "Unknown error"
                    error_type = "extraction_error"
                    
                    if self.provider_name == "anthropic" and "result" in data:
                        result = data["result"]
                        if result.get("type") == "error":
                            error_info = result.get("error", {})
                            if isinstance(error_info, dict) and "error" in error_info:
                                error_details = error_info["error"]
                                error_message = error_details.get("message", "Unknown Anthropic error")
                                error_type = error_details.get("type", "anthropic_error")
                            else:
                                error_message = str(error_info)
                                error_type = "anthropic_error"
                    
                    error_result = BatchError(
                        custom_id=custom_id,
                        error_type=error_type,
                        error_message=error_message,
                        raw_data=data,
                    )
                    results.append(error_result)

            except Exception as e:
                error_result = BatchError(
                    custom_id="unknown",
                    error_type="json_parse_error",
                    error_message=f"Failed to parse JSON: {e}",
                    raw_data={"raw_line": line},
                )
                results.append(error_result)

        return results

    def _extract_from_response(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structured data from provider-specific response format"""
        try:
            if self.provider_name == "openai":
                # OpenAI JSON schema response
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                return json.loads(content)

            elif self.provider_name == "anthropic":
                # Anthropic batch response format
                if "result" not in data:
                    return None
                    
                result = data["result"]
                
                # Check if result is an error
                if result.get("type") == "error":
                    # Return None to indicate error, let caller handle
                    return None
                
                # Handle successful message result
                if result.get("type") == "succeeded" and "message" in result:
                    content = result["message"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Try tool_use first
                        for item in content:
                            if item.get("type") == "tool_use":
                                return item.get("input", {})
                        
                        # Fallback to text content and parse JSON
                        for item in content:
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                try:
                                    return json.loads(text)
                                except json.JSONDecodeError:
                                    continue
                
                return None

            elif self.provider_name == "google":
                # Handle Google GenAI batch response format
                # Documentation: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#batch_prediction_inputs_and_outputs
                if "response" in data and "candidates" in data["response"]:
                    # Google GenAI batch response format
                    candidates = data["response"]["candidates"]
                    if candidates and len(candidates) > 0:
                        content = candidates[0]["content"]["parts"][0]["text"]
                        return json.loads(content)
                elif "candidates" in data and len(data["candidates"]) > 0:
                    # Direct Gemini API response
                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(content)
                return None

        except Exception:
            return None

        return None


class BatchJob:
    """Legacy BatchJob class for backward compatibility"""

    @classmethod
    def parse_from_file(
        cls, file_path: str, response_model: type[T]
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        with open(file_path) as file:
            content = file.read()
        return cls.parse_from_string(content, response_model)

    @classmethod
    def parse_from_string(
        cls, content: str, response_model: type[T]
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        """Enhanced parser that works with all providers using JSON schema"""
        res: list[T] = []
        error_objs: list[dict[Any, Any]] = []

        lines = content.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                extracted_data = cls._extract_structured_data(data)

                if extracted_data:
                    try:
                        result = response_model(**extracted_data)
                        res.append(result)
                    except Exception:
                        error_objs.append(data)
                else:
                    error_objs.append(data)

            except Exception:
                error_objs.append({"error": "Failed to parse JSON", "raw_line": line})

        return res, error_objs

    @classmethod
    def _extract_structured_data(cls, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structured data from various provider response formats"""
        try:
            # Try OpenAI JSON schema format first
            if "response" in data and "body" in data["response"]:
                choices = data["response"]["body"].get("choices", [])
                if choices:
                    message = choices[0].get("message", {})

                    # JSON schema response
                    if "content" in message:
                        content = message["content"]
                        if isinstance(content, str):
                            return json.loads(content)

                    # Tool calls (legacy)
                    if "tool_calls" in message:
                        tool_call = message["tool_calls"][0]
                        return json.loads(tool_call["function"]["arguments"])

            # Try Anthropic format
            if "result" in data and "message" in data["result"]:
                content = data["result"]["message"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Tool use response
                    for item in content:
                        if item.get("type") == "tool_use":
                            return item.get("input", {})
                    # Text response with JSON
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            return json.loads(text)

            # Try Gemini format
            if "candidates" in data:
                candidates = data["candidates"]
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        text = parts[0].get("text", "")
                        return json.loads(text)

        except Exception:
            pass

        return None

    @classmethod
    def create_from_messages(
        cls,
        messages_batch: Union[
            list[list[dict[str, Any]]], Iterable[list[dict[str, Any]]]
        ],
        model: str,
        response_model: type[BaseModel],
        file_path: str,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 1.0,
    ):
        """Create batch file from messages using provider detection"""
        # Detect provider from model name
        use_anthropic = "claude" in model.lower()
        use_gemini = "gemini" in model.lower()

        # Use the new BatchProcessor for unified handling
        if use_anthropic:
            full_model = f"anthropic/{model}"
        elif use_gemini:
            full_model = f"google/{model}"
        else:
            full_model = f"openai/{model}"

        try:
            processor = BatchProcessor(full_model, response_model)
            processor.create_batch_from_messages(
                list(messages_batch), file_path, max_tokens, temperature
            )
        except Exception:
            # Fallback to legacy implementation for backward compatibility
            cls._create_legacy_format(
                messages_batch,
                model,
                response_model,
                file_path,
                max_tokens,
                temperature,
            )

    @classmethod
    def _create_legacy_format(
        cls,
        messages_batch: Union[
            list[list[dict[str, Any]]], Iterable[list[dict[str, Any]]]
        ],
        model: str,
        response_model: type[BaseModel],
        file_path: str,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 1.0,
    ):
        """Legacy implementation for backward compatibility"""
        use_anthropic = "claude" in model.lower()

        if use_anthropic:
            _, kwargs = handle_response_model(
                response_model=response_model, mode=instructor.Mode.ANTHROPIC_JSON
            )
            with open(file_path, "w") as file:
                for messages in messages_batch:
                    # Format specifically for Anthropic batch API
                    request = {
                        "custom_id": str(uuid.uuid4()),
                        "params": {
                            "model": model,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "messages": messages,
                            **kwargs,
                        },
                    }
                    file.write(json.dumps(request) + "\n")
        else:
            # Existing OpenAI format
            _, kwargs = handle_response_model(
                response_model=response_model, mode=instructor.Mode.TOOLS
            )
            with open(file_path, "w") as file:
                for messages in messages_batch:
                    batch_model = BatchModel(
                        custom_id=str(uuid.uuid4()),
                        body=RequestBody(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        ),
                        method="POST",
                        url="/v1/chat/completions",
                    )
                    file.write(batch_model.model_dump_json() + "\n")


# Keep legacy models for backward compatibility
class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = Field(default=1000)
    temperature: Optional[float] = Field(default=1.0)
    tools: Optional[list[Tool]]
    tool_choice: Optional[dict[str, Any]]


class BatchModel(BaseModel):
    custom_id: str
    body: RequestBody
    url: str
    method: str
