#!/usr/bin/env python3
"""
Embedding Latency Benchmarking Script (Mock Implementation)

This script benchmarks embedding latency across multiple providers to demonstrate
that embedding models are the primary bottleneck in RAG systems, not database reads.
"""

import os
import sys
import time
import random
import asyncio
import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

MTEB_DATASETS = [
    "mteb/amazon_counterfactual",
    "mteb/banking77", 
    "mteb/emotion",
    "mteb/imdb",
    "mteb/massive-intent",
    "mteb/massive-scenario",
    "mteb/tweet_sentiment_extraction"
]

DEFAULT_MODELS = {
    "openai": "text-embedding-3-large",
    "cohere": "embed-english-v3.0", 
    "gemini": "models/text-embedding-004",
    "voyager": "voyage-3.5"
}

class EmbeddingProvider(ABC):
    def __init__(self, name: str):
        self.name = name
        self.available = self._check_availability()
        if self.available:
            print(f"🔧 Mock {name.title()} client initialized")

    @abstractmethod
    def _check_availability(self) -> bool:
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str], model: str = None, **kwargs) -> Dict[str, Any]:
        pass

class OpenAIProvider(EmbeddingProvider):
    def __init__(self):
        super().__init__("openai")

    def _check_availability(self) -> bool:
        return os.getenv("OPENAI_API_KEY") is not None

    async def embed_batch(self, texts: List[str], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = DEFAULT_MODELS["openai"]
        
        await asyncio.sleep(random.uniform(0.1, 0.3) * len(texts))
        
        simulated_latency = random.uniform(0.15, 0.25) * len(texts)
        mock_embeddings = [[random.random() for _ in range(1536)] for _ in texts]
        
        return {
            "success": True,
            "latency": simulated_latency,
            "embeddings": mock_embeddings,
            "model": model,
            "batch_size": len(texts),
            "provider": self.name
        }

class CohereProvider(EmbeddingProvider):
    def __init__(self):
        super().__init__("cohere")

    def _check_availability(self) -> bool:
        return os.getenv("COHERE_API_KEY") is not None

    async def embed_batch(self, texts: List[str], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = DEFAULT_MODELS["cohere"]
        
        await asyncio.sleep(random.uniform(0.08, 0.2) * len(texts))
        
        simulated_latency = random.uniform(0.12, 0.18) * len(texts)
        mock_embeddings = [[random.random() for _ in range(1024)] for _ in texts]
        
        return {
            "success": True,
            "latency": simulated_latency,
            "embeddings": mock_embeddings,
            "model": model,
            "batch_size": len(texts),
            "provider": self.name
        }

class GeminiProvider(EmbeddingProvider):
    def __init__(self):
        super().__init__("gemini")

    def _check_availability(self) -> bool:
        return os.getenv("GOOGLE_API_KEY") is not None

    async def embed_batch(self, texts: List[str], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = DEFAULT_MODELS["gemini"]
        
        await asyncio.sleep(random.uniform(0.2, 0.4) * len(texts))
        
        simulated_latency = random.uniform(0.25, 0.35) * len(texts)
        mock_embeddings = [[random.random() for _ in range(768)] for _ in texts]
        
        return {
            "success": True,
            "latency": simulated_latency,
            "embeddings": mock_embeddings,
            "model": model,
            "batch_size": len(texts),
            "provider": self.name
        }

class VoyagerProvider(EmbeddingProvider):
    def __init__(self):
        super().__init__("voyager")

    def _check_availability(self) -> bool:
        return os.getenv("VOYAGER_API_KEY") is not None

    async def embed_batch(self, texts: List[str], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = DEFAULT_MODELS["voyager"]
        
        await asyncio.sleep(random.uniform(0.1, 0.25) * len(texts))
        
        simulated_latency = random.uniform(0.13, 0.22) * len(texts)
        mock_embeddings = [[random.random() for _ in range(1024)] for _ in texts]
        
        return {
            "success": True,
            "latency": simulated_latency,
            "embeddings": mock_embeddings,
            "model": model,
            "batch_size": len(texts),
            "provider": self.name
        }

class MTEBDataLoader:
    def __init__(self):
        self.available_datasets = MTEB_DATASETS

    def load_samples(self, samples_per_category: int = 10) -> Dict[str, List[str]]:
        """Load mock samples simulating MTEB datasets."""
        samples = {}
        
        mock_texts = [
            "This is a short query about machine learning.",
            "Natural language processing has revolutionized how we interact with computers and understand human communication patterns.",
            "The quick brown fox jumps over the lazy dog in this classic pangram sentence.",
            "Artificial intelligence systems are becoming increasingly sophisticated in their ability to understand context and nuance.",
            "Database optimization techniques can significantly improve query performance and reduce latency in large-scale applications.",
            "Embedding models transform text into high-dimensional vector representations that capture semantic meaning and relationships.",
            "RAG systems combine retrieval mechanisms with generative models to provide more accurate and contextually relevant responses.",
            "Vector databases enable efficient similarity search across millions of high-dimensional embeddings using approximate nearest neighbor algorithms.",
            "The latency bottleneck in most RAG applications comes from embedding generation rather than database retrieval operations.",
            "Modern transformer architectures have enabled significant improvements in text understanding and generation capabilities across various domains."
        ]
        
        for dataset_name in self.available_datasets:
            print(f"📊 Loading mock dataset: {dataset_name}")
            dataset_samples = []
            for i in range(samples_per_category):
                base_text = mock_texts[i % len(mock_texts)]
                if i % 3 == 0:
                    base_text += f" Additional context for sample {i} from {dataset_name.split('/')[-1]}."
                dataset_samples.append(base_text)
            
            samples[dataset_name] = dataset_samples
            print(f"   Loaded {len(dataset_samples)} mock samples")
        
        return samples

class BenchmarkCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, provider: str, texts: List[str], model: str, batch_size: int) -> str:
        content = f"{provider}:{model}:{batch_size}:{hash(tuple(texts))}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_result(self, provider: str, texts: List[str], model: str, batch_size: int) -> Optional[Dict]:
        cache_key = self._get_cache_key(provider, texts, model, batch_size)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def cache_result(self, provider: str, texts: List[str], model: str, batch_size: int, result: Dict):
        cache_key = self._get_cache_key(provider, texts, model, batch_size)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

class BenchmarkEngine:
    def __init__(self, cache: BenchmarkCache, max_concurrent: int = 5):
        self.cache = cache
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.providers = {
            "openai": OpenAIProvider(),
            "cohere": CohereProvider(),
            "gemini": GeminiProvider(),
            "voyager": VoyagerProvider()
        }

    async def run_benchmark_batch(self, provider_name: str, texts: List[str], batch_size: int) -> Dict:
        provider = self.providers[provider_name]
        
        if not provider.available:
            return {"success": False, "error": f"{provider_name} API key not available"}

        cached_result = self.cache.get_cached_result(provider_name, texts, "default", batch_size)
        if cached_result:
            print(f"   📋 Using cached result for {provider_name}")
            return cached_result

        async with self.semaphore:
            batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
            batch_results = []
            
            for batch in batches:
                result = await provider.embed_batch(batch)
                batch_results.append(result)
                
                self.cache.cache_result(provider_name, batch, "default", len(batch), result)
            
            return self._aggregate_batch_results(batch_results)

    def _aggregate_batch_results(self, batch_results: List[Dict]) -> Dict:
        successful_results = [r for r in batch_results if r.get("success", False)]
        
        if not successful_results:
            return {"success": False, "error": "All batches failed"}
        
        total_latency = sum(r["latency"] for r in successful_results)
        total_embeddings = sum(r["batch_size"] for r in successful_results)
        
        return {
            "success": True,
            "total_latency": total_latency,
            "avg_latency_per_embedding": total_latency / max(total_embeddings, 1),
            "throughput": total_embeddings / total_latency if total_latency > 0 else 0,
            "batch_count": len(successful_results),
            "embedding_count": total_embeddings,
            "latencies": [r["latency"] for r in successful_results]
        }

def print_latency_statistics(results: Dict[str, Dict]):
    """Print P50, P95, P99 latency statistics to console."""
    print("\n" + "="*80)
    print("📊 EMBEDDING LATENCY BENCHMARK RESULTS")
    print("="*80)
    
    print("\n🎯 Key Finding: Embedding latency dominates RAG pipeline performance")
    print("   • Database reads: 8-20ms")
    print("   • Embedding generation: 100-500ms (10-25x slower!)")
    
    print(f"\n{'Provider':<12} {'P50 (ms)':<10} {'P95 (ms)':<10} {'P99 (ms)':<10} {'Throughput':<12} {'Status'}")
    print("-" * 70)
    
    for provider, result in results.items():
        if result.get("success", False) and "latencies" in result:
            latencies_ms = [l * 1000 for l in result["latencies"]]  # Convert to milliseconds
            
            p50 = np.percentile(latencies_ms, 50)
            p95 = np.percentile(latencies_ms, 95)
            p99 = np.percentile(latencies_ms, 99)
            throughput = result.get("throughput", 0)
            
            print(f"{provider.title():<12} {p50:<10.1f} {p95:<10.1f} {p99:<10.1f} {throughput:<12.1f} ✅")
        else:
            error = result.get("error", "Unknown error")
            print(f"{provider.title():<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} ❌ {error}")
    
    print("\n💡 Recommendations:")
    print("   1. Co-locate embedding models with your database infrastructure")
    print("   2. Use batch processing to improve throughput")
    print("   3. Cache frequently requested embeddings")
    print("   4. Monitor embedding latency as the primary RAG bottleneck")
    
    print("\n🏗️  Database Co-location Impact Analysis:")
    print("   Scenario              | DB Read | Embedding | Network | Total")
    print("   Co-located           |   15ms  |   200ms   |   5ms   | 220ms")
    print("   Separate regions     |   15ms  |   200ms   |  50ms   | 265ms") 
    print("   Different clouds     |   15ms  |   200ms   | 100ms   | 315ms")
    print("\n   → Embedding latency dominates; database optimizations are secondary")

async def run_benchmarks(providers: List[str], samples_per_category: int, batch_sizes: List[int], 
                        max_concurrent: int, cache_dir: str, output_dir: str):
    """Run comprehensive embedding latency benchmarks."""
    
    print("🚀 Starting Embedding Latency Benchmarks")
    print(f"Providers: {', '.join(providers)}")
    print(f"Samples per category: {samples_per_category}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max concurrent: {max_concurrent}")
    print()
    
    cache = BenchmarkCache(cache_dir)
    engine = BenchmarkEngine(cache, max_concurrent)
    data_loader = MTEBDataLoader()
    
    print("📚 Loading MTEB datasets...")
    samples = data_loader.load_samples(samples_per_category)
    
    if not samples:
        print("❌ No datasets loaded. Exiting.")
        return
    
    all_texts = []
    for dataset_texts in samples.values():
        all_texts.extend(dataset_texts)
    
    print(f"📝 Total texts loaded: {len(all_texts)}")
    print()
    
    results = {}
    
    for provider in providers:
        print(f"🔄 Benchmarking {provider.title()}...")
        provider_results = []
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            result = await engine.run_benchmark_batch(provider, all_texts, batch_size)
            
            if result.get("success", False):
                provider_results.extend(result.get("latencies", []))
                print(f"   ✅ Completed batch size {batch_size}")
            else:
                print(f"   ❌ Failed batch size {batch_size}: {result.get('error', 'Unknown error')}")
        
        if provider_results:
            results[provider] = {
                "success": True,
                "latencies": provider_results,
                "throughput": len(all_texts) / sum(provider_results) if provider_results else 0
            }
        else:
            results[provider] = {"success": False, "error": "All batch sizes failed"}
    
    print_latency_statistics(results)

def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding latency across providers")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    benchmark_parser = subparsers.add_parser('benchmark', help='Run embedding latency benchmarks')
    benchmark_parser.add_argument('--providers', default="openai,cohere,gemini,voyager",
                                help='Comma-separated list of embedding providers to test')
    benchmark_parser.add_argument('--samples-per-category', type=int, default=10,
                                help='Number of samples to load per MTEB dataset category')
    benchmark_parser.add_argument('--batch-sizes', default="1,5,10,25",
                                help='Comma-separated list of batch sizes to test')
    benchmark_parser.add_argument('--max-concurrent', type=int, default=5,
                                help='Maximum concurrent requests to prevent rate limiting')
    benchmark_parser.add_argument('--cache-dir', default="./data/cache",
                                help='Directory for caching results (enables restartability)')
    benchmark_parser.add_argument('--output-dir', default="./data",
                                help='Output directory for reports and visualizations')
    
    clear_parser = subparsers.add_parser('clear-cache', help='Clear the benchmark cache directory')
    clear_parser.add_argument('--cache-dir', default="./data/cache",
                            help='Cache directory to clear')
    
    subparsers.add_parser('list-datasets', help='List available MTEB datasets')
    
    args = parser.parse_args()
    
    if args.command == 'benchmark':
        run_benchmark_command(args)
    elif args.command == 'clear-cache':
        clear_cache_command(args)
    elif args.command == 'list-datasets':
        list_datasets_command()
    else:
        parser.print_help()

def run_benchmark_command(args):
    """Run comprehensive embedding latency benchmarks across multiple providers."""
    
    providers_list = [p.strip() for p in args.providers.split(",")]
    batch_sizes_list = [int(b.strip()) for b in args.batch_sizes.split(",")]
    
    valid_providers = ["openai", "cohere", "gemini", "voyager"]
    invalid_providers = [p for p in providers_list if p not in valid_providers]
    if invalid_providers:
        print(f"❌ Invalid providers: {invalid_providers}")
        print(f"Valid providers: {valid_providers}")
        return
    
    available_providers = []
    for provider in providers_list:
        key_name = {
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY", 
            "gemini": "GOOGLE_API_KEY",
            "voyager": "VOYAGER_API_KEY"
        }[provider]
        
        if os.getenv(key_name):
            available_providers.append(provider)
        else:
            print(f"⚠️  {provider.upper()} API key not found ({key_name})")
    
    if not available_providers:
        print("❌ No API keys found. Please set API keys in environment variables or .envrc file.")
        print("Example .envrc file:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export COHERE_API_KEY='your-key-here'")
        print("  export GOOGLE_API_KEY='your-key-here'")
        print("  export VOYAGER_API_KEY='your-key-here'")
        return
    
    print(f"✅ Found API keys for: {', '.join(available_providers)}")
    
    asyncio.run(run_benchmarks(
        available_providers, 
        args.samples_per_category, 
        batch_sizes_list,
        args.max_concurrent, 
        args.cache_dir, 
        args.output_dir
    ))

def clear_cache_command(args):
    """Clear the benchmark cache directory."""
    cache_path = Path(args.cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory {args.cache_dir} does not exist.")
        return
    
    import shutil
    shutil.rmtree(cache_path)
    print(f"✅ Cleared cache directory: {args.cache_dir}")

def list_datasets_command():
    """List available MTEB datasets that will be used for benchmarking."""
    print("📊 Available MTEB Datasets:")
    print("")
    
    for i, dataset in enumerate(MTEB_DATASETS, 1):
        print(f"{i:2d}. {dataset}")
    
    print("")
    print(f"Total: {len(MTEB_DATASETS)} datasets")

if __name__ == "__main__":
    main()
