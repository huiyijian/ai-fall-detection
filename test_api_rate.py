import time
import requests
import json
import concurrent.futures
import statistics

API_URL = "https://models.sjtu.edu.cn/api/v1/chat/completions"
API_KEY = "sk-8Rhs83j2UcIWs0aPWqT2aw"

# List of models to test
MODELS = [
    "deepseek-r1",
    "deepseek-v3",
    "qwen3coder",
    "qwen3vl"
]

def test_request(model_name, req_id):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Use a simple prompt to test basic responsiveness
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello! Reply with 'OK'."}],
        "max_tokens": 10,
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            usage = result.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            return {
                "success": True,
                "latency": latency,
                "tokens": total_tokens,
                "model": model_name,
                "status": 200
            }
        else:
            return {
                "success": False,
                "latency": latency,
                "tokens": 0,
                "model": model_name,
                "status": response.status_code,
                "error": response.text[:100] # Truncate error
            }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "latency": end_time - start_time,
            "tokens": 0,
            "model": model_name,
            "status": -1,
            "error": str(e)
        }

def test_model(model_name, concurrent_requests=3):
    print(f"\nTesting Model: {model_name}")
    print("-" * 50)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(test_request, model_name, i) for i in range(concurrent_requests)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            print(".", end="", flush=True)
    print(" Done")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    avg_latency = 0
    if successful:
        latencies = [r["latency"] for r in successful]
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        print(f"  Success Rate: {len(successful)}/{len(results)}")
        print(f"  Avg Latency:  {avg_latency:.2f}s")
        print(f"  Min/Max:      {min_latency:.2f}s / {max_latency:.2f}s")
    else:
        print(f"  Success Rate: 0/{len(results)}")
        if failed:
             print(f"  Error sample: {failed[0].get('error')}")

    return {
        "model": model_name,
        "avg_latency": avg_latency,
        "success_rate": len(successful) / len(results) if results else 0,
        "total_tokens": sum(r["tokens"] for r in successful)
    }

def main():
    print(f"Starting API Test for {len(MODELS)} models...")
    print(f"API Endpoint: {API_URL}")
    
    summary = []
    
    for model in MODELS:
        # Run 3 concurrent requests per model to test stability
        stats = test_model(model, concurrent_requests=3)
        summary.append(stats)
        time.sleep(1) # Brief pause between models

    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    print(f"{'Model':<15} | {'Avg Latency':<12} | {'Success':<8} | {'Tokens'}")
    print("-" * 60)
    for stat in summary:
        print(f"{stat['model']:<15} | {stat['avg_latency']:.2f}s       | {stat['success_rate']*100:.0f}%      | {stat['total_tokens']}")
    print("-" * 60)

if __name__ == "__main__":
    main()
