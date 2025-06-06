#!/usr/bin/env python3
"""
Test script for llama-mode: server integration.
"""

import requests
import time
import sys

API_KEY = "5a3cced28a86d9be260847f1e6a4e584028fdc3d8bca052810a493380a0cbd95"
BASE_URL = "http://polymouth.teach.cs.toronto.edu:5000"
TEST_PROMPT = "What is Python programming language?"

TEST_MODELS = [
    {
        "name": "DeepSeek-V2.5-IQ1_M",
        "type": "llama.cpp",
        "description": "Small llama.cpp model"
    },
    {
        "name": "DeepSeek-V3-0324-UD-IQ2_XXS", 
        "type": "llama.cpp",
        "description": "Medium llama.cpp model"
    },
    {
        "name": "DeepSeek-V3-0324-UD-Q2_K_XL",
        "type": "llama.cpp", 
        "description": "Large llama.cpp model"
    },
    {
        "name": "deepseek-coder-v2:latest",
        "type": "ollama",
        "description": "Ollama model (should fallback)"
    },
    {
        "name": "deepseek-v2.5:236b-q5_1",
        "type": "ollama",
        "description": "Large Ollama model"
    }
]

def test_model_mode(model, mode):
    """Test a specific model with a specific mode."""
    headers = {
        'X-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': model,
        'content': TEST_PROMPT,
        'llama_mode': mode
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/chat", headers=headers, json=data, timeout=600)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            preview = result[:100] + "..." if len(str(result)) > 100 else str(result)
            print(f"  {mode.upper()}: SUCCESS ({duration:.1f}s) - {preview}")
            return True, duration
        else:
            print(f"  {mode.upper()}: FAILED ({duration:.1f}s) - {response.text[:100]}...")
            return False, duration
            
    except requests.exceptions.Timeout:
        print(f"  {mode.upper()}: TIMEOUT (600s)")
        return False, 600
    except Exception as e:
        print(f"  {mode.upper()}: ERROR - {str(e)[:100]}...")
        return False, 0

def run_tests():
    """Run tests across all models and modes."""
    print("Testing llama-server integration")
    
    results = []
    
    for model_info in TEST_MODELS:
        model = model_info["name"]
        print(f"\nTesting {model} ({model_info['type']})")
        
        # Test CLI mode
        cli_success, cli_time = test_model_mode(model, "cli")
        
        # Test server mode
        server_success, server_time = test_model_mode(model, "server")
        
        results.append({
            'model': model,
            'type': model_info['type'],
            'cli_success': cli_success,
            'cli_time': cli_time,
            'server_success': server_success,
            'server_time': server_time
        })
        
        if cli_success and server_success:
            speed_diff = cli_time - server_time
            faster = "server" if speed_diff > 0 else "cli"
            print(f"  Performance: {faster} mode faster by {abs(speed_diff):.1f}s")
    
    # Summary
    print("RESULTS SUMMARY")

    cli_successes = sum(1 for r in results if r['cli_success'])
    server_successes = sum(1 for r in results if r['server_success'])
    total_models = len(results)
    
    print(f"CLI mode: {cli_successes}/{total_models} successful")
    print(f"Server mode: {server_successes}/{total_models} successful")
    
    print("\nDetailed results:")
    for result in results:
        model = result['model'][:25] + "..." if len(result['model']) > 25 else result['model']
        cli_status = "PASS" if result['cli_success'] else "FAIL"
        server_status = "PASS" if result['server_success'] else "FAIL"
        print(f"{model:<30} | CLI: {cli_status} ({result['cli_time']:.1f}s) | Server: {server_status} ({result['server_time']:.1f}s)")
    
    # Performance analysis
    successful_both = [r for r in results if r['cli_success'] and r['server_success']]
    if successful_both:
        avg_cli_time = sum(r['cli_time'] for r in successful_both) / len(successful_both)
        avg_server_time = sum(r['server_time'] for r in successful_both) / len(successful_both)
        print(f"\nPerformance comparison:")
        print(f"Average CLI time: {avg_cli_time:.1f}s")
        print(f"Average server time: {avg_server_time:.1f}s")
        if avg_server_time < avg_cli_time:
            print(f"Server mode is {avg_cli_time - avg_server_time:.1f}s faster on average")
        else:
            print(f"CLI mode is {avg_server_time - avg_cli_time:.1f}s faster on average")
    
    return cli_successes, server_successes, total_models

def main():
    cli_successes, server_successes, total_models = run_tests()
    
    if cli_successes == total_models and server_successes == total_models:
        print("\nAll tests passed!")
        return 0
    elif server_successes > 0:
        print(f"\nServer mode integration working ({server_successes}/{total_models} models)")
        return 0
    else:
        print(f"\nIssues found - CLI: {cli_successes}/{total_models}, Server: {server_successes}/{total_models}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
    