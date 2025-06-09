from flask import Flask, request, jsonify, abort
import ollama
import subprocess
import os
import requests
from typing import Optional
from dotenv import load_dotenv
import glob

from .redis_helper import REDIS_CONNECTION

# Load environment variables from .env file
load_dotenv()

app = Flask('AI server')

DEFAULT_MODEL = 'deepseek-coder-v2:latest'

# Llama.cpp configuration
LLAMA_CPP_CLI = '/data1/llama.cpp/bin/llama-cli'
GGUF_DIR = '/data1/GGUF'

_llama_server_url = os.getenv('LLAMA_SERVER_URL')  # e.g., http://localhost:8080 or localhost:8080
LLAMA_SERVER_URL = f"http://{_llama_server_url}" if _llama_server_url and not _llama_server_url.startswith(('http://', 'https://')) else _llama_server_url

def chat_with_llama_server_http(model: str, content: str, system_prompt: Optional[str] = None, timeout: int = 300) -> str:
    """Handle chat using llama-server HTTP API."""
    if not LLAMA_SERVER_URL:
        raise Exception("LLAMA_SERVER_URL environment variable not set")
    
    try:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': content})
        
        response = requests.post(
            f'{LLAMA_SERVER_URL}/v1/chat/completions',
            json={
                'model': model,
                'messages': messages,
                'stream': False,
                'max_tokens': 512
            },
            headers={'Content-Type': 'application/json'},
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            else:
                raise Exception("Invalid response format from llama-server")
        else:
            raise Exception(f"Llama-server HTTP error")
                
    except requests.Timeout:
        raise Exception(f"Llama-server request timed out for model {model}")
    except requests.RequestException as e:
        raise Exception(f"Llama-server request failed: {str(e)}")

def resolve_model_path(model: str) -> Optional[str]:
    """Resolve model name to full GGUF file path using glob pattern."""
    pattern = os.path.join(GGUF_DIR, model, "*.gguf")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def is_llamacpp_available(model: str) -> bool:
    """Check if model is available in llama.cpp."""
    return resolve_model_path(model) is not None

def chat_with_ollama(model: str, content: str, system_prompt: Optional[str] = None) -> str:
    """Handle chat using ollama."""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': content})
    
    response = ollama.chat(
        model=model,
        messages=messages,
        stream=False
    )
    return response.message.content

def chat_with_llamacpp(model: str, content: str, system_prompt: Optional[str] = None, timeout: int = 300) -> str:
    """Handle chat using llama.cpp CLI."""
    model_path = resolve_model_path(model)
    
    if not model_path:
        raise ValueError(f"Model not found: {model}")
    
    # Format prompt with system prompt if provided
    if system_prompt:
        formatted_prompt = f"System: {system_prompt}\n\nUser: {content}\n\nAssistant:"
    else:
        formatted_prompt = content
    
    cmd = [
        LLAMA_CPP_CLI,
        '-m', model_path,
        '--n-gpu-layers', '40',
        '-p', formatted_prompt,
        '-n', '512',
        '--single-turn'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            timeout=timeout,
            check=True
        )
        
        stdout_text = result.stdout.decode('utf-8', errors='replace')

        response = stdout_text.strip()
        return response if response else "No response generated."
        
    except subprocess.TimeoutExpired:
        raise Exception(f"Llama.cpp request timed out for model {model}")
    except subprocess.CalledProcessError as e:
        stderr_text = ""
        if e.stderr:
            stderr_text = e.stderr.decode('utf-8', errors='replace')
        raise Exception(f"Llama.cpp failed for {model}: {stderr_text.strip() if stderr_text else 'Unknown error'}")
    except FileNotFoundError:
        raise Exception("Llama.cpp CLI not found")

def chat_with_model(model: str, content: str, llama_mode: str = "cli", system_prompt: Optional[str] = None) -> str:
    """Route chat request based on llama_mode: server (external), cli, or ollama fallback."""
    if is_llamacpp_available(model):
        if llama_mode == "server":
            if not LLAMA_SERVER_URL:
                raise Exception("LLAMA_SERVER_URL environment variable not set for server mode")
            return chat_with_llama_server_http(model, content, system_prompt)
        elif llama_mode == "cli":
            return chat_with_llamacpp(model, content, system_prompt)
        else:
            raise ValueError(f"Invalid llama_mode: '{llama_mode}'. Valid options are 'server' or 'cli'.")
    else:
        # Model not available in llama.cpp, use ollama
        return chat_with_ollama(model, content, system_prompt)


def authenticate() -> str:
    """Authenticate the given request using an API key."""
    api_key = request.headers.get('X-API-KEY')
    if not api_key:
        abort(401, description="Missing API key")

    user = REDIS_CONNECTION.get(f"api-key:{api_key}")
    if not user:
        abort(401, description="Invalid API key")

    return user


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat request with optional llama_mode and system prompt parameters."""
    authenticate()
    params = request.get_json()
    model = params.get('model', DEFAULT_MODEL)
    content = params.get('content', '')
    llama_mode = params.get('llama_mode', 'cli')
    system_prompt = params.get('system_prompt')
    
    if not content.strip():
        abort(400, description='Missing prompt content')

    response_content = chat_with_model(model, content, llama_mode, system_prompt)
    return jsonify(response_content)

@app.errorhandler(Exception)
def internal_error(error):
    return jsonify({"error": str(error)}), 500
