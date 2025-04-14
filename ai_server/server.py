import os

from flask import Flask, request, jsonify, abort
import ollama
import redis

app = Flask('AI server')

REDIS_URL = os.environ["REDIS_URL"]

REDIS_CONNECTION = redis.Redis.from_url(REDIS_URL)

DEFAULT_MODEL = 'deepseek-coder-v2:latest'

def authenticate() -> str:
    """Authenticate the given request using an API key."""
    api_key = request.headers.get('X-API-KEY')
    if not api_key:
        abort(401, description="Missing API key")

    user = REDIS_CONNECTION.get(f"api_key:{api_key}")
    if not user:
        abort(401, description="Invalid API key")

    return user


@app.route('/chat', methods=['POST'])
def chat():
    """Handle request to ollama.chat."""
    authenticate()
    params = request.get_json()
    model = params.get('model', DEFAULT_MODEL)
    content = params.get('content', '')
    if not content.strip():
        abort(400, description='Missing prompt content')

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': content}],
        stream=False
    )

    return jsonify(response)

@app.errorhandler(Exception)
def internal_error(error):
    return jsonify({"error": str(error)}), 500
