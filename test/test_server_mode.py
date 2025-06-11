import pytest
import os
from unittest.mock import patch, MagicMock
import sys

os.environ.setdefault('REDIS_URL', 'redis://localhost:6379')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_server.server import (
    chat_with_llama_server_http,
    chat_with_model
)

# Test models
TEST_LLAMACPP_MODEL = 'DeepSeek-V3-0324-UD-IQ2_XXS'
TEST_OLLAMA_MODEL = 'deepseek-coder-v2:latest'


class TestLlamaServerHTTP:
    """Test llama.cpp server HTTP functionality."""
    
    @patch('ai_server.server.requests.post')
    @patch('ai_server.server.LLAMA_SERVER_URL', 'http://localhost:8080')
    def test_chat_with_llama_server_http_success(self, mock_post):
        """Test successful HTTP chat with llama-server."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Server response from DeepSeek V3'}}]
        }
        mock_post.return_value = mock_response
        
        result = chat_with_llama_server_http(TEST_LLAMACPP_MODEL, 'Hello from server')
        
        assert result == "Server response from DeepSeek V3"
        
        # Verify correct API call
        args, kwargs = mock_post.call_args
        assert args[0] == 'http://localhost:8080/v1/chat/completions'
        assert kwargs['json']['model'] == TEST_LLAMACPP_MODEL
        assert kwargs['json']['messages'][0]['content'] == 'Hello from server'
    
    @patch('ai_server.server.LLAMA_SERVER_URL', None)
    def test_chat_with_llama_server_http_no_url(self):
        """Test HTTP chat when LLAMA_SERVER_URL is not set."""
        with pytest.raises(Exception, match="LLAMA_SERVER_URL environment variable not set"):
            chat_with_llama_server_http(TEST_LLAMACPP_MODEL, 'Hello')
    
    @patch('ai_server.server.requests.post')
    @patch('ai_server.server.LLAMA_SERVER_URL', 'http://localhost:8080')
    def test_chat_with_llama_server_http_error_response(self, mock_post):
        """Test HTTP chat when server returns error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception, match="Llama-server HTTP error"):
            chat_with_llama_server_http(TEST_LLAMACPP_MODEL, 'Hello')
    
    @patch('ai_server.server.requests.post')
    @patch('ai_server.server.LLAMA_SERVER_URL', 'http://localhost:8080')
    def test_chat_with_llama_server_http_invalid_response_format(self, mock_post):
        """Test HTTP chat when server returns invalid response format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'error': 'Invalid request'}  # Missing choices
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception, match="Invalid response format from llama-server"):
            chat_with_llama_server_http(TEST_LLAMACPP_MODEL, 'Hello')


class TestServerModeRouting:
    """Test server mode routing and fallback logic."""
    
    @patch('ai_server.server.chat_with_llama_server_http')
    @patch('ai_server.server.is_llamacpp_available')
    @patch('ai_server.server.LLAMA_SERVER_URL', 'http://localhost:8080')
    def test_server_mode_uses_llamacpp_when_available(self, mock_available, mock_chat_server):
        """Test server mode routes to llama-server when model is available."""
        mock_available.return_value = True
        mock_chat_server.return_value = "Server response from DeepSeek V3"
        
        result = chat_with_model(TEST_LLAMACPP_MODEL, 'Explain code', llama_mode='server')
        
        assert result == "Server response from DeepSeek V3"
        mock_available.assert_called_once_with(TEST_LLAMACPP_MODEL)
        mock_chat_server.assert_called_once_with(TEST_LLAMACPP_MODEL, 'Explain code')
    
    @patch('ai_server.server.chat_with_ollama')
    @patch('ai_server.server.is_llamacpp_available')
    @patch('ai_server.server.LLAMA_SERVER_URL', 'http://localhost:8080')
    def test_server_mode_fallback_to_ollama_when_unavailable(self, mock_available, mock_chat_ollama):
        """Test server mode falls back to ollama when model not available in llama.cpp."""
        mock_available.return_value = False
        mock_chat_ollama.return_value = "Ollama fallback response"
        
        result = chat_with_model(TEST_OLLAMA_MODEL, 'Debug code', llama_mode='server')
        
        assert result == "Ollama fallback response"
        mock_available.assert_called_once_with(TEST_OLLAMA_MODEL)
        mock_chat_ollama.assert_called_once_with(TEST_OLLAMA_MODEL, 'Debug code')
    
    @patch('ai_server.server.is_llamacpp_available')
    @patch('ai_server.server.LLAMA_SERVER_URL', None)
    def test_server_mode_requires_server_url(self, mock_available):
        """Test server mode requires LLAMA_SERVER_URL to be set."""
        mock_available.return_value = True
        
        with pytest.raises(Exception, match="LLAMA_SERVER_URL environment variable not set"):
            chat_with_model(TEST_LLAMACPP_MODEL, 'Hello', llama_mode='server')
    
    @patch('ai_server.server.is_llamacpp_available')
    def test_invalid_llama_mode_raises_error(self, mock_available):
        """Test that invalid llama_mode raises ValueError."""
        mock_available.return_value = True
        
        with pytest.raises(ValueError, match="Invalid llama_mode: 'invalid'"):
            chat_with_model(TEST_LLAMACPP_MODEL, 'Hello', llama_mode='invalid')


class TestServerModeIntegration:
    """Test complete server mode integration flows."""
    
    @patch('ai_server.server.requests.post')
    @patch('ai_server.server.glob.glob')
    @patch('ai_server.server.LLAMA_SERVER_URL', 'http://localhost:8080')
    def test_complete_server_flow_with_real_model(self, mock_glob, mock_requests):
        """Test complete server flow: model resolution → HTTP API call."""
        model_path = f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/{TEST_LLAMACPP_MODEL}.gguf'
        
        # Mock model found (only checked once for availability in server mode)
        mock_glob.return_value = [model_path]
        
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Server integration test successful!'}}]
        }
        mock_requests.return_value = mock_response
        
        result = chat_with_model(TEST_LLAMACPP_MODEL, 'Integration test', llama_mode='server')
        
        assert result == "Server integration test successful!"
        # In server mode, glob.glob only called once for is_llamacpp_available
        mock_glob.assert_called_once_with(f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/*.gguf')
        mock_requests.assert_called_once()
    
    @patch('ai_server.server.ollama.chat')
    @patch('ai_server.server.glob.glob')
    @patch('ai_server.server.LLAMA_SERVER_URL', 'http://localhost:8080')
    def test_complete_server_fallback_flow_to_ollama(self, mock_glob, mock_ollama):
        """Test complete server fallback flow: model not found → fallback to ollama."""
        # Mock model not found in llama.cpp
        mock_glob.return_value = []
        
        # Mock successful ollama response
        mock_response = MagicMock()
        mock_response.message.content = "Ollama server fallback integration test successful!"
        mock_ollama.return_value = mock_response
        
        result = chat_with_model(TEST_OLLAMA_MODEL, 'Fallback test', llama_mode='server')
        
        assert result == "Ollama server fallback integration test successful!"
        mock_glob.assert_called_once_with(f'/data1/GGUF/{TEST_OLLAMA_MODEL}/*.gguf')
        mock_ollama.assert_called_once() 