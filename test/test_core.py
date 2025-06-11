import pytest
import os
from unittest.mock import patch, MagicMock
import sys

os.environ.setdefault('REDIS_URL', 'redis://localhost:6379')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_server.server import (
    resolve_model_path,
    is_llamacpp_available,
    chat_with_ollama
)

# Test models
TEST_LLAMACPP_MODEL = 'DeepSeek-V3-0324-UD-IQ2_XXS'
TEST_OLLAMA_MODEL = 'deepseek-coder-v2:latest'


class TestModelResolution:
    """Test core model resolution functionality."""
    
    @patch('ai_server.server.glob.glob')
    def test_resolve_model_path_found(self, mock_glob):
        """Test model path resolution when model exists."""
        model_path = f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/{TEST_LLAMACPP_MODEL}.gguf'
        mock_glob.return_value = [model_path]
        
        result = resolve_model_path(TEST_LLAMACPP_MODEL)
        
        assert result == model_path
        mock_glob.assert_called_once_with(f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/*.gguf')
    
    @patch('ai_server.server.glob.glob')
    def test_resolve_model_path_not_found(self, mock_glob):
        """Test model path resolution when model doesn't exist."""
        mock_glob.return_value = []
        
        result = resolve_model_path('nonexistent-model')
        
        assert result is None
    
    @patch('ai_server.server.resolve_model_path')
    def test_is_llamacpp_available_true(self, mock_resolve):
        """Test model availability check when model exists."""
        mock_resolve.return_value = f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/{TEST_LLAMACPP_MODEL}.gguf'
        
        result = is_llamacpp_available(TEST_LLAMACPP_MODEL)
        
        assert result is True
        mock_resolve.assert_called_once_with(TEST_LLAMACPP_MODEL)
    
    @patch('ai_server.server.resolve_model_path')
    def test_is_llamacpp_available_false(self, mock_resolve):
        """Test model availability check when model doesn't exist."""
        mock_resolve.return_value = None
        
        result = is_llamacpp_available('nonexistent-model')
        
        assert result is False


class TestOllamaCore:
    """Test core ollama functionality used as fallback."""
    
    @patch('ai_server.server.ollama.chat')
    def test_chat_with_ollama_success(self, mock_ollama_chat):
        """Test successful chat with ollama."""
        mock_response = MagicMock()
        mock_response.message.content = "Hello! I'm DeepSeek Coder V2. I can help you with coding tasks."
        mock_ollama_chat.return_value = mock_response
        
        result = chat_with_ollama(TEST_OLLAMA_MODEL, 'Help me write a Python function')
        
        assert result == "Hello! I'm DeepSeek Coder V2. I can help you with coding tasks."
        mock_ollama_chat.assert_called_once_with(
            model=TEST_OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': 'Help me write a Python function'}],
            stream=False
        )
    
    @patch('ai_server.server.ollama.chat')
    def test_chat_with_ollama_service_unavailable(self, mock_ollama_chat):
        """Test ollama chat when service is unavailable."""
        mock_ollama_chat.side_effect = Exception("Ollama service is not running")
        
        with pytest.raises(Exception, match="Ollama service is not running"):
            chat_with_ollama(TEST_OLLAMA_MODEL, 'Hello')
    
    @patch('ai_server.server.ollama.chat')
    def test_chat_with_ollama_model_not_found(self, mock_ollama_chat):
        """Test ollama chat when model is not found."""
        mock_ollama_chat.side_effect = Exception("model 'nonexistent:latest' not found")
        
        with pytest.raises(Exception, match="model 'nonexistent:latest' not found"):
            chat_with_ollama('nonexistent:latest', 'Hello') 
            