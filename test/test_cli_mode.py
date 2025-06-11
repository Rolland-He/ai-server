import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock
import sys

os.environ.setdefault('REDIS_URL', 'redis://localhost:6379')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_server.server import (
    chat_with_llamacpp,
    chat_with_model
)

# Test models
TEST_LLAMACPP_MODEL = 'DeepSeek-V3-0324-UD-IQ2_XXS'
TEST_OLLAMA_MODEL = 'deepseek-coder-v2:latest'


class TestLlamaCppCLI:
    """Test llama.cpp CLI execution."""
    
    @patch('ai_server.server.subprocess.run')
    @patch('ai_server.server.resolve_model_path')
    def test_chat_with_llamacpp_success(self, mock_resolve, mock_subprocess):
        """Test successful CLI chat with llama.cpp."""
        model_path = f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/{TEST_LLAMACPP_MODEL}.gguf'
        mock_resolve.return_value = model_path
        
        mock_result = MagicMock()
        mock_result.stdout = b'I can help you with DeepSeek V3.'
        mock_subprocess.return_value = mock_result
        
        result = chat_with_llamacpp(TEST_LLAMACPP_MODEL, 'Hello, can you help me code?')
        
        assert result == "I can help you with DeepSeek V3."
        mock_resolve.assert_called_once_with(TEST_LLAMACPP_MODEL)
        
        # Verify correct CLI command structure
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        assert '/data1/llama.cpp/bin/llama-cli' in cmd
        assert '-m' in cmd and model_path in cmd
        assert '--n-gpu-layers' in cmd and '40' in cmd
        assert '--single-turn' in cmd
    
    @patch('ai_server.server.resolve_model_path')
    def test_chat_with_llamacpp_model_not_found(self, mock_resolve):
        """Test CLI chat when model is not found."""
        mock_resolve.return_value = None
        
        with pytest.raises(ValueError, match="Model not found: nonexistent-model"):
            chat_with_llamacpp('nonexistent-model', 'Hello')
    
    @patch('ai_server.server.subprocess.run')
    @patch('ai_server.server.resolve_model_path')
    def test_chat_with_llamacpp_subprocess_error(self, mock_resolve, mock_subprocess):
        """Test CLI chat when subprocess fails."""
        mock_resolve.return_value = f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/{TEST_LLAMACPP_MODEL}.gguf'
        
        error = subprocess.CalledProcessError(1, 'cmd')
        error.stderr = b'CUDA out of memory'
        mock_subprocess.side_effect = error
        
        with pytest.raises(Exception, match=f"Llama.cpp failed for {TEST_LLAMACPP_MODEL}: CUDA out of memory"):
            chat_with_llamacpp(TEST_LLAMACPP_MODEL, 'Hello')


class TestCLIModeRouting:
    """Test CLI mode routing and fallback logic."""
    
    @patch('ai_server.server.chat_with_llamacpp')
    @patch('ai_server.server.is_llamacpp_available')
    def test_cli_mode_uses_llamacpp_when_available(self, mock_available, mock_chat_llamacpp):
        """Test CLI mode routes to llama.cpp when model is available."""
        mock_available.return_value = True
        mock_chat_llamacpp.return_value = "CLI response from DeepSeek V3"
        
        result = chat_with_model(TEST_LLAMACPP_MODEL, 'Write a function', llama_mode='cli')
        
        assert result == "CLI response from DeepSeek V3"
        mock_available.assert_called_once_with(TEST_LLAMACPP_MODEL)
        mock_chat_llamacpp.assert_called_once_with(TEST_LLAMACPP_MODEL, 'Write a function')
    
    @patch('ai_server.server.chat_with_ollama')
    @patch('ai_server.server.is_llamacpp_available')
    def test_cli_mode_fallback_to_ollama_when_unavailable(self, mock_available, mock_chat_ollama):
        """Test CLI mode falls back to ollama when model not available in llama.cpp."""
        mock_available.return_value = False
        mock_chat_ollama.return_value = "Ollama response from DeepSeek Coder"
        
        result = chat_with_model(TEST_OLLAMA_MODEL, 'Help with coding', llama_mode='cli')
        
        assert result == "Ollama response from DeepSeek Coder"
        mock_available.assert_called_once_with(TEST_OLLAMA_MODEL)
        mock_chat_ollama.assert_called_once_with(TEST_OLLAMA_MODEL, 'Help with coding')
    
    @patch('ai_server.server.chat_with_llamacpp')
    @patch('ai_server.server.is_llamacpp_available')
    def test_default_mode_is_cli(self, mock_available, mock_chat_llamacpp):
        """Test that default mode is CLI when no llama_mode specified."""
        mock_available.return_value = True
        mock_chat_llamacpp.return_value = "Default CLI mode response"
        
        result = chat_with_model(TEST_LLAMACPP_MODEL, 'Help me')  # No llama_mode specified
        
        assert result == "Default CLI mode response"
        mock_available.assert_called_once_with(TEST_LLAMACPP_MODEL)
        mock_chat_llamacpp.assert_called_once_with(TEST_LLAMACPP_MODEL, 'Help me')


class TestCLIModeIntegration:
    """Test complete CLI mode integration flows."""
    
    @patch('ai_server.server.subprocess.run')
    @patch('ai_server.server.glob.glob')
    def test_complete_cli_flow_with_real_model(self, mock_glob, mock_subprocess):
        """Test complete CLI flow: model resolution → CLI execution."""
        model_path = f'/data1/GGUF/{TEST_LLAMACPP_MODEL}/{TEST_LLAMACPP_MODEL}.gguf'
        
        mock_glob.return_value = [model_path]
        mock_result = MagicMock()
        mock_result.stdout = b'Complete integration test successful with DeepSeek V3!'
        mock_subprocess.return_value = mock_result
        
        result = chat_with_model(TEST_LLAMACPP_MODEL, 'Integration test', llama_mode='cli')
        
        assert result == "Complete integration test successful with DeepSeek V3!"
        # Verify glob called twice: once for availability check, once for CLI execution
        assert mock_glob.call_count == 2
        mock_subprocess.assert_called_once()
    
    @patch('ai_server.server.ollama.chat')
    @patch('ai_server.server.glob.glob')
    def test_complete_cli_fallback_flow_to_ollama(self, mock_glob, mock_ollama):
        """Test complete CLI fallback flow: model not found → fallback to ollama."""
        # Mock model not found in llama.cpp
        mock_glob.return_value = []
        
        # Mock successful ollama response
        mock_response = MagicMock()
        mock_response.message.content = "Ollama CLI fallback integration test successful!"
        mock_ollama.return_value = mock_response
        
        result = chat_with_model(TEST_OLLAMA_MODEL, 'Fallback test', llama_mode='cli')
        
        assert result == "Ollama CLI fallback integration test successful!"
        mock_glob.assert_called_once_with(f'/data1/GGUF/{TEST_OLLAMA_MODEL}/*.gguf')
        mock_ollama.assert_called_once() 