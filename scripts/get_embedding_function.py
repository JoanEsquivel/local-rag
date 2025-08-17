
"""
Embedding Function Module

This module provides a centralized way to get embedding functions for the RAG system.
Supports both OpenAI and Ollama (local) embeddings with proper configuration management.

What is an embedding?
- An embedding is a way to turn text into numbers so that a computer can understand and compare it.
- Words and sentences are just letters to a computer—it doesn't "understand" them like we do.
- An embedding converts text into a set of numbers (a vector) that captures the meaning of the text.
- Similar words or sentences will have similar numbers (vectors), making it easy to compare and find related content.
- Example: "cat" and "dog" will have closer embeddings than "cat" and "airplane" because they are more similar in meaning.

Embeddings are used in many applications, including natural language processing, computer vision, and recommendation systems.
"""

import os
import logging
import requests
from typing import Optional, Union
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants with environment variable support
class EmbeddingConfig:
    """Configuration class for embedding models"""
    
    # Ollama Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # OpenAI Configuration  
    OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Provider selection
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()


def _check_ollama_health(base_url: str) -> bool:
    """Check if Ollama server is running and accessible"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False


def _check_ollama_model_exists(base_url: str, model: str) -> bool:
    """Check if the specified model exists in Ollama"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return model in model_names
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to check Ollama models: {e}")
        return False


def get_ollama_embeddings(
    model: Optional[str] = None, 
    base_url: Optional[str] = None
) -> OllamaEmbeddings:
    """
    Get Ollama embeddings with health checks and error handling
    
    Args:
        model: Ollama model name (defaults to config)
        base_url: Ollama server URL (defaults to config)
        
    Returns:
        OllamaEmbeddings instance
        
    Raises:
        ConnectionError: If Ollama server is not accessible
        ValueError: If model is not available
    """
    model = model or EmbeddingConfig.OLLAMA_MODEL
    base_url = base_url or EmbeddingConfig.OLLAMA_BASE_URL
    
    # Health check
    if not _check_ollama_health(base_url):
        raise ConnectionError(
            f"Ollama server not accessible at {base_url}. "
            f"Please ensure Ollama is running: 'ollama serve'"
        )
    
    # Model availability check
    if not _check_ollama_model_exists(base_url, model):
        available_models_response = requests.get(f"{base_url}/api/tags")
        available_models = [m.get("name", "") for m in available_models_response.json().get("models", [])]
        raise ValueError(
            f"Model '{model}' not found in Ollama. "
            f"Available models: {available_models}. "
            f"Install with: 'ollama pull {model}'"
        )
    
    logger.info(f"Using Ollama embeddings: {model} at {base_url}")
    
    return OllamaEmbeddings(
        model=model,
        base_url=base_url
    )


def get_openai_embeddings(
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> OpenAIEmbeddings:
    """
    Get OpenAI embeddings with proper configuration
    
    Args:
        model: OpenAI model name (defaults to config)
        api_key: OpenAI API key (defaults to env var)
        
    Returns:
        OpenAIEmbeddings instance
        
    Raises:
        ValueError: If API key is not provided
    """
    model = model or EmbeddingConfig.OPENAI_MODEL
    api_key = api_key or EmbeddingConfig.OPENAI_API_KEY
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    logger.info(f"Using OpenAI embeddings: {model}")
    
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key
    )


def get_embedding_function(
    provider: Optional[str] = None,
    **kwargs
) -> Union[OllamaEmbeddings, OpenAIEmbeddings]:
    """
    Get embedding function based on configuration or specified provider
    
    Args:
        provider: "ollama" or "openai" (defaults to config)
        **kwargs: Additional arguments passed to provider-specific functions
        
    Returns:
        Embedding function instance
        
    Raises:
        ValueError: If provider is not supported
        ConnectionError: If provider is not accessible
    """
    provider = provider or EmbeddingConfig.EMBEDDING_PROVIDER
    
    if provider == "ollama":
        return get_ollama_embeddings(**kwargs)
    elif provider == "openai":
        return get_openai_embeddings(**kwargs)
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: 'ollama', 'openai'"
        )


# Store the new function under a different name for tests
get_embedding_function_new = get_embedding_function

# For backward compatibility, keep original function behavior
def get_embedding_function_legacy():
    """Legacy function for backward compatibility - maintains original behavior"""
    return get_ollama_embeddings()  # Original behavior was to return Ollama embeddings

# Override with legacy version to maintain compatibility for existing code
get_embedding_function = get_embedding_function_legacy