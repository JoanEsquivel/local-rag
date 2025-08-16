# Environment Variables Configuration

This document describes the environment variables available for configuring the RAG system.

## 🔧 Configuration Options

Create a `.env` file in the project root with the following variables:

### Embedding Provider Selection

```env
# Options: "ollama" (default), "openai"
EMBEDDING_PROVIDER=ollama
```

### Ollama Configuration (Local)

```env
# Ollama embedding model name (default: bge-m3:567m)
OLLAMA_EMBEDDING_MODEL=bge-m3:567m

# Ollama server URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

### OpenAI Configuration (Cloud)

```env
# OpenAI API key (required only if using OpenAI embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI embedding model (default: text-embedding-3-large)
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

## 🔄 Usage Examples

### Example 1: Using Ollama (Default)
```env
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=bge-m3:567m
OLLAMA_BASE_URL=http://localhost:11434
```

### Example 2: Using OpenAI
```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### Example 3: Custom Ollama Setup
```env
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://custom-ollama-server:11434
```

## 🎯 Benefits of New Architecture

- **Flexibility**: Switch between local and cloud embeddings easily
- **Error Handling**: Automatic health checks and clear error messages
- **Configuration**: Environment-based configuration for different environments
- **Logging**: Better visibility into what's happening
- **Validation**: Automatic model availability checks
