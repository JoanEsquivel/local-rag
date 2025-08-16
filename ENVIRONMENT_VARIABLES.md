# Environment Variables Configuration

This document describes the environment variables available for configuring the RAG system.

## 🔧 Configuration Options

Create a `.env` file in the project root with the following variables:

### Embedding Provider Selection

```env
# Options: "ollama" (default), "openai"
EMBEDDING_PROVIDER=ollama
```

### Database Configuration

```env
# ChromaDB collection name (default: qwen2.5-7b-instruct)
CHROMA_COLLECTION_NAME=qwen2.5-7b-instruct

# Document chunking settings
CHUNK_SIZE=800                    # Size of each text chunk
CHUNK_OVERLAP=80                  # Overlap between chunks
MAX_DOCUMENTS=                    # Limit number of documents (optional)

# Processing options
ENABLE_INCREMENTAL_UPDATES=true   # Skip existing documents
ENABLE_METADATA_ENRICHMENT=true   # Add extra metadata
BATCH_SIZE=100                    # Batch size for database operations
```

### RAG Query Configuration

```env
# LLM Provider for RAG queries
RAG_LLM_PROVIDER=ollama           # "ollama" or "openai"
OLLAMA_LLM_MODEL=qwen2.5:7b-instruct  # Ollama model for queries
OPENAI_LLM_MODEL=gpt-4o-mini      # OpenAI model for queries

# Retrieval settings
RETRIEVAL_K=5                     # Number of documents to retrieve
SIMILARITY_THRESHOLD=0.0          # Minimum similarity score
MAX_CONTEXT_LENGTH=4000           # Maximum context length

# LLM parameters
LLM_TEMPERATURE=0.0               # Model temperature (0.0-2.0)
LLM_MAX_TOKENS=                   # Maximum tokens (optional)

# Query processing
ENABLE_CONTEXT_FILTERING=true     # Filter low-similarity documents
ENABLE_RESPONSE_CLEANING=true     # Clean model artifacts
QUERY_OUTPUT_FORMAT=json          # Default output format
INCLUDE_DEBUG_INFO=false          # Include debug information
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

### Example 4: Performance Optimized Setup
```env
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=bge-m3:567m
CHUNK_SIZE=1200
CHUNK_OVERLAP=120
BATCH_SIZE=200
ENABLE_INCREMENTAL_UPDATES=true
```

### Example 5: Development/Testing Setup
```env
EMBEDDING_PROVIDER=ollama
CHUNK_SIZE=400
CHUNK_OVERLAP=40
MAX_DOCUMENTS=50
ENABLE_METADATA_ENRICHMENT=true
```

## 🎯 Benefits of New Architecture

### For Embeddings
- **Flexibility**: Switch between local and cloud embeddings easily
- **Error Handling**: Automatic health checks and clear error messages
- **Configuration**: Environment-based configuration for different environments
- **Logging**: Better visibility into what's happening
- **Validation**: Automatic model availability checks

### For Database Setup
- **Dynamic Path Resolution**: Works from any directory (scripts/ or project root)
- **Intelligent Processing**: Automatic duplicate detection and incremental updates
- **Configurable Chunking**: Adjust chunk size and overlap via environment variables
- **Batch Processing**: Handle large document sets efficiently
- **Rich Metadata**: Automatic metadata enrichment with processing statistics
- **Comprehensive Logging**: Detailed progress tracking and error reporting
- **CLI Enhancement**: Rich command-line interface with multiple options
