# Evaluate of RAG responses using RAGAS metrics & Pytest

Local RAG Setup for Testing Purposes

## Description

This project explores how to set up a local Retrieval-Augmented Generation (RAG) system for testing purposes. RAG is a powerful technique that combines retrieval-based and generation-based models to improve the quality and relevance of generated text.

## Quick Start

If you want to get started quickly:

1. **Install Ollama** and pull required models:
   ```bash
   ollama pull bge-m3:567m
   ollama pull qwen2.5:7b-instruct
   ```

2. **Set up the project**:
   ```bash
   git clone <your-repo-url>
   cd personalRag
   python -m venv local-ragas-env
   source local-ragas-env/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up database and query**:
   ```bash
   cd scripts
   python database_setup.py
   python query.py "What are the main types of pizza?"
   ```

## Table of Contents

- [Quick Start](#quick-start)
- [OS requirements](#os-requirements)
- [Installation](#installation)
- [Ollama Setup](#ollama-setup)
- [Environment Variables](#environment-variables)
- [How the RAG System Works](#how-the-rag-system-works)
- [Architecture Improvements](#️-architecture-improvements)
- [Database Setup](#database-setup)
- [RAG System Usage](#rag-system-usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## OS requirements

- **Python 3.10+** (tested with Python 3.13)
- **pip** (Python package manager)
- **git** (version control)
- **Ollama** (for local AI models)

### Virtual Environment Setup

Create and manage a virtual environment for this project:

**For macOS/Linux:**
```bash
# Create virtual environment
python -m venv local-ragas-env

# Activate virtual environment
source local-ragas-env/bin/activate

# Deactivate virtual environment (when done)
deactivate
```

**For Windows:**
```bash
# Create virtual environment
python -m venv local-ragas-env

# Activate virtual environment
.\local-ragas-env\Scripts\activate.bat

# Deactivate virtual environment (when done)
deactivate
```

## Installation

1. Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd personalRag
    ```

2. Create and activate virtual environment:
    ```bash
    python -m venv local-ragas-env
    source local-ragas-env/bin/activate  # On Windows: .\local-ragas-env\Scripts\activate.bat
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Ollama Setup

This project uses Ollama for local embeddings and language model inference. Follow these steps to set up Ollama:

1. **Install Ollama**: 
   - Visit [https://ollama.ai](https://ollama.ai) and download Ollama for your operating system
   - Follow the installation instructions for your platform

2. **Install required models**:
   ```bash
   # Install the embedding model (BGE-M3)
   ollama pull bge-m3:567m
   
   # Install the language model (Qwen2.5 - used by this project)
   ollama pull qwen2.5:7b-instruct
   ```

3. **Verify Ollama is running**:
   ```bash
   # Check if Ollama service is running (should show installed models)
   ollama list
   
   # If Ollama is not running, start it
   ollama serve
   ```

## Environment Variables

The system now supports flexible configuration through environment variables. Create a `.env` file in the project root for customization:

### 🔧 Basic Configuration (Optional)

```env
# Embedding provider: "ollama" (default) or "openai"
EMBEDDING_PROVIDER=ollama

# Ollama configuration (default values shown)
OLLAMA_EMBEDDING_MODEL=bge-m3:567m
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI configuration (only needed if using OpenAI)
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### 🎯 Benefits of New Architecture

- **🛡️ Error Handling**: Automatic health checks and clear error messages
- **⚙️ Flexible Configuration**: Easy switching between local and cloud embeddings  
- **📊 Better Logging**: Visibility into what models are being used
- **🔄 Environment-Based**: Different configurations for development/testing/production

### 📚 Full Configuration Guide

See `ENVIRONMENT_VARIABLES.md` for complete configuration options and examples.

**Note**: No configuration is required for default Ollama setup - it works out of the box!

## How the RAG System Works

This diagram shows the complete RAG (Retrieval-Augmented Generation) workflow using your current setup with **BGE-M3 embeddings** and **Qwen2.5 LLM**, both running locally via Ollama.

```mermaid
graph TD
    A["🙋 User asks a question<br/>python query.py 'What is pizza?'"] --> B["📄 query.py script starts"]
    
    B --> C["🗄️ Connect to ChromaDB<br/>(Local Vector Database)"]
    C --> C1["📂 Find database folder<br/>scripts/: ../chroma<br/>root/: chroma"]
    C --> C2["📋 Use collection: qwen2.5-7b-instruct<br/>(Named after LLM model)"]
    
    B --> D["🧠 Load Embedding Model<br/>(Converts text to numbers)"]
    D --> D1["🤖 BGE-M3 Model via Ollama<br/>Runs locally on localhost:11434<br/>Creates 1024-dimensional vectors"]
    
    D1 --> E["🔢 Convert user question to vector<br/>Example: 'What is pizza?' → [0.1, -0.5, 0.3, ...]<br/>(1024 numbers representing meaning)"]
    
    E --> F["🔍 Vector Similarity Search<br/>Compare question vector with stored document vectors<br/>Find top 5 most similar chunks"]
    
    F -->|Found documents| G["📝 Collect relevant text chunks<br/>Combine all found content into context"]
    F -->|No documents found| Z1["❌ No relevant info found<br/>Answer: 'I don't know based on context'"]
    
    G --> H["📋 Create AI prompt<br/>Context: [retrieved chunks]<br/>Question: [user question]"]
    
    H --> I["🤖 Send to Qwen2.5 Model<br/>(7B parameter LLM via Ollama)<br/>Local AI that reads and responds"]
    
    I --> J["💭 AI processes context + question<br/>Generates response with <think> tags<br/>(Internal reasoning)"]
    
    J --> K["🧹 Clean response<br/>Remove <think>...</think> blocks<br/>Keep only final answer"]
    
    K --> L["📊 Package response<br/>JSON: {answer: '...', docs: [...]}"]
    L --> M["✅ Display result to user"]

    %% Database Setup Process (Background)
    N["📚 Database Setup Process<br/>(database_setup.py)"] --> N1["📄 Load PDFs from data/ folder<br/>Using PyPDFDirectoryLoader"]
    N1 --> N2["✂️ Split into chunks<br/>800 characters each, 80 overlap<br/>Creates manageable pieces"]
    N2 --> N3["🔢 Convert chunks to vectors<br/>Using same BGE-M3 model<br/>Store in ChromaDB"]
    N3 --> N4["💾 Save to ChromaDB<br/>Each chunk gets unique ID<br/>Ready for similarity search"]

    %% Key Concepts for QA Engineers
    subgraph concepts["🎓 Key Concepts for QA Engineers"]
        K1["🔤 Embeddings:<br/>Numbers that represent text meaning<br/>Similar texts = similar numbers"]
        K2["🗃️ ChromaDB:<br/>Database optimized for vector search<br/>Like Google for your documents"]
        K3["🎯 Similarity Search:<br/>Find documents with similar meaning<br/>Not exact word matches"]
        K4["🤖 LLM (Large Language Model):<br/>AI that reads context and answers<br/>Like ChatGPT but running locally"]
        K5["📊 Vector Dimensions:<br/>BGE-M3 creates 1024 numbers per text<br/>More dimensions = better understanding"]
    end

    %% Common Issues
    C -->|Database empty| Z2["⚠️ Database not set up<br/>Run: python database_setup.py"]
    D1 -.->|Model mismatch| Z3["⚠️ Different embedding models<br/>Must use BGE-M3 for both indexing & search"]
    F -.->|Bad PDF quality| Z4["⚠️ PDF text extraction failed<br/>Check if PDF contains readable text"]
    I -.->|Ollama not running| Z5["⚠️ Cannot reach Ollama<br/>Start: ollama serve"]

    %% Styling
    classDef errorNode fill:#ffebee,stroke:#f44336,stroke-width:2px,color:#000
    classDef processNode fill:#e8f5e8,stroke:#4caf50,stroke-width:2px,color:#000
    classDef dataNode fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000
    classDef userNode fill:#e3f2fd,stroke:#2196f3,stroke-width:3px,color:#000
    classDef conceptNode fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#000
    classDef setupNode fill:#e0f2f1,stroke:#009688,stroke-width:2px,color:#000
    
    class Z1,Z2,Z3,Z4,Z5 errorNode
    class B,D,F,H,I,J,K processNode
    class C,C1,C2,D1,E,G,L,M dataNode
    class A,M userNode
    class K1,K2,K3,K4,K5 conceptNode
    class N,N1,N2,N3,N4 setupNode
```

### 🎓 Understanding RAG for QA Engineers

**RAG (Retrieval-Augmented Generation)** combines two key technologies:

1. **🔍 Retrieval**: Finding relevant information from your documents
2. **💭 Generation**: Using AI to create answers based on that information

**Key Components:**

- **📊 Embeddings**: Convert text into numbers (vectors) that capture meaning. Similar concepts have similar numbers.
- **🗃️ ChromaDB**: A specialized database for storing and searching these vectors efficiently.
- **🎯 Similarity Search**: Instead of exact keyword matching, finds documents with similar *meaning*.
- **🤖 LLM**: The AI that reads the retrieved context and generates human-like answers.

**Why This Approach Works:**
- Your AI answers are grounded in your specific documents
- Reduces "hallucination" (making up facts)
- Works with any document type (PDFs, text files, etc.)
- Runs completely locally (no data sent to external APIs)

### 🔄 **Simple Flow Summary:**
1. **User asks** → 2. **Find similar docs** → 3. **Ask AI with context** → 4. **Return clean answer**

## 🏗️ Architecture Improvements

The embedding system has been enhanced with enterprise-grade features while maintaining 100% backward compatibility:

### ✅ **Enhanced Features**

#### **🛡️ Robust Error Handling**
- **Automatic health checks** for Ollama server connectivity
- **Model validation** - confirms required models are installed
- **Clear error messages** with specific resolution steps
- **Graceful fallbacks** and timeout handling

#### **⚙️ Flexible Configuration**
- **Environment-based** configuration for different deployments
- **Multiple providers** - switch between Ollama (local) and OpenAI (cloud)
- **Runtime model selection** without code changes
- **Zero-config defaults** - works out of the box

#### **📊 Observability & Debugging**
- **Structured logging** with appropriate log levels
- **Model usage visibility** - see which models are active
- **Performance monitoring** - track embedding generation
- **Debug-friendly** error traces

#### **🔧 Developer Experience**
- **Type hints** for better IDE support
- **Comprehensive documentation** with examples
- **Clean separation** of concerns
- **Easy testing** and mocking capabilities

### 🎯 **Backward Compatibility Guaranteed**

```python
# ✅ Original code continues to work unchanged
from scripts.get_embedding_function import get_embedding_function
embeddings = get_embedding_function()  # Still works exactly the same

# 🚀 New capabilities available when needed
from scripts.get_embedding_function import get_embedding_function_new
embeddings = get_embedding_function_new(provider="openai")  # Switch providers easily
```

### 🔬 **For QA Engineers**

The improved architecture provides:
- **Better error diagnostics** when tests fail
- **Environment isolation** for different test scenarios  
- **Configurable backends** for testing various models
- **Health check endpoints** for monitoring test infrastructure

## RAGAS Evaluation Metrics

RAGAS helps us measure how good our RAG system is by testing 4 key areas. Think of it like a report card for your AI:

```mermaid
graph TD
    A["🧪 RAGAS Evaluation Process"] --> B["📊 4 Quality Metrics"]
    
    B --> C1["🎯 Context Precision<br/>Are the retrieved docs relevant?"]
    B --> C2["🔍 Context Recall<br/>Did we find all the important info?"]
    B --> C3["✅ Faithfulness<br/>Is the answer truthful to the docs?"]
    B --> C4["📝 Response Relevancy<br/>Does the answer match the question?"]
    
    %% Context Precision Flow
    C1 --> CP1["📥 Input: Question + Retrieved Docs"]
    CP1 --> CP2["🤖 AI judges: Which docs help answer?"]
    CP2 --> CP3["📊 Score: Relevant docs ÷ Total docs"]
    CP3 --> CP4["🟢 High Score (0.8-1.0): Most docs are useful<br/>🟡 Medium Score (0.5-0.7): Some docs help<br/>🔴 Low Score (0.0-0.4): Many irrelevant docs"]
    
    %% Context Recall Flow  
    C2 --> CR1["📥 Input: Question + Retrieved Docs + Ground Truth"]
    CR1 --> CR2["🤖 AI checks: Did we find all key info?"]
    CR2 --> CR3["📊 Score: Found key points ÷ Total key points"]
    CR3 --> CR4["🟢 High Score (0.8-1.0): Found most important info<br/>🟡 Medium Score (0.5-0.7): Missing some details<br/>🔴 Low Score (0.0-0.4): Missed lots of key info"]
    
    %% Faithfulness Flow
    C3 --> F1["📥 Input: Question + Retrieved Docs + AI Answer"]
    F1 --> F2["🤖 AI checks: Are facts in answer from docs?"]
    F2 --> F3["📊 Score: Verified claims ÷ Total claims"]
    F3 --> F4["🟢 High Score (0.8-1.0): Answer sticks to facts<br/>🟡 Medium Score (0.5-0.7): Some unsupported claims<br/>🔴 Low Score (0.0-0.4): Answer makes things up"]
    
    %% Response Relevancy Flow
    C4 --> RR1["📥 Input: Question + AI Answer"]
    RR1 --> RR2["🤖 AI checks: Does answer address question?"]
    RR2 --> RR3["📊 Score: Relevant parts ÷ Total answer"]
    RR3 --> RR4["🟢 High Score (0.8-1.0): Answer directly addresses question<br/>🟡 Medium Score (0.5-0.7): Partially answers question<br/>🔴 Low Score (0.0-0.4): Answer is off-topic"]

    %% Styling
    classDef metricNode fill:#e8f5e8,stroke:#4caf50,stroke-width:2px,color:#000
    classDef processNode fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000
    classDef scoreNode fill:#e3f2fd,stroke:#2196f3,stroke-width:2px,color:#000
    classDef resultNode fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#000
    
    class C1,C2,C3,C4 metricNode
    class CP1,CR1,F1,RR1,CP2,CR2,F2,RR2 processNode
    class CP3,CR3,F3,RR3 scoreNode
    class CP4,CR4,F4,RR4 resultNode
```

### 📊 **What Each Metric Tests:**

| Metric | What it asks | What you need | Good Score means | Bad Score means |
|--------|-------------|---------------|------------------|-----------------|
| **🎯 Context Precision** | "Are these docs helpful?" | Question + Retrieved docs | Search found useful info | Search returned junk |
| **🔍 Context Recall** | "Did we find everything?" | Question + Docs + Truth | Found most key details | Missed important info |
| **✅ Faithfulness** | "Is the answer honest?" | Question + Docs + Answer | AI stuck to the facts | AI made things up |
| **📝 Response Relevancy** | "Does this answer the question?" | Question + Answer | Direct, focused answer | Off-topic response |

### 🎯 **Quick Interpretation Guide:**

- **Score 0.8-1.0**: 🟢 **Excellent** - System working great!
- **Score 0.5-0.7**: 🟡 **Needs work** - Some issues to fix
- **Score 0.0-0.4**: 🔴 **Poor** - Major problems, needs attention

### 🔧 **Common Issues & Solutions:**

| Low Score In | Probable Cause | How to Fix |
|-------------|----------------|------------|
| **Context Precision** | Search returning irrelevant docs | Improve embeddings, adjust chunk size |
| **Context Recall** | Missing important documents | Add more docs, improve search parameters |
| **Faithfulness** | AI hallucinating facts | Better prompts, check retrieved context |
| **Response Relevancy** | AI going off-topic | Improve prompt template, tune AI model |

## Database Setup

Before using the RAG system, you need to set up the vector database with your documents:

1. **Place your PDF documents** in the `data/` directory

2. **Set up the database** (run from the project root):
   ```bash
   # Activate virtual environment
   source local-ragas-env/bin/activate
   
   # Navigate to scripts directory
   cd scripts
   
   # Create/populate the vector database
   python database_setup.py
   
   # Optional: Reset the database if you want to start fresh
   python database_setup.py --reset
   ```

3. **Verify database creation**:
   - Check that a `chroma/` directory was created in the project root
   - The database will contain embeddings of your PDF documents

## RAG System Usage

Once the database is set up, you can query your documents:

### Basic Query Usage

```bash
# Navigate to scripts directory (if not already there)
cd scripts

# Activate virtual environment
source ../local-ragas-env/bin/activate

# Run a query
python query.py "What are the main types of pizza?"

# Example with a specific question
python query.py "Where was the largest pizza ever made?"
```

### Query Response Format

The system returns a JSON response with:
- `answer`: The AI-generated response based on retrieved documents
- `retrieved_docs`: List of relevant document chunks with metadata

Example response:
```json
{
  "answer": "The largest pizza ever made measured over 13,000 square feet in Rome, Italy, in 2012.",
  "retrieved_docs": [
    {
      "file_name": "data/pizza_rag_test_expanded.pdf:6:0",
      "page_content": "Fun Facts About Pizza\nThe largest pizza ever made..."
    }
  ]
}
```

## Testing

The project includes RAGAS evaluation tests for measuring RAG system performance:

> **Note**: Tests require an OpenAI API key for RAGAS evaluation metrics. Set `OPENAI_API_KEY` in your environment or `.env` file.

1. **Set the PYTHONPATH environment variable**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/your/personalRag"
   # Example: export PYTHONPATH="${PYTHONPATH}:/Users/joanesquivel/Desktop/personalRag"
   ```

2. **Run specific tests** (from project root):
   ```bash
   # Activate virtual environment
   source local-ragas-env/bin/activate
   
   # Run individual tests (-s flag shows print statements)
   pytest tests/test_response_relevancy.py -s
   pytest tests/test_context_precision.py -s
   pytest tests/test_context_recall.py -s
   pytest tests/test_faithfulness.py -s
   
   # Run all tests
   pytest tests/ -s
   ```

3. **Available test metrics**:
   - **Response Relevancy**: Measures how relevant the answer is to the question
   - **Context Precision**: Evaluates if retrieved contexts are relevant to the question
   - **Context Recall**: Checks if the retrieved context contains information needed to answer
   - **Faithfulness**: Measures if the answer is faithful to the retrieved context


## Contributing

If you want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push to the branch.
5. Create a pull request.


## License

This project is licensed under the MIT License, which means you can use, copy, modify, and distribute the software, but you must include the original license and copyright notice in any copies or substantial portions of the software. See the LICENSE file for more details.