"""
RAG Query Module

This module provides enterprise-grade RAG (Retrieval-Augmented Generation) querying capabilities
with advanced configuration, error handling, and observability features.

Key Features:
- Flexible LLM provider configuration (Ollama, OpenAI)
- Advanced retrieval strategies with configurable parameters
- Comprehensive error handling and logging
- Rich response formatting and metadata
- Performance monitoring and metrics
- Environment-based configuration
- Multiple output formats (JSON, structured)
"""

import argparse
import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema.document import Document

try:
    from get_embedding_function import get_embedding_function
except ImportError:
    from scripts.get_embedding_function import get_embedding_function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"


class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    STRUCTURED = "structured"
    TEXT = "text"


@dataclass
class QueryConfig:
    """Configuration class for RAG queries"""
    
    # Database Configuration
    chroma_path: str = None
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "qwen2.5-7b-instruct")
    
    # LLM Configuration
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "ollama")
    
    # Ollama LLM Settings
    ollama_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # OpenAI LLM Settings
    openai_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # Retrieval Configuration
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    
    # Model Parameters
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: Optional[int] = None if not os.getenv("LLM_MAX_TOKENS") else int(os.getenv("LLM_MAX_TOKENS"))
    
    # Processing Options
    enable_context_filtering: bool = os.getenv("ENABLE_CONTEXT_FILTERING", "true").lower() == "true"
    enable_response_cleaning: bool = os.getenv("ENABLE_RESPONSE_CLEANING", "true").lower() == "true"
    enable_metadata_enrichment: bool = os.getenv("ENABLE_METADATA_ENRICHMENT", "true").lower() == "true"
    
    # Output Configuration
    output_format: str = os.getenv("QUERY_OUTPUT_FORMAT", "json")
    include_debug_info: bool = os.getenv("INCLUDE_DEBUG_INFO", "false").lower() == "true"
    
    def __post_init__(self):
        """Initialize dynamic paths and validate configuration"""
        if self.chroma_path is None:
            self.chroma_path = self._resolve_chroma_path()
        
        # Validate configuration
        self._validate_config()
    
    def _resolve_chroma_path(self) -> str:
        """Resolve ChromaDB path based on current working directory"""
        current_dir = Path.cwd()
        if current_dir.name == 'scripts':
            return str(current_dir.parent / "chroma")
        else:
            return str(current_dir / "chroma")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.llm_provider == LLMProvider.OPENAI.value and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI LLM provider")
        
        if self.retrieval_k <= 0:
            raise ValueError(f"Retrieval k must be positive, got: {self.retrieval_k}")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"Temperature must be between 0 and 2, got: {self.temperature}")


@dataclass
class QueryResult:
    """Structured result from RAG query"""
    
    # Core Results
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    
    # Metadata
    query: str
    timestamp: str
    processing_time: float
    
    # Configuration Used
    config_used: Dict[str, Any]
    
    # Performance Metrics
    retrieval_count: int
    context_length: int
    response_length: int
    
    # Debug Information (optional)
    debug_info: Optional[Dict[str, Any]] = None


class ResponseProcessor:
    """Handles response cleaning and formatting"""
    
    @staticmethod
    def clean_think_process(text: str) -> str:
        """
        Remove <think>...</think> reasoning process from model output
        
        Some models (like deepseek-r1) include reasoning in <think> tags
        which should be excluded from the final answer.
        """
        # Remove everything between <think> and </think> tags (including the tags)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        cleaned_text = cleaned_text.strip()
        
        # Remove multiple consecutive newlines
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        return cleaned_text
    
    @staticmethod
    def clean_response_artifacts(text: str) -> str:
        """Remove common response artifacts and formatting issues"""
        # Remove markdown code block markers if they wrap the entire response
        text = re.sub(r'^```[\w]*\n|```$', '', text.strip(), flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        
        # Clean up bullet points and formatting
        text = re.sub(r'\n-\s*\n', '\n- ', text)
        
        return text.strip()


class LLMManager:
    """Manages different LLM providers and configurations"""
    
    def __init__(self, config: QueryConfig):
        self.config = config
        self._model = None
    
    @property
    def model(self):
        """Lazy-loaded LLM model instance"""
        if self._model is None:
            self._model = self._create_model()
        return self._model
    
    def _create_model(self):
        """Create LLM model based on configuration"""
        try:
            if self.config.llm_provider == LLMProvider.OLLAMA.value:
                return self._create_ollama_model()
            elif self.config.llm_provider == LLMProvider.OPENAI.value:
                return self._create_openai_model()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to create LLM model: {e}")
            raise
    
    def _create_ollama_model(self):
        """Create Ollama model instance"""
        logger.info(f"Initializing Ollama model: {self.config.ollama_model}")
        
        model_kwargs = {
            "temperature": self.config.temperature,
            "model": self.config.ollama_model,
            "base_url": self.config.ollama_base_url,
        }
        
        if self.config.max_tokens:
            model_kwargs["num_predict"] = self.config.max_tokens
        
        return ChatOllama(**model_kwargs)
    
    def _create_openai_model(self):
        """Create OpenAI model instance"""
        logger.info(f"Initializing OpenAI model: {self.config.openai_model}")
        
        model_kwargs = {
            "temperature": self.config.temperature,
            "model": self.config.openai_model,
            "openai_api_key": self.config.openai_api_key,
        }
        
        if self.config.max_tokens:
            model_kwargs["max_tokens"] = self.config.max_tokens
        
        return ChatOpenAI(**model_kwargs)


class VectorStoreManager:
    """Manages vector database operations"""
    
    def __init__(self, config: QueryConfig):
        self.config = config
        self._db = None
    
    @property
    def db(self) -> Chroma:
        """Lazy-loaded ChromaDB instance"""
        if self._db is None:
            self._db = self._create_db_connection()
        return self._db
    
    def _create_db_connection(self) -> Chroma:
        """Create ChromaDB connection"""
        try:
            db_path = Path(self.config.chroma_path)
            if not db_path.exists():
                raise FileNotFoundError(
                    f"ChromaDB not found at {self.config.chroma_path}. "
                    f"Please run database_setup.py first."
                )
            
            embedding_function = get_embedding_function()
            
            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=embedding_function,
                collection_name=self.config.collection_name
            )
            
            logger.info(f"Connected to ChromaDB at {self.config.chroma_path}")
            return db
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise ConnectionError(f"Cannot connect to ChromaDB: {e}")
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with optional filtering
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of (document, score) tuples
        """
        k = k or self.config.retrieval_k
        score_threshold = score_threshold or self.config.similarity_threshold
        
        try:
            # Get similarity search results
            results = self.db.similarity_search_with_score(query, k=k)
            
            # Apply score filtering if enabled
            if self.config.enable_context_filtering and score_threshold > 0:
                filtered_results = [
                    (doc, score) for doc, score in results 
                    if score <= score_threshold  # Lower score = higher similarity in some embeddings
                ]
                
                if filtered_results:
                    results = filtered_results
                    logger.info(f"Filtered {len(results)} documents based on similarity threshold")
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise


class PromptManager:
    """Manages prompt templates and formatting"""
    
    # Default prompt template
    DEFAULT_TEMPLATE = """Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}"""
    
    # Advanced prompt template with instructions
    ADVANCED_TEMPLATE = """You are a helpful AI assistant. Answer the question based ONLY on the provided context. 

INSTRUCTIONS:
- Use only information from the context provided below
- If the context doesn't contain enough information to answer the question, say so
- Be specific and cite relevant details from the context
- Do not add information not present in the context

CONTEXT:
{context}

---

QUESTION: {question}

ANSWER:"""
    
    def __init__(self, config: QueryConfig):
        self.config = config
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get configured prompt template"""
        template = os.getenv("CUSTOM_PROMPT_TEMPLATE", self.DEFAULT_TEMPLATE)
        
        # Use advanced template if context filtering is enabled
        if self.config.enable_context_filtering:
            template = self.ADVANCED_TEMPLATE
        
        return ChatPromptTemplate.from_template(template)
    
    def format_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            content = doc.page_content.strip()
            
            # Add document metadata if enrichment is enabled
            if self.config.enable_metadata_enrichment:
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"Document {i} (Source: {source}):\n{content}")
            else:
                context_parts.append(content)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length] + "\n\n[Context truncated due to length...]"
            logger.warning(f"Context truncated to {self.config.max_context_length} characters")
        
        return context


class RAGQueryEngine:
    """Main RAG query engine orchestrating all components"""
    
    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or QueryConfig()
        self.llm_manager = LLMManager(self.config)
        self.vector_store = VectorStoreManager(self.config)
        self.prompt_manager = PromptManager(self.config)
        self.response_processor = ResponseProcessor()
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'last_query_time': None
        }
    
    def query(
        self, 
        query_text: str,
        **kwargs
    ) -> QueryResult:
        """
        Execute RAG query with comprehensive error handling and logging
        
        Args:
            query_text: The question to ask
            **kwargs: Override configuration parameters
            
        Returns:
            QueryResult with answer and metadata
            
        Raises:
            ValueError: If query is invalid
            ConnectionError: If database/LLM is not accessible
            RuntimeError: If query processing fails
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing query: {query_text[:100]}...")
            
            # Step 1: Retrieve relevant documents
            documents = self.vector_store.similarity_search(
                query_text,
                k=kwargs.get('k'),
                score_threshold=kwargs.get('score_threshold')
            )
            
            if not documents:
                logger.warning("No documents retrieved for query")
                return self._create_no_results_response(query_text, start_time)
            
            # Step 2: Format context
            context = self.prompt_manager.format_context(documents)
            
            # Step 3: Generate prompt
            prompt_template = self.prompt_manager.get_prompt_template()
            messages = prompt_template.format_messages(
                context=context, 
                question=query_text
            )
            
            # Step 4: Get LLM response
            logger.debug(f"Sending prompt to {self.config.llm_provider} model")
            response = self.llm_manager.model.invoke(messages)
            response_text = response.content.strip()
            
            # Step 5: Process response
            if self.config.enable_response_cleaning:
                response_text = self.response_processor.clean_think_process(response_text)
                response_text = self.response_processor.clean_response_artifacts(response_text)
            
            # Step 6: Build result
            processing_time = (datetime.now() - start_time).total_seconds()
            result = self._build_query_result(
                query_text, response_text, documents, 
                processing_time, context
            )
            
            # Update stats
            self._update_query_stats(processing_time)
            
            logger.info(f"Query completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query failed after {processing_time:.2f}s: {e}")
            raise RuntimeError(f"Query processing failed: {e}") from e
    
    def _create_no_results_response(self, query_text: str, start_time: datetime) -> QueryResult:
        """Create response when no documents are retrieved"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResult(
            answer="I don't have enough information in the provided context to answer your question.",
            retrieved_docs=[],
            query=query_text,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            config_used=asdict(self.config),
            retrieval_count=0,
            context_length=0,
            response_length=0,
            debug_info={"reason": "no_documents_retrieved"} if self.config.include_debug_info else None
        )
    
    def _build_query_result(
        self,
        query_text: str,
        response_text: str,
        documents: List[Tuple[Document, float]],
        processing_time: float,
        context: str
    ) -> QueryResult:
        """Build comprehensive query result"""
        
        # Build retrieved docs list
        retrieved_docs = []
        for doc, score in documents:
            doc_info = {
                "file_name": doc.metadata.get("id", "Unknown File Name"),
                "page_content": doc.page_content
            }
            
            # Add enriched metadata if enabled
            if self.config.enable_metadata_enrichment:
                doc_info.update({
                    "similarity_score": float(score),
                    "content_length": len(doc.page_content),
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown")
                })
            
            retrieved_docs.append(doc_info)
        
        # Prepare debug info
        debug_info = None
        if self.config.include_debug_info:
            debug_info = {
                "context_used": context,
                "model_provider": self.config.llm_provider,
                "model_name": self.config.ollama_model if self.config.llm_provider == "ollama" else self.config.openai_model,
                "retrieval_params": {
                    "k": self.config.retrieval_k,
                    "similarity_threshold": self.config.similarity_threshold
                },
                "response_processing": {
                    "cleaning_enabled": self.config.enable_response_cleaning,
                    "filtering_enabled": self.config.enable_context_filtering
                }
            }
        
        return QueryResult(
            answer=response_text,
            retrieved_docs=retrieved_docs,
            query=query_text,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            config_used=asdict(self.config),
            retrieval_count=len(documents),
            context_length=len(context),
            response_length=len(response_text),
            debug_info=debug_info
        )
    
    def _update_query_stats(self, processing_time: float):
        """Update query performance statistics"""
        self.query_stats['total_queries'] += 1
        
        # Calculate running average
        current_avg = self.query_stats['avg_processing_time']
        total_queries = self.query_stats['total_queries']
        self.query_stats['avg_processing_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
        
        self.query_stats['last_query_time'] = processing_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        return self.query_stats.copy()


class OutputFormatter:
    """Handles different output formats"""
    
    @staticmethod
    def format_json(result: QueryResult, pretty: bool = True) -> str:
        """Format result as JSON"""
        data = {
            "answer": result.answer,
            "retrieved_docs": result.retrieved_docs
        }
        
        # Add metadata if debug mode
        if result.debug_info:
            data["metadata"] = {
                "query": result.query,
                "timestamp": result.timestamp,
                "processing_time": result.processing_time,
                "retrieval_count": result.retrieval_count,
                "debug_info": result.debug_info
            }
        
        return json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)
    
    @staticmethod
    def format_structured(result: QueryResult) -> str:
        """Format result in structured text format"""
        output = []
        output.append(f"Query: {result.query}")
        output.append(f"Timestamp: {result.timestamp}")
        output.append(f"Processing Time: {result.processing_time:.2f}s")
        output.append("")
        output.append("Answer:")
        output.append(result.answer)
        output.append("")
        output.append(f"Retrieved Documents ({result.retrieval_count}):")
        
        for i, doc in enumerate(result.retrieved_docs, 1):
            output.append(f"\n{i}. {doc['file_name']}")
            if 'similarity_score' in doc:
                output.append(f"   Similarity: {doc['similarity_score']:.4f}")
            output.append(f"   Content: {doc['page_content'][:200]}...")
        
        return "\n".join(output)
    
    @staticmethod
    def format_text(result: QueryResult) -> str:
        """Format result as plain text (answer only)"""
        return result.answer


# Backward compatibility functions
def query_rag(query_text: str) -> str:
    """
    Legacy function for backward compatibility
    
    Maintains the exact same interface as the original function
    for use in tests and existing code.
    """
    try:
        config = QueryConfig()
        engine = RAGQueryEngine(config)
        result = engine.query(query_text)
        
        # Format as original JSON structure
        output = {
            "answer": result.answer,
            "retrieved_docs": [
                {
                    "file_name": doc["file_name"],
                    "page_content": doc["page_content"]
                }
                for doc in result.retrieved_docs
            ]
        }
        
        return json.dumps(output, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Legacy query_rag failed: {e}")
        # Return error in original format
        return json.dumps({
            "answer": f"Error processing query: {e}",
            "retrieved_docs": []
        }, indent=2, ensure_ascii=False)


def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Query the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query.py "What are the main types of pizza?"
  python query.py "How is pizza made?" --format structured
  python query.py "Pizza ingredients" --debug --k 10
  python query.py "Pizza history" --provider openai
        """
    )
    
    parser.add_argument("query_text", type=str, help="The query text")
    parser.add_argument("--format", choices=["json", "structured", "text"], 
                       default="json", help="Output format")
    parser.add_argument("--debug", action="store_true", 
                       help="Include debug information")
    parser.add_argument("--k", type=int, help="Number of documents to retrieve")
    parser.add_argument("--provider", choices=["ollama", "openai"], 
                       help="LLM provider to use")
    parser.add_argument("--temperature", type=float, 
                       help="Model temperature (0.0-2.0)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create configuration with CLI overrides
        config = QueryConfig()
        
        if args.debug:
            config.include_debug_info = True
        if args.k:
            config.retrieval_k = args.k
        if args.provider:
            config.llm_provider = args.provider
        if args.temperature is not None:
            config.temperature = args.temperature
        
        # Execute query
        engine = RAGQueryEngine(config)
        result = engine.query(args.query_text)
        
        # Format output
        formatter = OutputFormatter()
        if args.format == "json":
            output = formatter.format_json(result)
        elif args.format == "structured":
            output = formatter.format_structured(result)
        else:  # text
            output = formatter.format_text(result)
        
        print(output)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
