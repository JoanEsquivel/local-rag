"""
Enhanced Test Configuration Module - conftest.py

This module provides enterprise-grade pytest fixtures and configuration for RAGAS evaluation testing.
Integrates with the existing project infrastructure while applying senior SDET best practices.

Architecture Features:
- Configuration management using existing project's environment system
- Flexible provider switching (Ollama/OpenAI) based on environment
- Robust error handling and fallback mechanisms
- Comprehensive test data management
- Performance monitoring and test observability
- Type safety with comprehensive type hints
- Resource management and cleanup
- Environment validation and health checks

Senior SDET Best Practices Applied:
- Fixture parameterization for multiple test scenarios
- Factory patterns for dynamic fixture creation
- Resource lifecycle management
- Test isolation and independence
- Comprehensive error reporting
- Environment validation
- Performance monitoring
- Memory management optimization
"""

import pytest
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Generator
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules using existing infrastructure
from scripts.get_embedding_function import get_embedding_function_new as get_embedding_function, EmbeddingConfig
from scripts.query import QueryConfig, LLMManager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure test-specific logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
test_logger = logging.getLogger('test_framework')


@dataclass
class TestConfiguration:
    """Centralized test configuration using project's environment system"""
    
    # Test Environment Settings
    test_environment: str = os.getenv("TEST_ENVIRONMENT", "development")
    enable_performance_monitoring: bool = os.getenv("ENABLE_TEST_PERFORMANCE_MONITORING", "true").lower() == "true"
    enable_verbose_logging: bool = os.getenv("ENABLE_TEST_VERBOSE_LOGGING", "false").lower() == "true"
    enable_legacy_logging: bool = os.getenv("ENABLE_LEGACY_LOGGING", "false").lower() == "true"
    logging_format: str = os.getenv("TEST_LOGGING_FORMAT", "enhanced")  # "enhanced", "legacy", "both"
    test_timeout: int = int(os.getenv("TEST_TIMEOUT", "300"))  # 5 minutes default
    
    # Provider Configuration (inheriting from project)
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    ragas_llm_provider: str = os.getenv("RAGAS_LLM_PROVIDER", os.getenv("RAG_LLM_PROVIDER", "ollama"))
    
    # Model Configuration
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m")
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Test-specific Model Settings
    test_temperature: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))  # Deterministic for testing
    test_max_tokens: Optional[int] = int(os.getenv("TEST_MAX_TOKENS", "1000")) if os.getenv("TEST_MAX_TOKENS") else None
    
    # Quality Thresholds
    minimum_faithfulness_score: float = float(os.getenv("TEST_MIN_FAITHFULNESS", "0.5"))
    minimum_relevancy_score: float = float(os.getenv("TEST_MIN_RELEVANCY", "0.5"))
    minimum_precision_score: float = float(os.getenv("TEST_MIN_PRECISION", "0.5"))
    minimum_recall_score: float = float(os.getenv("TEST_MIN_RECALL", "0.5"))
    
    def __post_init__(self):
        """Validate configuration and log settings"""
        if self.enable_verbose_logging:
            test_logger.info(f"Test Configuration: {asdict(self)}")
        
        # Validate required settings
        if self.ragas_llm_provider == "openai" and not self.openai_api_key:
            test_logger.warning("OpenAI provider selected but no API key found. Tests may fail.")


@pytest.fixture(scope="session")
def test_config() -> TestConfiguration:
    """
    Session-scoped test configuration fixture
    
    Returns:
        TestConfiguration: Centralized test configuration
    """
    return TestConfiguration()


@pytest.fixture(scope="session")
def environment_health_check(test_config: TestConfiguration) -> Dict[str, bool]:
    """
    Perform comprehensive environment health checks
    
    Args:
        test_config: Test configuration
        
    Returns:
        Dict[str, bool]: Health check results
    """
    health_status = {}
    
    # Check Ollama availability
    if test_config.embedding_provider == "ollama" or test_config.ragas_llm_provider == "ollama":
        try:
            import requests
            response = requests.get(f"{test_config.ollama_base_url}/api/tags", timeout=5)
            health_status["ollama_server"] = response.status_code == 200
        except Exception as e:
            test_logger.warning(f"Ollama health check failed: {e}")
            health_status["ollama_server"] = False
    
    # Check OpenAI availability
    if test_config.embedding_provider == "openai" or test_config.ragas_llm_provider == "openai":
        health_status["openai_api_key"] = bool(test_config.openai_api_key)
    
    # Check test data availability
    test_data_dir = Path(__file__).parent / "data"
    health_status["test_data"] = test_data_dir.exists() and any(test_data_dir.glob("*.json"))
    
    # Log health status
    test_logger.info(f"Environment Health Check: {health_status}")
    
    return health_status


class PerformanceMonitor:
    """Performance monitoring for test execution"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
    
    def start(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        test_logger.info(f"Performance monitoring started for: {self.test_name}")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics = {
                "test_name": self.test_name,
                "duration_seconds": duration,
                "timestamp": time.time()
            }
            test_logger.info(f"Test {self.test_name} completed in {duration:.2f} seconds")
        return self.metrics


@pytest.fixture
def performance_monitor(request) -> Generator[PerformanceMonitor, None, None]:
    """
    Performance monitoring fixture for individual tests
    
    Args:
        request: Pytest request object
        
    Yields:
        PerformanceMonitor: Performance monitoring instance
    """
    monitor = PerformanceMonitor(request.node.name)
    monitor.start()
    yield monitor
    metrics = monitor.stop()
    
    # Store metrics for potential test reporting
    if not hasattr(request.config, "_performance_metrics"):
        request.config._performance_metrics = []
    request.config._performance_metrics.append(metrics)


@pytest.fixture(scope="session")
def langchain_llm_ragas_wrapper(test_config: TestConfiguration, environment_health_check: Dict[str, bool]) -> LangchainLLMWrapper:
    """
    Enhanced LLM wrapper for RAGAS evaluation with provider flexibility
    
    Args:
        test_config: Test configuration
        environment_health_check: Environment health status
        
    Returns:
        LangchainLLMWrapper: Configured LLM wrapper for RAGAS
        
    Raises:
        pytest.skip: If required services are unavailable
    """
    provider = test_config.ragas_llm_provider
    test_logger.info(f"Initializing RAGAS LLM wrapper with provider: {provider}")
    
    try:
        if provider == "ollama":
            if not environment_health_check.get("ollama_server", False):
                pytest.skip("Ollama server not available for RAGAS testing")
            
            llm = ChatOllama(
                model=test_config.ollama_llm_model,
                temperature=test_config.test_temperature,
                base_url=test_config.ollama_base_url,
                num_predict=test_config.test_max_tokens
            )
            test_logger.info(f"Created Ollama LLM: {test_config.ollama_llm_model}")
            
        elif provider == "openai":
            if not environment_health_check.get("openai_api_key", False):
                pytest.skip("OpenAI API key not available for RAGAS testing")
            
            llm_kwargs = {
                "model": test_config.openai_llm_model,
                "temperature": test_config.test_temperature,
                "openai_api_key": test_config.openai_api_key
            }
            if test_config.test_max_tokens:
                llm_kwargs["max_tokens"] = test_config.test_max_tokens
            
            llm = ChatOpenAI(**llm_kwargs)
            test_logger.info(f"Created OpenAI LLM: {test_config.openai_llm_model}")
            
        else:
            raise ValueError(f"Unsupported RAGAS LLM provider: {provider}")
        
        return LangchainLLMWrapper(llm)
        
    except Exception as e:
        test_logger.error(f"Failed to create RAGAS LLM wrapper: {e}")
        pytest.skip(f"Could not initialize RAGAS LLM: {e}")


@pytest.fixture(scope="session")
def embeddings_provider(test_config: TestConfiguration, environment_health_check: Dict[str, bool]):
    """
    Enhanced embeddings provider using project's infrastructure
    
    Args:
        test_config: Test configuration
        environment_health_check: Environment health status
        
    Returns:
        Union[OllamaEmbeddings, OpenAIEmbeddings]: Configured embeddings provider
        
    Raises:
        pytest.skip: If required services are unavailable
    """
    provider = test_config.embedding_provider
    test_logger.info(f"Initializing embeddings provider: {provider}")
    
    try:
        if provider == "ollama":
            if not environment_health_check.get("ollama_server", False):
                pytest.skip("Ollama server not available for embeddings")
            
            embeddings = get_embedding_function(
                provider="ollama",
                model=test_config.ollama_embedding_model,
                base_url=test_config.ollama_base_url
            )
            test_logger.info(f"Created Ollama embeddings: {test_config.ollama_embedding_model}")
            
        elif provider == "openai":
            if not environment_health_check.get("openai_api_key", False):
                pytest.skip("OpenAI API key not available for embeddings")
            
            embeddings = get_embedding_function(
                provider="openai",
                model=test_config.openai_embedding_model,
                api_key=test_config.openai_api_key
            )
            test_logger.info(f"Created OpenAI embeddings: {test_config.openai_embedding_model}")
            
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")
        
        return embeddings
        
    except Exception as e:
        test_logger.error(f"Failed to create embeddings provider: {e}")
        pytest.skip(f"Could not initialize embeddings: {e}")


# Backward compatibility alias
@pytest.fixture(scope="session")
def get_embeddings(embeddings_provider):
    """Backward compatibility fixture for existing tests"""
    return embeddings_provider


class TestDataManager:
    """Enhanced test data management with validation and caching"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._cache: Dict[str, Dict] = {}
        self._validate_test_data()
    
    def _validate_test_data(self):
        """Validate test data structure and content"""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Test data directory not found: {self.base_path}")
        
        json_files = list(self.base_path.glob("*.json"))
        if not json_files:
            test_logger.warning(f"No JSON test data files found in {self.base_path}")
    
    def load_data(self, file_name: str) -> Dict[str, Any]:
        """
        Load and cache test data with validation
        
        Args:
            file_name: Name of the JSON file (without .json extension)
            
        Returns:
            Dict[str, Any]: Loaded test data
            
        Raises:
            FileNotFoundError: If test data file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        if file_name in self._cache:
            return self._cache[file_name]
        
        file_path = self.base_path / f"{file_name}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Validate data structure
            if not isinstance(data, dict):
                raise ValueError(f"Invalid test data format in {file_name}: expected dict")
            
            if "questions" not in data:
                test_logger.warning(f"No 'questions' key found in {file_name}")
            
            self._cache[file_name] = data
            test_logger.debug(f"Loaded test data: {file_name}")
            return data
            
        except json.JSONDecodeError as e:
            test_logger.error(f"Invalid JSON in {file_name}: {e}")
            raise
    
    def get_question(self, file_name: str, key_name: str) -> str:
        """
        Get a specific question from test data
        
        Args:
            file_name: Test data file name
            key_name: Question key
            
        Returns:
            str: Question text
            
        Raises:
            KeyError: If question key doesn't exist
        """
        data = self.load_data(file_name)
        questions = data.get("questions", {})
        
        if key_name not in questions:
            available_keys = list(questions.keys())
            raise KeyError(f"Question '{key_name}' not found in {file_name}. Available: {available_keys}")
        
        question = str(questions[key_name])
        test_logger.debug(f"Retrieved question '{key_name}' from {file_name}")
        return question
    
    def get_reference(self, file_name: str, key_name: str) -> str:
        """
        Get reference data (ground truth) for testing
        
        Args:
            file_name: Test data file name
            key_name: Reference key
            
        Returns:
            str: Reference text
        """
        return self.get_question(file_name, key_name)  # Same structure for now
    
    def list_available_questions(self, file_name: str) -> List[str]:
        """
        List all available question keys in a test data file
        
        Args:
            file_name: Test data file name
            
        Returns:
            List[str]: Available question keys
        """
        data = self.load_data(file_name)
        return list(data.get("questions", {}).keys())


@pytest.fixture(scope="session")
def test_data_manager() -> TestDataManager:
    """
    Test data manager fixture for centralized data access
    
    Returns:
        TestDataManager: Configured test data manager
    """
    test_data_path = Path(__file__).parent / "data"
    return TestDataManager(test_data_path)


@pytest.fixture(scope="session")
def cats_test_data_manager() -> TestDataManager:
    """
    Test data manager for cats dataset
    
    Returns:
        TestDataManager: Configured cats test data manager
    """
    cats_data_path = Path(__file__).parent / "cats_dataset" / "data"
    return TestDataManager(cats_data_path)


@pytest.fixture
def unified_test_data_manager(test_data_manager: TestDataManager, cats_test_data_manager: TestDataManager):
    """
    Unified test data manager that can handle both datasets intelligently
    
    Returns:
        Callable: Function to get data from either dataset
    """
    def _get_data_manager(dataset_type: str) -> TestDataManager:
        if dataset_type.lower() in ["cats", "cats_dataset"]:
            return cats_test_data_manager
        else:
            return test_data_manager
    
    return _get_data_manager


@pytest.fixture
def get_question(test_data_manager: TestDataManager) -> Callable[[str, str], str]:
    """
    Enhanced question retrieval fixture with error handling
    
    Args:
        test_data_manager: Test data manager instance
        
    Returns:
        Callable: Function to retrieve questions
    """
    def _get_question(file_name: str, key_name: str) -> str:
        try:
            return test_data_manager.get_question(file_name, key_name)
        except (FileNotFoundError, KeyError) as e:
            test_logger.error(f"Failed to get question: {e}")
            pytest.fail(f"Test data error: {e}")

    return _get_question


@pytest.fixture
def get_reference(test_data_manager: TestDataManager) -> Callable[[str, str], str]:
    """
    Enhanced reference retrieval fixture with error handling
    
    Args:
        test_data_manager: Test data manager instance
        
    Returns:
        Callable: Function to retrieve references
    """
    def _get_reference(file_name: str, key_name: str) -> str:
        try:
            return test_data_manager.get_reference(file_name, key_name)
        except (FileNotFoundError, KeyError) as e:
            test_logger.error(f"Failed to get reference: {e}")
            pytest.fail(f"Test data error: {e}")
    
    return _get_reference


class EnhancedTestLogger:
    """Enhanced logging with structured output and performance tracking"""
    
    def __init__(self, test_config: TestConfiguration):
        self.test_config = test_config
        self.performance_data: List[Dict[str, Any]] = []
    
    def log_test_result(
        self,
        question: str,
        response: str,
        retrieved_contexts: List[Dict[str, Any]],
        reference: Optional[str] = None,
        score: Optional[float] = None,
        metric_name: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> None:
        """
        Enhanced logging with structured output and performance tracking
        
        Args:
            question: Test question
            response: Model response
            retrieved_contexts: Retrieved document contexts
            reference: Ground truth reference (optional)
            score: RAGAS metric score (optional)
            metric_name: Name of the RAGAS metric (optional)
            execution_time: Test execution time (optional)
        """
        # Format contexts for readability
        formatted_contexts = "\n".join([
            f"📄 File: {context.get('file_name', 'Unknown')}\n"
            f"   Similarity: {context.get('similarity_score', 'N/A')}\n"
            f"   Content: {context.get('page_content', '')[:200]}...\n"
            for context in retrieved_contexts
        ])
        
        # Create structured log entry
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metric": metric_name or "Unknown",
            "score": score,
            "passed": score >= self._get_threshold(metric_name) if score is not None else None,
            "execution_time": execution_time,
            "question": question,
            "response": response,
            "reference": reference,
            "context_count": len(retrieved_contexts)
        }
        
        # Store performance data
        if execution_time:
            self.performance_data.append(log_entry)
        
                    # Format console output
            separator = "=" * 80
            score_text = f"{score:.4f}" if score is not None else 'N/A'
            status_text = '✅ PASSED' if log_entry['passed'] else '❌ FAILED' if log_entry['passed'] is not None else '⏳ PENDING'
            time_text = f"{execution_time:.2f}s" if execution_time is not None else 'N/A'
            
            log_output = f"""
    {separator}
    🧪 RAGAS TEST RESULT - {metric_name or 'EVALUATION'}
    {separator}
    
    📊 METRICS:
       Score: {score_text}
       Threshold: {self._get_threshold(metric_name)}
       Status: {status_text}
       Execution Time: {time_text}

❓ QUESTION:
   {question}

🤖 RESPONSE:
   {response}

📚 RETRIEVED CONTEXTS ({len(retrieved_contexts)}):
{formatted_contexts}

📖 REFERENCE (Ground Truth):
   {reference or 'Not provided'}

{separator}
"""
        
        # Intelligent logging format selection based on configuration
        logging_format = self.test_config.logging_format.lower()
        
        if logging_format == "legacy" or self.test_config.enable_legacy_logging:
            # Use legacy logging format
            legacy_log = self._generate_legacy_log(question, response, retrieved_contexts, reference, score)
            print(legacy_log)
            
        elif logging_format == "both":
            # Show both formats for comprehensive debugging
            print("\n🔄 LEGACY FORMAT:")
            legacy_log = self._generate_legacy_log(question, response, retrieved_contexts, reference, score)
            print(legacy_log)
            print("\n🚀 ENHANCED FORMAT:")
            print(log_output)
            
        elif logging_format == "enhanced" or self.test_config.enable_verbose_logging:
            # Use enhanced logging format (default)
            print(log_output)
            
        else:
            # Fallback to simplified output
            status_emoji = "✅" if log_entry['passed'] else "❌" if log_entry['passed'] is not None else "⏳"
            score_text = f"{score:.4f}" if score is not None else 'N/A'
            time_text = f"{execution_time:.2f}s" if execution_time is not None else 'N/A'
            print(f"{status_emoji} {metric_name}: {score_text} ({time_text})")
    
    def _generate_legacy_log(
        self,
        question: str,
        response: str,
        retrieved_contexts: List[Dict[str, Any]],
        reference: Optional[str] = None,
        score: Optional[float] = None
    ) -> str:
        """
        Generate legacy-style log format for backward compatibility and debugging
        
        Args:
            question: Test question
            response: Model response
            retrieved_contexts: Retrieved document contexts
            reference: Ground truth reference (optional)
            score: RAGAS metric score (optional)
            
        Returns:
            str: Legacy formatted log string
        """
        # Format contexts in legacy style
        formatted_contexts = "\n".join([
            f"File: {context.get('file_name', 'Unknown')}\nContent: {context.get('page_content', '')}\n"
            for context in retrieved_contexts
        ])
        
        # Create legacy log format (exactly as originally implemented)
        log = "\n".join([
            "--------------------------------\n",
            "===== LOG START =====",
            f"Question: {question}",
            "---------------------",
            f"Response: {response}",
            "---------------------",
            "Retrieved Contexts:",
            formatted_contexts,
            "---------------------",
            f"Reference: {reference}" if reference else "Reference: None",
            "---------------------",
            f"Score: {score}" if score is not None else "Score: None",
            "====== LOG END ====== \n",
            "--------------------------------"
        ])
        
        return log
    
    def _get_threshold(self, metric_name: Optional[str]) -> float:
        """Get threshold for specific metric"""
        thresholds = {
            "faithfulness": self.test_config.minimum_faithfulness_score,
            "response_relevancy": self.test_config.minimum_relevancy_score,
            "context_precision": self.test_config.minimum_precision_score,
            "context_recall": self.test_config.minimum_recall_score
        }
        return thresholds.get(metric_name.lower() if metric_name else "", 0.5)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not self.performance_data:
            return {}
        
        total_tests = len(self.performance_data)
        total_time = sum(entry.get("execution_time", 0) for entry in self.performance_data)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        passed_tests = sum(1 for entry in self.performance_data if entry.get("passed", False))
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "performance_data": self.performance_data
        }


@pytest.fixture
def enhanced_test_logger(test_config: TestConfiguration) -> EnhancedTestLogger:
    """
    Enhanced test logger fixture
    
    Args:
        test_config: Test configuration
        
    Returns:
        EnhancedTestLogger: Enhanced logging instance
    """
    return EnhancedTestLogger(test_config)


@pytest.fixture
def print_log(enhanced_test_logger: EnhancedTestLogger, performance_monitor: PerformanceMonitor) -> Callable:
    """
    Backward compatible logging fixture with enhanced capabilities
    
    Args:
        enhanced_test_logger: Enhanced logger instance
        performance_monitor: Performance monitoring instance
        
    Returns:
        Callable: Logging function
    """
    def _log(
        question: str,
        response: str,
        retrieved_contexts: Union[List[Dict], List[str]],
        reference: Optional[str] = None,
        score: Optional[float] = None
    ) -> None:
        # Convert string contexts to dict format for backward compatibility
        if retrieved_contexts and isinstance(retrieved_contexts[0], str):
            retrieved_contexts = [{"page_content": ctx, "file_name": "Unknown"} for ctx in retrieved_contexts]
        
        # Extract metric name from performance monitor
        metric_name = performance_monitor.test_name.split("_")[-1] if "_" in performance_monitor.test_name else "unknown"
        
        # Get current execution time from performance monitor
        execution_time = None
        if performance_monitor.start_time:
            execution_time = time.time() - performance_monitor.start_time
        
        enhanced_test_logger.log_test_result(
            question=question,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
            score=score,
            metric_name=metric_name,
            execution_time=execution_time
        )
    
    return _log


# Additional utility fixtures for advanced testing scenarios

@pytest.fixture
def mock_embeddings_provider():
    """Mock embeddings provider for offline testing"""
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1024  # Mock 1024-dimensional vector
    mock_embeddings.embed_documents.return_value = [[0.1] * 1024] * 5  # Mock 5 documents
    return mock_embeddings


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for offline testing"""
    mock_llm = Mock()
    mock_llm.invoke.return_value.content = "This is a mock response for testing purposes."
    return LangchainLLMWrapper(mock_llm)


@pytest.fixture(scope="session")
def test_summary(request):
    """Session-scoped fixture to collect and report test summaries"""
    yield
    
    # Generate test summary at end of session
    if hasattr(request.config, "_performance_metrics"):
        metrics = request.config._performance_metrics
        total_time = sum(m.get("duration_seconds", 0) for m in metrics)
        avg_time = total_time / len(metrics) if metrics else 0
        
        test_logger.info(f"""
🧪 TEST SESSION SUMMARY
========================
Total Tests: {len(metrics)}
Total Execution Time: {total_time:.2f}s
Average Test Time: {avg_time:.2f}s
========================
        """)


# Parameterized fixtures for testing multiple scenarios

@pytest.fixture(params=["ollama", "openai"])
def provider_scenario(request, test_config: TestConfiguration):
    """Parameterized fixture for testing different providers"""
    provider = request.param
    
    # Skip if provider not available
    if provider == "openai" and not test_config.openai_api_key:
        pytest.skip(f"OpenAI provider not available (no API key)")
    
    # Override configuration for this test
    with patch.dict(os.environ, {"RAGAS_LLM_PROVIDER": provider, "EMBEDDING_PROVIDER": provider}):
        yield provider


@pytest.fixture(params=[0.0, 0.3, 0.7])
def temperature_scenario(request):
    """Parameterized fixture for testing different temperature settings"""
    temperature = request.param
    with patch.dict(os.environ, {"TEST_TEMPERATURE": str(temperature)}):
        yield temperature


# Context managers for test isolation

@contextmanager
def isolated_test_environment(**env_overrides):
    """Context manager for isolated test environment"""
    original_env = dict(os.environ)
    try:
        os.environ.update(env_overrides)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture
def isolated_environment():
    """Fixture providing isolated environment context manager"""
    return isolated_test_environment
