"""
Database Setup Module for RAG System

This module provides enterprise-grade document processing and vector database management
for the RAG system. It handles document loading, chunking, embedding generation,
and ChromaDB population with proper error handling and configuration management.

Key Features:
- Flexible document loading from multiple sources
- Configurable text chunking strategies
- Intelligent duplicate detection and incremental updates
- Path resolution that works from any directory
- Comprehensive error handling and logging
- Environment-based configuration
"""

import argparse
import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

try:
    from get_embedding_function import get_embedding_function
except ImportError:
    from scripts.get_embedding_function import get_embedding_function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Configuration class for database setup"""
    
    # Path Configuration - Dynamic resolution
    chroma_path: str = None
    data_path: str = None
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "qwen2.5-7b-instruct")
    
    # Document Processing Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    max_documents: Optional[int] = None if not os.getenv("MAX_DOCUMENTS") else int(os.getenv("MAX_DOCUMENTS"))
    
    # Processing Options
    enable_incremental_updates: bool = os.getenv("ENABLE_INCREMENTAL_UPDATES", "true").lower() == "true"
    enable_metadata_enrichment: bool = os.getenv("ENABLE_METADATA_ENRICHMENT", "true").lower() == "true"
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
    
    def __post_init__(self):
        """Initialize dynamic paths after instantiation"""
        if self.chroma_path is None:
            self.chroma_path = self._resolve_chroma_path()
        if self.data_path is None:
            self.data_path = self._resolve_data_path()
    
    def _resolve_chroma_path(self) -> str:
        """Resolve ChromaDB path based on current working directory"""
        current_dir = Path.cwd()
        if current_dir.name == 'scripts':
            return str(current_dir.parent / "chroma")
        else:
            return str(current_dir / "chroma")
    
    def _resolve_data_path(self) -> str:
        """Resolve data path based on current working directory"""
        current_dir = Path.cwd()
        if current_dir.name == 'scripts':
            return str(current_dir.parent / "data")
        else:
            return str(current_dir / "data")


class DocumentProcessor:
    """Handles document loading and processing operations"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.stats = {
            'documents_loaded': 0,
            'chunks_created': 0,
            'processing_time': 0
        }
    
    def load_documents(self, data_path: Optional[str] = None) -> List[Document]:
        """
        Load documents from specified directory with error handling
        
        Args:
            data_path: Path to documents directory (defaults to config)
            
        Returns:
            List of loaded documents
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If no documents found or documents are invalid
        """
        data_path = data_path or self.config.data_path
        data_path_obj = Path(data_path)
        
        if not data_path_obj.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_path}. "
                f"Please create the directory and add PDF files."
            )
        
        if not data_path_obj.is_dir():
            raise ValueError(f"Data path is not a directory: {data_path}")
        
        # Check for PDF files
        pdf_files = list(data_path_obj.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(
                f"No PDF files found in {data_path}. "
                f"Please add PDF files to process."
            )
        
        logger.info(f"Found {len(pdf_files)} PDF files in {data_path}")
        
        try:
            start_time = datetime.now()
            loader = PyPDFDirectoryLoader(str(data_path))
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content could be extracted from PDF files. Check if PDFs contain readable text.")
            
            # Apply document limit if specified
            if self.config.max_documents:
                documents = documents[:self.config.max_documents]
                logger.info(f"Limited to {self.config.max_documents} documents")
            
            # Enrich metadata if enabled
            if self.config.enable_metadata_enrichment:
                documents = self._enrich_document_metadata(documents)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats.update({
                'documents_loaded': len(documents),
                'processing_time': processing_time
            })
            
            logger.info(f"Successfully loaded {len(documents)} document pages in {processing_time:.2f}s")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {data_path}: {e}")
            raise
    
    def _enrich_document_metadata(self, documents: List[Document]) -> List[Document]:
        """Enrich document metadata with additional information"""
        for doc in documents:
            # Add processing timestamp
            doc.metadata['processed_at'] = datetime.now().isoformat()
            
            # Add document statistics
            doc.metadata['content_length'] = len(doc.page_content)
            doc.metadata['word_count'] = len(doc.page_content.split())
            
            # Improve source path (make relative to project root)
            if 'source' in doc.metadata:
                source_path = Path(doc.metadata['source'])
                try:
                    # Try to make path relative to project root
                    current_dir = Path.cwd()
                    if current_dir.name == 'scripts':
                        project_root = current_dir.parent
                    else:
                        project_root = current_dir
                    
                    relative_path = source_path.relative_to(project_root)
                    doc.metadata['source'] = str(relative_path)
                except ValueError:
                    # Keep original path if can't make relative
                    pass
        
        return documents
    
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        Split documents into chunks with configurable parameters
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
            
        Returns:
            List of document chunks
            
        Raises:
            ValueError: If chunking parameters are invalid
        """
        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap
        
        if chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got: {chunk_size}")
        
        if chunk_overlap >= chunk_size:
            raise ValueError(f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})")
        
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
        
        try:
            start_time = datetime.now()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
            chunks = text_splitter.split_documents(documents)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.stats['chunks_created'] = len(chunks)
            
            logger.info(
                f"Split {len(documents)} documents into {len(chunks)} chunks "
                f"(size: {chunk_size}, overlap: {chunk_overlap}) in {processing_time:.2f}s"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise


class ChromaDBManager:
    """Manages ChromaDB operations with advanced features"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._db = None
        self.stats = {
            'existing_docs': 0,
            'new_docs_added': 0,
            'duplicates_skipped': 0
        }
    
    @property
    def db(self) -> Chroma:
        """Lazy-loaded ChromaDB instance"""
        if self._db is None:
            try:
                self._db = Chroma(
                    persist_directory=self.config.chroma_path,
                    embedding_function=get_embedding_function(),
                    collection_name=self.config.collection_name
                )
                logger.info(f"Connected to ChromaDB at {self.config.chroma_path}")
            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB: {e}")
                raise ConnectionError(f"Cannot connect to ChromaDB: {e}")
        
        return self._db
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            existing_items = self.db.get(include=["metadatas"])
            
            stats = {
                'total_documents': len(existing_items["ids"]),
                'collection_name': self.config.collection_name,
                'database_path': self.config.chroma_path,
                'last_updated': datetime.now().isoformat()
            }
            
            # Add metadata analysis if available
            if existing_items.get("metadatas"):
                sources = set()
                for metadata in existing_items["metadatas"]:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
                stats['unique_sources'] = len(sources)
                stats['source_files'] = sorted(list(sources))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    def add_documents_batch(
        self,
        chunks: List[Document],
        batch_size: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Add documents to ChromaDB with batching and duplicate detection
        
        Args:
            chunks: List of document chunks to add
            batch_size: Size of batches for processing (defaults to config)
            
        Returns:
            Dictionary with processing statistics
            
        Raises:
            ValueError: If chunks are invalid
            ConnectionError: If database is not accessible
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return {'new_docs_added': 0, 'duplicates_skipped': 0}
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            # Calculate chunk IDs
            chunks_with_ids = self._calculate_chunk_ids(chunks)
            
            # Get existing document IDs
            existing_items = self.db.get(include=[])
            existing_ids = set(existing_items["ids"])
            self.stats['existing_docs'] = len(existing_ids)
            
            logger.info(f"Found {len(existing_ids)} existing documents in database")
            
            # Filter out duplicates if incremental updates are enabled
            if self.config.enable_incremental_updates:
                new_chunks = [
                    chunk for chunk in chunks_with_ids
                    if chunk.metadata["id"] not in existing_ids
                ]
                self.stats['duplicates_skipped'] = len(chunks_with_ids) - len(new_chunks)
            else:
                new_chunks = chunks_with_ids
                self.stats['duplicates_skipped'] = 0
            
            if not new_chunks:
                logger.info("✅ No new documents to add")
                return self.stats
            
            logger.info(f"👉 Adding {len(new_chunks)} new documents to database")
            
            # Process in batches
            total_added = 0
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                batch_ids = [chunk.metadata["id"] for chunk in batch]
                
                try:
                    self.db.add_documents(batch, ids=batch_ids)
                    total_added += len(batch)
                    
                    if len(new_chunks) > batch_size:
                        logger.info(f"Processed batch {i//batch_size + 1}/{(len(new_chunks)-1)//batch_size + 1}")
                
                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                    raise
            
            self.stats['new_docs_added'] = total_added
            logger.info(f"✅ Successfully added {total_added} documents to database")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def _calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """
        Calculate unique IDs for document chunks
        
        Creates IDs in format: "source_file.pdf:page:chunk_index"
        """
        if not chunks:
            return []
        
        last_page_id = None
        current_chunk_index = 0
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            current_page_id = f"{source}:{page}"
            
            # If the page ID is the same as the last one, increment the index
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            
            # Calculate the chunk ID
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            
            # Add it to the chunk metadata
            chunk.metadata["id"] = chunk_id
        
        return chunks
    
    def clear_database(self, confirm: bool = False) -> bool:
        """
        Clear the ChromaDB database with safety confirmation
        
        Args:
            confirm: If True, skip interactive confirmation
            
        Returns:
            True if database was cleared, False otherwise
        """
        db_path = Path(self.config.chroma_path)
        
        if not db_path.exists():
            logger.info("Database directory doesn't exist, nothing to clear")
            return True
        
        if not confirm:
            # In production/automated environments, require explicit confirmation
            logger.warning(f"Database clear requested for: {db_path}")
            logger.warning("Use confirm=True parameter to proceed")
            return False
        
        try:
            # Close database connection first
            if self._db is not None:
                self._db = None
            
            shutil.rmtree(db_path)
            logger.info(f"✨ Database cleared: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise


class DatabaseSetup:
    """Main class orchestrating the database setup process"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.processor = DocumentProcessor(self.config)
        self.db_manager = ChromaDBManager(self.config)
        self.setup_stats = {}
    
    def setup_database(
        self,
        data_path: Optional[str] = None,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Complete database setup process
        
        Args:
            data_path: Custom data directory path
            force_rebuild: If True, clear database before setup
            
        Returns:
            Setup statistics and results
        """
        start_time = datetime.now()
        
        try:
            logger.info("🚀 Starting database setup process")
            
            # Clear database if requested
            if force_rebuild:
                logger.info("🗑️ Clearing existing database")
                self.db_manager.clear_database(confirm=True)
            
            # Load and process documents
            logger.info("📚 Loading documents")
            documents = self.processor.load_documents(data_path)
            
            logger.info("✂️ Splitting documents into chunks")
            chunks = self.processor.split_documents(documents)
            
            # Add to ChromaDB
            logger.info("💾 Adding chunks to ChromaDB")
            db_stats = self.db_manager.add_documents_batch(chunks)
            
            # Compile final statistics
            total_time = (datetime.now() - start_time).total_seconds()
            
            self.setup_stats = {
                'success': True,
                'total_time': total_time,
                'documents_processed': self.processor.stats['documents_loaded'],
                'chunks_created': self.processor.stats['chunks_created'],
                'existing_docs': db_stats.get('existing_docs', 0),
                'new_docs_added': db_stats.get('new_docs_added', 0),
                'duplicates_skipped': db_stats.get('duplicates_skipped', 0),
                'config': {
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap,
                    'collection_name': self.config.collection_name,
                    'data_path': self.config.data_path,
                    'chroma_path': self.config.chroma_path
                }
            }
            
            logger.info(f"✅ Database setup completed successfully in {total_time:.2f}s")
            return self.setup_stats
            
        except Exception as e:
            error_msg = f"Database setup failed: {e}"
            logger.error(error_msg)
            
            self.setup_stats = {
                'success': False,
                'error': str(e),
                'total_time': (datetime.now() - start_time).total_seconds()
            }
            
            raise RuntimeError(error_msg) from e
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        return self.db_manager.get_database_stats()


# Backward compatibility functions
def load_documents() -> List[Document]:
    """Legacy function for backward compatibility"""
    config = DatabaseConfig()
    processor = DocumentProcessor(config)
    return processor.load_documents()


def split_documents(documents: List[Document]) -> List[Document]:
    """Legacy function for backward compatibility"""
    config = DatabaseConfig()
    processor = DocumentProcessor(config)
    return processor.split_documents(documents)


def add_to_chroma(chunks: List[Document]) -> None:
    """Legacy function for backward compatibility"""
    config = DatabaseConfig()
    db_manager = ChromaDBManager(config)
    db_manager.add_documents_batch(chunks)


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Legacy function for backward compatibility"""
    config = DatabaseConfig()
    db_manager = ChromaDBManager(config)
    return db_manager._calculate_chunk_ids(chunks)


def clear_database() -> None:
    """Legacy function for backward compatibility"""
    config = DatabaseConfig()
    db_manager = ChromaDBManager(config)
    db_manager.clear_database(confirm=True)


def main():
    """Main function for CLI usage - maintains backward compatibility"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Set up vector database for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python database_setup.py                    # Setup with default settings
  python database_setup.py --reset            # Clear and rebuild database
  python database_setup.py --data-path ./docs # Use custom data directory
  python database_setup.py --chunk-size 1000  # Use custom chunk size
        """
    )
    
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    parser.add_argument("--data-path", help="Custom data directory path")
    parser.add_argument("--chunk-size", type=int, help="Custom chunk size")
    parser.add_argument("--chunk-overlap", type=int, help="Custom chunk overlap")
    parser.add_argument("--collection-name", help="Custom collection name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create configuration
        config = DatabaseConfig()
        
        # Override with command line arguments
        if args.chunk_size:
            config.chunk_size = args.chunk_size
        if args.chunk_overlap:
            config.chunk_overlap = args.chunk_overlap
        if args.collection_name:
            config.collection_name = args.collection_name
        
        # Create database setup instance
        db_setup = DatabaseSetup(config)
        
        # Handle reset flag with legacy output
        if args.reset:
            print("✨ Clearing Database")
            db_setup.db_manager.clear_database(confirm=True)
        
        # Run setup process
        stats = db_setup.setup_database(
            data_path=args.data_path,
            force_rebuild=False  # Don't force rebuild if --reset already handled
        )
        
        # Legacy output format for backward compatibility
        print(f"Number of existing documents in DB: {stats['existing_docs']}")
        
        if stats['new_docs_added'] > 0:
            print(f"👉 Adding new documents: {stats['new_docs_added']}")
        else:
            print("✅ No new documents to add")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
