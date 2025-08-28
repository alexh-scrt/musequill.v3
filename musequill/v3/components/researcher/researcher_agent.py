"""
Refactored Researcher Component

Implements the researcher functionality using the standardized component interface.
Uses the existing model definitions from musequill/v3/models/researcher_agent_model.py
and configuration from musequill.services.backend.researcher.
"""

import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, cast, Iterable
from urllib.parse import urlparse
from uuid import uuid4
import traceback
import logging
from hashlib import sha256

import chromadb
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from tavily import TavilyClient
from difflib import SequenceMatcher
from pydantic import BaseModel, Field

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)

# Import the existing model definitions
from musequill.v3.models.researcher_agent_model import (
    QueryStatus, SearchResult, ProcessedChunk, ResearchResults, ResearchQuery
)

# Import the existing configuration (adjust path as needed)
from .researcher_agent_config import ResearcherConfig

logger = logging.getLogger(__name__)


# Input/Output Models for the Component Interface
class ResearcherInput(BaseModel):
    """Input model for researcher component."""
    research_id: str = Field(description="Unique identifier for the research session")
    queries: List[ResearchQuery] = Field(description="List of research queries to execute")
    force_refresh: bool = Field(default=False, description="Force refresh of cached results")


class ResearcherOutput(BaseModel):
    """Output model for researcher component."""
    research_id: str = Field(description="Research session identifier")
    updated_queries: List[ResearchQuery] = Field(description="Updated queries with results")
    total_chunks: int = Field(description="Total chunks stored")
    total_sources: int = Field(description="Total sources processed")
    stats: Dict[str, Any] = Field(description="Execution statistics")
    chroma_storage_info: Dict[str, Any] = Field(description="ChromaDB storage information")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")


class ResearcherComponent(BaseComponent[ResearcherInput, ResearcherOutput, ResearcherConfig]):
    """
    Researcher component implementing the standardized component interface.
    
    Uses existing models from musequill/v3/models/researcher_agent_model.py and
    configuration from musequill.services.backend.researcher.
    """
    
    def __init__(self, config: ComponentConfiguration[ResearcherConfig]):
        super().__init__(config)
        
        # Initialize clients
        self.tavily_client: Optional[TavilyClient] = None
        self.chroma_client: Optional[chromadb.HttpClient] = None
        self.chroma_collection = None
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        
        # Content tracking for deduplication
        self.content_hashes: Set[str] = set()
        self.processed_urls: Set[str] = set()
        
        # Statistics tracking
        self.stats = {
            'queries_processed': 0,
            'queries_failed': 0,
            'total_chunks_stored': 0,
            'total_sources_processed': 0,
            'duplicate_content_filtered': 0,
            'low_quality_filtered': 0,
            'processing_start_time': None
        }
    
    async def initialize(self) -> bool:
        """Initialize the researcher component."""
        try:
            # Initialize Tavily client
            if not self.config.specific_config.tavily_api_key:
                logger.error("Tavily API key not provided")
                return False
            
            self.tavily_client = TavilyClient(api_key=self.config.specific_config.tavily_api_key)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.HttpClient(
                host=self.config.specific_config.chroma_host,
                port=self.config.specific_config.chroma_port
            )
            
            # Get or create collection (using the same approach as original)
            self.chroma_collection = self._get_or_create_collection_safe(
                self.config.specific_config
            )
            
            # Initialize Ollama embeddings
            self.embeddings = OllamaEmbeddings(
                base_url=self.config.specific_config.ollama_base_url,
                model=getattr(self.config.specific_config, 'embedding_model', 'nomic-embed-text')
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.specific_config.chunk_size,
                chunk_overlap=self.config.specific_config.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            self.stats['processing_start_time'] = time.time()
            logger.info(f"Researcher component {self.config.component_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize researcher component: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def process(self, input_data: ResearcherInput) -> ResearcherOutput:
        """Process research queries and return results."""
        try:
            logger.info(f"Starting research for {len(input_data.queries)} queries")
            
            # Execute research queries
            results = await self._execute_queries_concurrently(
                input_data.queries, 
                input_data.research_id
            )
            
            # Update query statuses
            updated_queries = self._update_query_statuses(input_data.queries, results)
            
            # Calculate totals
            total_chunks = sum(
                sum(r.total_chunks_stored for r in result_list) 
                for result_list in results.values()
            )
            total_sources = sum(
                sum(r.total_sources for r in result_list)
                for result_list in results.values()
            )
            
            # Get storage info
            storage_info = self._get_chroma_storage_info(input_data.research_id)
            
            return ResearcherOutput(
                research_id=input_data.research_id,
                updated_queries=updated_queries,
                total_chunks=total_chunks,
                total_sources=total_sources,
                stats=self.get_current_stats(),
                chroma_storage_info=storage_info
            )
            
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            logger.error(traceback.format_exc())
            
            # Mark all queries as failed
            failed_queries = self._mark_queries_failed(input_data.queries, str(e))
            
            return ResearcherOutput(
                research_id=input_data.research_id,
                updated_queries=failed_queries,
                total_chunks=0,
                total_sources=0,
                stats=self.get_current_stats(),
                chroma_storage_info=self._get_chroma_storage_info(input_data.research_id),
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        """Perform health check on the researcher component."""
        try:
            # Check Tavily client
            if not self.tavily_client:
                return False
            
            # Check ChromaDB connection
            if not self.chroma_client or not self.chroma_collection:
                return False
            
            # Try a simple ChromaDB operation
            try:
                self.chroma_collection.count()
            except Exception:
                return False
            
            # Check Ollama embeddings
            if not self.embeddings:
                return False
            
            # Try a simple embedding operation
            try:
                await self.embeddings.aembed_query("health check")
            except Exception:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup researcher component resources."""
        try:
            # Close connections if needed
            self.tavily_client = None
            self.chroma_client = None
            self.chroma_collection = None
            self.embeddings = None
            self.text_splitter = None
            
            # Clear tracking sets
            self.content_hashes.clear()
            self.processed_urls.clear()
            
            logger.info(f"Researcher component {self.config.component_id} cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup researcher component: {e}")
            return False
    
    # Helper method for collection management (from original implementation)
    def _get_or_create_collection_safe(self, config: ResearcherConfig, drop_if_exists: bool = False):
        """Safely get or create ChromaDB collection, handling dimension mismatches."""
        collection_name = config.chroma_collection_name
        expected_embedding_model = getattr(config, 'embedding_model', 'nomic-embed-text')
        
        # Get expected dimensions for current model
        model_dimensions = {
            'nomic-embed-text': 768,
            'mxbai-embed-large': 1024,
            'all-minilm': 384,
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        
        try:
            # Try to get existing collection
            if drop_if_exists:
                try:
                    self.chroma_client.delete_collection(name=collection_name)
                except:
                    pass  # Collection didn't exist
            
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # Check if the collection metadata indicates a different embedding model
            collection_metadata = collection.metadata or {}
            stored_model = collection_metadata.get('embedding_model', 'unknown')
            
            expected_dims = model_dimensions.get(expected_embedding_model, 768)
            stored_dims = model_dimensions.get(stored_model, None)
            
            # If dimensions don't match, recreate collection
            if stored_dims and stored_dims != expected_dims:
                logger.warning(f"Collection '{collection_name}' expects {stored_dims}D embeddings "
                             f"(model: {stored_model}), but current model '{expected_embedding_model}' "
                             f"produces {expected_dims}D embeddings. Recreating collection.")
                
                # Delete and recreate
                self.chroma_client.delete_collection(name=collection_name)
                collection = self._create_new_collection(config, expected_embedding_model, expected_dims)
            else:
                logger.info(f"Using existing collection '{collection_name}' with model: {stored_model}")
            
            return collection
            
        except (Exception, NotFoundError):
            # Collection doesn't exist, create new one
            logger.info(f"Creating new collection '{collection_name}'")
            expected_dims = model_dimensions.get(expected_embedding_model, 768)
            return self._create_new_collection(config, expected_embedding_model, expected_dims)
    
    def _create_new_collection(self, config: ResearcherConfig, embedding_model: str, dimensions: int):
        """Create a new ChromaDB collection with proper metadata."""
        try:
            collection = self.chroma_client.create_collection(
                name=config.chroma_collection_name,
                metadata={
                    "embedding_model": embedding_model,
                    "dimensions": dimensions,
                    "created_at": datetime.now().isoformat(),
                    "description": "Research content collection with embeddings"
                }
            )
            logger.info(f"Created new collection '{config.chroma_collection_name}' "
                       f"for model '{embedding_model}' ({dimensions}D)")
            return collection
        except Exception as e:
            logger.error(f"Failed to create new collection: {e}")
            raise
    
    # Private methods from the original implementation
    async def _execute_queries_concurrently(
        self, 
        queries: List[ResearchQuery], 
        research_id: str
    ) -> Dict[str, List[ResearchResults]]:
        """Execute research queries with controlled concurrency."""
        results: Dict[str, List[ResearchResults]] = {}
        
        # Sort queries by priority
        sorted_queries = self._sort_research_queries_by_priority(queries)
        
        # Process queries in batches
        batch_size = self.config.specific_config.max_concurrent_queries
        
        for i in range(0, len(sorted_queries), batch_size):
            batch = sorted_queries[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} queries")
            
            # Execute batch concurrently
            batch_results = await self._execute_query_batch(batch, research_id)
            results.update(batch_results)
            
            # Rate limiting between batches
            if i + batch_size < len(sorted_queries):
                await asyncio.sleep(self.config.specific_config.rate_limit_delay)
        
        return results
    
    def _sort_research_queries_by_priority(self, queries: List[ResearchQuery]) -> List[ResearchQuery]:
        """Sort queries by priority."""
        priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
        return sorted(
            queries, 
            key=lambda query: priority_order.get(query.priority, 4)
        )
    
    async def _execute_query_batch(
        self, 
        queries: List[ResearchQuery], 
        research_id: str
    ) -> Dict[str, List[ResearchResults]]:
        """Execute a batch of queries concurrently."""
        tasks = []
        
        for query in queries:
            task = asyncio.create_task(
                self._research_single_query(query, research_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_results: Dict[str, List[ResearchResults]] = {}
        
        for results in task_results:
            if isinstance(results, Exception):
                logger.error(f"Query failed: {results}")
                self.stats['queries_failed'] += 1
            else:
                if isinstance(results, list):
                    for result in cast(Iterable[ResearchResults], results):
                        batch_results.setdefault(result.query, []).append(result)
                        self.stats['queries_processed'] += 1
        
        return batch_results
    
    async def _research_single_query(
        self, 
        query: ResearchQuery, 
        research_id: str
    ) -> List[ResearchResults]:
        """Research a single query with retries."""
        results: List[ResearchResults] = []
        start_time = time.time()
        
        main_query: str = query.get_query()
        research_queries: List[str] = query.get_questions() or []
        research_queries.append(main_query)
        
        for _query in research_queries:
            for attempt in range(self.config.specific_config.query_retry_attempts):
                try:
                    logger.info(f"Executing query [{_query}] (attempt {attempt + 1})")
                    
                    # Perform web search
                    search_results = await self._perform_web_search(_query)
                    
                    # Filter and validate results
                    filtered_results = self._filter_search_results(search_results)
                    
                    # Process content and create chunks
                    processed_chunks = await self._process_search_results(
                        filtered_results, query, research_id
                    )
                    
                    # Store chunks in vector database
                    chunks_stored = await self._store_chunks_in_chroma(processed_chunks, research_id)
                    
                    # Calculate quality statistics
                    quality_stats = self._calculate_quality_stats(processed_chunks)
                    
                    execution_time = time.time() - start_time
                    
                    result = ResearchResults(
                        query=query.category or "unknown",
                        search_results=filtered_results,
                        processed_chunks=[],  # Don't include full chunks in response
                        total_chunks_stored=chunks_stored,
                        total_sources=len(filtered_results),
                        quality_stats=quality_stats,
                        execution_time=execution_time,
                        status='completed'
                    )
                    
                    results.append(result)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.error(f"Query attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.specific_config.query_retry_attempts - 1:
                        # Final attempt failed
                        result = ResearchResults(
                            query=query.category or "unknown",
                            search_results=[],
                            processed_chunks=[],
                            total_chunks_stored=0,
                            total_sources=0,
                            quality_stats={},
                            execution_time=time.time() - start_time,
                            status='failed',
                            error_message=str(e)
                        )
                        results.append(result)
        
        return results
    
    async def _perform_web_search(self, query: str) -> List[SearchResult]:
        """Perform web search using Tavily."""
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth=self.config.specific_config.tavily_search_depth,
                max_results=self.config.specific_config.tavily_max_results,
                include_answer=self.config.specific_config.tavily_include_answer,
                include_raw_content=self.config.specific_config.tavily_include_raw_content
            )
            
            search_results = []
            for result in response.get('results', []):
                search_result = SearchResult(
                    url=result.get('url', ''),
                    title=result.get('title', ''),
                    content=result.get('content', ''),
                    score=result.get('score', 0.0),
                    published_date=result.get('published_date'),
                    domain=urlparse(result.get('url', '')).netloc,
                    query=query,
                    tavily_answer=response.get('answer') if self.config.specific_config.tavily_include_answer else None
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return []
    
    def _filter_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter search results based on quality and domain restrictions."""
        filtered_results = []
        
        for result in results:
            # Skip blocked domains
            if self._is_domain_blocked(result.domain):
                continue
            
            # Skip already processed URLs
            if result.url in self.processed_urls:
                continue
            
            # Assess content quality
            if not self._assess_content_quality(result):
                self.stats['low_quality_filtered'] += 1
                continue
            
            filtered_results.append(result)
            self.processed_urls.add(result.url)
        
        return filtered_results
    
    def _is_domain_blocked(self, domain: str) -> bool:
        """Check if domain is blocked."""
        blocked_domains = getattr(self.config.specific_config, 'blocked_domains', [])
        for blocked in blocked_domains:
            if blocked.lower() in domain.lower():
                return True
        return False
    
    def _is_domain_trusted(self, domain: str) -> bool:
        """Check if domain is trusted."""
        trusted_domains = getattr(self.config.specific_config, 'trusted_domains', [])
        for trusted in trusted_domains:
            if trusted.lower() in domain.lower():
                return True
        return False
    
    def _assess_content_quality(self, result: SearchResult) -> bool:
        """Assess content quality based on various criteria."""
        if not self.config.specific_config.enable_content_filtering:
            return True
        
        content = result.tavily_answer or result.content
        
        if not content or len(content.strip()) < self.config.specific_config.min_chunk_size:
            return False
        
        # Basic quality scoring
        quality_score = 0.0
        
        # Domain trust score
        if self._is_domain_trusted(result.domain):
            quality_score += 0.3
        
        # Tavily score contribution
        quality_score += result.score * 0.4
        
        # Content structure indicators
        sentence_count = len(re.findall(r'[.!?]+', content))
        if sentence_count > 3:
            quality_score += 0.2
        
        # Meaningful word count
        meaningful_words = len(re.findall(r'\b[a-zA-Z]{4,}\b', content))
        if meaningful_words > 50:
            quality_score += 0.1
        
        return quality_score >= self.config.specific_config.min_content_quality_score
    
    async def _process_search_results(
        self,
        results: List[SearchResult],
        query: ResearchQuery,
        research_id: str
    ) -> List[ProcessedChunk]:
        """Process search results into chunks with embeddings."""
        processed_chunks = []
        
        for result in results:
            try:
                # Get content
                content = result.tavily_answer or result.content
                if not content:
                    continue
                
                # Truncate if too long
                if len(content) > self.config.specific_config.max_content_length:
                    content = content[:self.config.specific_config.max_content_length]
                
                # Split into chunks
                text_chunks = self.text_splitter.split_text(content)
                
                for i, chunk_text in enumerate(text_chunks):
                    if len(chunk_text.strip()) < self.config.specific_config.min_chunk_size:
                        continue
                    
                    # Check for content duplication
                    if self.config.specific_config.filter_duplicate_content:
                        content_hash = self._get_content_hash(chunk_text)
                        if content_hash in self.content_hashes:
                            self.stats['duplicate_content_filtered'] += 1
                            continue
                        self.content_hashes.add(content_hash)
                    
                    # Generate embedding
                    embedding = await self.embeddings.aembed_query(chunk_text)
                    
                    # Create chunk metadata
                    chunk_id = str(uuid4())
                    metadata = {
                        'research_id': research_id,
                        'query_category': query.category or "unknown",
                        'query_topic': query.topic or "unknown",
                        'source_url': result.url,
                        'source_title': result.title,
                        'source_domain': result.domain,
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'processed_at': datetime.now(timezone.utc).isoformat(),
                        'tavily_score': result.score,
                        'query_priority': query.priority or "Medium",
                        'embedding_model': getattr(self.config.specific_config, 'embedding_model', 'nomic-embed-text')
                    }
                    
                    # Calculate quality score
                    quality_score = self._calculate_chunk_quality_score(chunk_text, result, query)
                    
                    source_info = {
                        'url': result.url,
                        'title': result.title,
                        'domain': result.domain,
                        'tavily_score': result.score
                    }
                    
                    processed_chunk = ProcessedChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        embedding=embedding,
                        metadata=metadata,
                        quality_score=quality_score,
                        source_info=source_info
                    )
                    
                    processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.error(f"Error processing result from {result.url}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_chunks)} chunks from {len(results)} search results")
        return processed_chunks
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_chunk_quality_score(
        self, 
        chunk_text: str, 
        result: SearchResult, 
        query: ResearchQuery
    ) -> float:
        """Calculate quality score for a content chunk."""
        score = 0.0
        
        # Base score from Tavily
        score += result.score * 0.3
        
        # Domain trust factor
        if self._is_domain_trusted(result.domain):
            score += 0.2
        
        # Content length factor
        text_length = len(chunk_text)
        if 200 <= text_length <= 800:
            score += 0.15
        elif 100 <= text_length <= 1200:
            score += 0.1
        
        # Query relevance
        query_words = set(query.get_query().lower().split())
        chunk_words = set(chunk_text.lower().split())
        relevance = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
        score += relevance * 0.2
        
        # Content structure indicators
        if len(re.findall(r'[.!?]', chunk_text)) >= 2:
            score += 0.1
        
        # Avoid spam content
        spam_indicators = ['click here', 'subscribe now', 'buy now', '!!!', 'free trial']
        if not any(indicator in chunk_text.lower() for indicator in spam_indicators):
            score += 0.05
        
        return min(1.0, score)
    
    async def _store_chunks_in_chroma(
        self, 
        chunks: List[ProcessedChunk], 
        research_id: str
    ) -> int:
        """Store processed chunks in ChromaDB."""
        if not chunks:
            return 0
        
        try:
            # Prepare data for batch insertion
            ids = [chunk.chunk_id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Insert in batches
            batch_size = self.config.specific_config.batch_size
            stored_count = 0
            
            for i in range(0, len(chunks), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                try:
                    self.chroma_collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    stored_count += len(batch_ids)
                except Exception as e:
                    logger.error(f"Failed to store batch of chunks: {e}")
            
            self.stats['total_chunks_stored'] += stored_count
            logger.info(f"Stored {stored_count} chunks in ChromaDB")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing chunks in ChromaDB: {e}")
            return 0
    
    def _calculate_quality_stats(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Calculate quality statistics for processed chunks."""
        if not chunks:
            return {}
        
        quality_scores = [chunk.quality_score for chunk in chunks]
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        
        # Source diversity
        unique_domains = len(set(chunk.source_info['domain'] for chunk in chunks))
        unique_urls = len(set(chunk.source_info['url'] for chunk in chunks))
        
        # Quality distribution
        high_quality = sum(1 for score in quality_scores if score >= 0.7)
        medium_quality = sum(1 for score in quality_scores if 0.4 <= score < 0.7)
        low_quality = sum(1 for score in quality_scores if score < 0.4)
        
        return {
            'total_chunks': len(chunks),
            'avg_quality_score': sum(quality_scores) / len(quality_scores),
            'min_quality_score': min(quality_scores),
            'max_quality_score': max(quality_scores),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'unique_domains': unique_domains,
            'unique_sources': unique_urls,
            'quality_distribution': {
                'high_quality': high_quality,
                'medium_quality': medium_quality,
                'low_quality': low_quality
            }
        }
    
    def _update_query_statuses(
        self,
        original_queries: List[ResearchQuery],
        results: Dict[str, List[ResearchResults]]
    ) -> List[ResearchQuery]:
        """Update query statuses based on research results."""
        updated_queries = []
        
        for query in original_queries:
            updated_query = query.model_copy()
            
            query_key = query.get_query()
            if query_key in results:
                result_list = results[query_key]
                if result_list:
                    # Use the first result for status
                    first_result = result_list[0]
                    updated_query.query_type = (
                        QueryStatus.COMPLETED if first_result.status == 'completed' 
                        else QueryStatus.FAILED
                    )
            
            updated_queries.append(updated_query)
        
        return updated_queries
    
    def _mark_queries_failed(
        self, 
        queries: List[ResearchQuery], 
        error_message: str
    ) -> List[ResearchQuery]:
        """Mark all queries as failed with error message."""
        updated_queries = []
        
        for query in queries:
            updated_query = query.model_copy()
            if updated_query.query_type.is_pending():
                updated_query.query_type = QueryStatus.FAILED
            updated_queries.append(updated_query)
        
        return updated_queries
    
    def _get_chroma_storage_info(self, research_id: str) -> Dict[str, Any]:
        """Get ChromaDB storage information for the research session."""
        try:
            results = self.chroma_collection.get(
                where={"research_id": research_id},
                include=["metadatas"]
            )
            
            return {
                'research_id': research_id,
                'total_chunks': len(results['metadatas']) if results['metadatas'] else 0,
                'collection_name': self.config.specific_config.chroma_collection_name,
                'host': self.config.specific_config.chroma_host,
                'port': self.config.specific_config.chroma_port
            }
            
        except Exception as e:
            return {
                'research_id': research_id,
                'error': str(e),
                'collection_name': self.config.specific_config.chroma_collection_name
            }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current execution statistics."""
        current_stats = self.stats.copy()
        
        if current_stats['processing_start_time']:
            current_stats['total_processing_time'] = time.time() - current_stats['processing_start_time']
        
        # Add collection statistics if available
        try:
            collection_count = self.chroma_collection.count()
            current_stats['total_chunks_in_collection'] = collection_count
        except:
            current_stats['total_chunks_in_collection'] = 'unavailable'
        
        # Add configuration info
        current_stats['embedding_model'] = getattr(
            self.config.specific_config, 'embedding_model', 'nomic-embed-text'
        )
        current_stats['ollama_base_url'] = self.config.specific_config.ollama_base_url
        
        return current_stats
    
    def is_similar_content(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar based on their first 100 characters.
        
        Args:
            text1: First text to compare
            text2: Second text to compare  
            threshold: Similarity threshold (0.0-1.0), default 0.8
            
        Returns:
            True if texts are similar, False otherwise
        """
        if not text1 or not text2:
            return False
        
        # Compare first 100 characters
        sample1 = text1[:100].strip().lower()
        sample2 = text2[:100].strip().lower()
        
        # Use SequenceMatcher to calculate similarity ratio
        similarity = SequenceMatcher(None, sample1, sample2).ratio()
        return similarity >= threshold
    
    def extract_refined_search_results(self, research_output: ResearcherOutput) -> Dict[str, List[str]]:
        """
        Extract and refine Tavily answers from research results organized by category.
        Filters out duplicate or highly similar answers based on content similarity.
        
        Args:
            research_output: ResearcherOutput containing processed research results
            
        Returns:
            Dict mapping category names to lists of unique tavily_answer strings
            Example: {
                "World-Building": ["answer1", "answer2", ...],
                "Character Development": ["answer1", "answer2", ...],
                ...
            }
        """
        category_answers = {}
        
        # Group queries by category first
        queries_by_category = {}
        for query in research_output.updated_queries:
            category = query.category or "Unknown"
            if category not in queries_by_category:
                queries_by_category[category] = []
            queries_by_category[category].append(query)
        
        # For each category, we need to look up the corresponding results
        # Since we don't have direct access to the detailed results here,
        # we'll need to search ChromaDB for the stored chunks
        for category, queries in queries_by_category.items():
            answers = []
            
            try:
                # Query ChromaDB for chunks related to this category
                chroma_results = self.chroma_collection.get(
                    where={
                        "research_id": research_output.research_id,
                        "query_category": category
                    },
                    include=["metadatas", "documents"]
                )
                
                if chroma_results and chroma_results['metadatas']:
                    # Extract unique answers from stored metadata or reconstruct from documents
                    seen_sources = set()
                    
                    for i, metadata in enumerate(chroma_results['metadatas']):
                        source_url = metadata.get('source_url', '')
                        source_title = metadata.get('source_title', '')
                        
                        # Create a unique identifier for this source
                        source_key = f"{source_url}:{source_title}"
                        
                        if source_key not in seen_sources:
                            seen_sources.add(source_key)
                            
                            # Use the document content as the "answer"
                            if i < len(chroma_results['documents']):
                                document_content = chroma_results['documents'][i]
                                
                                # Check if this content is similar to any existing answer
                                is_duplicate = any(
                                    self.is_similar_content(document_content, existing_answer)
                                    for existing_answer in answers
                                )
                                
                                if not is_duplicate and document_content:
                                    answers.append(document_content)
                
            except Exception as e:
                logger.warning(f"Failed to extract results for category '{category}': {e}")
                # Fallback: use a generic message
                answers.append(f"Research completed for {category} but detailed results unavailable")
            
            category_answers[category] = answers
        
        return category_answers
    
    def extract_refined_search_results_from_raw_data(
        self, 
        research_results: Dict[str, List[ResearchResults]]
    ) -> Dict[str, List[str]]:
        """
        Extract refined search results directly from raw ResearchResults data.
        This method works with the internal results format before they're stored.
        
        Args:
            research_results: Dict mapping query strings to lists of ResearchResults
            
        Returns:
            Dict mapping category names to lists of unique tavily_answer strings
        """
        category_answers = {}
        
        for query_string, result_list in research_results.items():
            for research_result in result_list:
                # Extract category from the research result
                category = research_result.query or "Unknown"
                
                if category not in category_answers:
                    category_answers[category] = []
                
                # Extract tavily answers from search results
                for search_result in research_result.search_results:
                    if search_result.tavily_answer:
                        tavily_answer = search_result.tavily_answer
                        
                        # Check if this answer is similar to any existing answer in this category
                        is_duplicate = any(
                            self.is_similar_content(tavily_answer, existing_answer)
                            for existing_answer in category_answers[category]
                        )
                        
                        if not is_duplicate:
                            category_answers[category].append(tavily_answer)
        
        return category_answers
    
    async def get_research_summary_by_category(self, research_id: str) -> Dict[str, List[str]]:
        """
        Get a refined summary of research results organized by category.
        This method queries the stored ChromaDB data and returns deduplicated content.
        
        Args:
            research_id: Research session identifier
            
        Returns:
            Dict mapping category names to lists of unique content strings
        """
        category_answers = {}
        
        try:
            # Query all chunks for this research session
            results = self.chroma_collection.get(
                where={"research_id": research_id},
                include=["metadatas", "documents"]
            )
            
            if not results['metadatas']:
                return {}
            
            # Group by category
            for i, metadata in enumerate(results['metadatas']):
                category = metadata.get('query_category', 'Unknown')
                
                if category not in category_answers:
                    category_answers[category] = []
                
                # Get the document content
                if i < len(results['documents']):
                    document_content = results['documents'][i]
                    
                    # Only include substantial content (more than 50 characters)
                    if len(document_content.strip()) > 50:
                        # Check for similarity with existing content in this category
                        is_duplicate = any(
                            self.is_similar_content(document_content, existing_content)
                            for existing_content in category_answers[category]
                        )
                        
                        if not is_duplicate:
                            category_answers[category].append(document_content.strip())
            
            # Sort answers by length (longer, more detailed answers first)
            for category in category_answers:
                category_answers[category].sort(key=len, reverse=True)
            
            return category_answers
            
        except Exception as e:
            logger.error(f"Failed to get research summary for {research_id}: {e}")
            return {}


# Helper function to create a properly configured researcher component
def create_researcher_component(
    component_name: str = "Research Agent",
    researcher_config: Optional[ResearcherConfig] = None
) -> ResearcherComponent:
    """Create a researcher component with proper configuration."""
    
    if researcher_config is None:
        researcher_config = ResearcherConfig()
    
    component_config = ComponentConfiguration[ResearcherConfig](
        component_type=ComponentType.MARKET_INTELLIGENCE,  # Closest matching type
        component_name=component_name,
        specific_config=researcher_config
    )
    
    return ResearcherComponent(component_config)


# Example usage and registration
def register_researcher_component():
    """Register the researcher component type with the global registry."""
    from musequill.v3.components.base.component_interface import component_registry
    
    component_registry.register_component_type("researcher", ResearcherComponent)
    return True