"""
Researcher Agent

Executes research queries using Tavily web search and stores results in Chroma vector database.
Processes and chunks content with quality filtering and deduplication.

Key Features:
- Web search using Tavily API with advanced search capabilities
- Content processing with intelligent chunking and quality filtering
- Vector storage in hosted Chroma database with comprehensive metadata
- Concurrent query processing with rate limiting and retry logic
- Content deduplication and similarity filtering
- Source quality assessment and domain filtering
- Comprehensive error handling and monitoring
- Ollama embeddings integration for local embeddings generation
"""

import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, cast, Iterable
from urllib.parse import urlparse
from uuid import uuid4
from dataclasses import dataclass
import traceback
import logging
from hashlib import sha256
import chromadb
from chromadb.errors import NotFoundError
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Changed from OpenAI to Ollama
from tavily import TavilyClient
from difflib import SequenceMatcher
if __name__ == '__main__':
    import sys
    from pathlib import Path
    project_root = Path(__name__).parent.parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    current_path = Path(__name__).parent
    sys.path.insert(1, str(current_path))

from musequill.services.backend.researcher import (
    ResearcherConfig,
    ResearchQuery,
    QueryStatus,
    ResearchResults,
    SearchResult,
    ProcessedChunk
)


logger = logging.getLogger(__name__)


class ResearcherAgent:
    """
    Researcher Agent that executes research queries and stores results in vector database.
    """
    
    def __init__(self, config: Optional[ResearcherConfig] = None):
        if not config:
            config = ResearcherConfig()
        
        self.config = config
        
        # Initialize clients
        self.tavily_client: Optional[TavilyClient] = None
        self.chroma_client: Optional[chromadb.HttpClient] = None
        self.chroma_collection = None
        self.embeddings: Optional[OllamaEmbeddings] = None  # Changed type annotation
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
        
        self._initialize_components()
        
        logger.info("Researcher Agent initialized with Ollama embeddings")
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Initialize Tavily client
            if self.config.tavily_api_key:
                self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
                logger.info("✅  Tavily client initialized")
            else:
                logger.error("🛑  Tavily API key not provided")
                raise ValueError("Tavily API key is required")
            
            # Initialize Chroma client
            self.chroma_client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
                settings=Settings(
                    chroma_server_authn_credentials=None,
                    chroma_server_authn_provider=None
                )
            )
            
            # Handle collection creation/recreation with dimension check
            self.chroma_collection = self._get_or_create_collection_safe(self.config, drop_if_exists=True)

            ollama_base_url = getattr(self.config, 'ollama_base_url', 'http://localhost:11434')
            embedding_model = getattr(self.config, 'embedding_model', 'nomic-embed-text')
            
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=ollama_base_url
            )
            
            logger.info(f"✅  Ollama embeddings initialized with model: {embedding_model}")
            
            # Initialize other components
            # from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            logger.info("✅  All components initialized successfully")
            
        except Exception as e:
            logger.error(f"🛑  Failed to initialize components: {e}")
            raise

    def _get_or_create_collection_safe(self, config: ResearcherConfig, drop_if_exists:bool = False):
        """
        Safely get or create ChromaDB collection, handling dimension mismatches.
        """
        collection_name = config.chroma_collection_name
        expected_embedding_model = getattr(config, 'embedding_model', 'nomic-embed-text')
        # Get expected dimensions for current model
        model_dimensions = {
            'nomic-embed-text': 768,
            'mxbai-embed-large': 1024,
            'all-minilm': 384,
            'text-embedding-ada-002': 1536,  # OpenAI model
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        
        try:
            # Try to get existing 
            if drop_if_exists:
                self.chroma_client.delete_collection(name=collection_name)

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
            
        except Exception or NotFoundError as e:
            # Collection doesn't exist, create new one
            logger.info(f"Creating new collection '{collection_name}': {e}")
            expected_dims = model_dimensions.get(expected_embedding_model, 768)
            return self._create_new_collection(config, expected_embedding_model, expected_dims)
    
    def _create_new_collection(self, config: ResearcherConfig, embedding_model: str, dimensions: int):
        """Create a new ChromaDB collection with proper metadata."""
        collection = self.chroma_client.create_collection(
            name=config.chroma_collection_name,
            metadata={
                "description": "Research materials for book writing",
                "embedding_model": embedding_model,
                "embedding_dimensions": dimensions,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "migration_from": "openai_to_local" if dimensions == 768 else "new_collection"
            }
        )
        logger.info(f"Created new collection '{config.chroma_collection_name}' "
                   f"for {embedding_model} ({dimensions} dimensions)")
        return collection



    async def execute_research(self, research_id:str, queries: List[ResearchQuery]) -> Dict[str, Any]:
        """
        Execute research for all pending queries in the state.
        
        Args:
            research_id: str - a unique id to associate the query results with
            queries: containing a list of research queries
            
        Returns:
            Dictionary with research results and updated state information
        """
        try:
            logger.info(f"Starting research execution for book {research_id}")
            self.stats['processing_start_time'] = time.time()
            
            # Filter pending queries
            pending_queries = [q for q in queries if q.query_type.is_pending()]
            
            if not pending_queries:
                logger.warning(f"No pending research queries found for book {research_id}")
                return {
                    'updated_queries': queries,
                    'total_chunks': 0,
                    'total_sources': 0,
                    'stats': self.stats,
                    'chroma_storage_info': self._get_chroma_storage_info(research_id)
                }
            
            len_pending_queries = len(pending_queries)
            pending_queries_idx = len_pending_queries if len_pending_queries < self.config.max_research_queries else self.config.max_research_queries
            pending_queries = pending_queries[:pending_queries_idx]
            logger.info(f"Processing {len(pending_queries)} research queries")
            
            # Execute research queries with concurrency control
            research_results = await self._execute_queries_concurrently(pending_queries, research_id)
            
            # Update query statuses
            updated_queries = self._update_query_statuses(queries, research_results)
            
            # Calculate totals
            total_chunks = sum(result.total_chunks_stored for result_list in research_results.values() for result in result_list)
            total_sources = sum(result.total_sources for result_list in research_results.values() for result in result_list)
            
            # Update statistics
            self.stats['total_chunks_stored'] += total_chunks
            self.stats['total_sources_processed'] += total_sources
            
            execution_time = time.time() - self.stats['processing_start_time']
            
            # Prepare ChromaDB storage information for state
            chroma_storage_info = self._get_chroma_storage_info(research_id)
            
            logger.info(
                f"Research execution completed for book {research_id}: "
                f"{total_chunks} chunks stored, {total_sources} sources processed, "
                f"execution time: {execution_time:.2f}s"
            )
            
            return {
                'updated_queries': updated_queries,
                'total_chunks': total_chunks,
                'total_sources': total_sources,
                'execution_time': execution_time,
                'stats': self.stats,
                'detailed_results': research_results,
                'chroma_storage_info': chroma_storage_info
            }
            
        except Exception as e:
            logger.error(f"Error executing research for book {research_id}: {e}")
            logger.error(traceback.format_exc())
            self.stats['queries_failed'] += len(pending_queries)
            
            # Return failure result with ChromaDB storage info
            return {
                'updated_queries': self._mark_queries_failed(queries, str(e)),
                'total_chunks': 0,
                'total_sources': 0,
                'error': str(e),
                'stats': self.stats,
                'chroma_storage_info': self._get_chroma_storage_info(research_id)
            }

    def _sort_research_queries_by_priority(self, research_queries:List[ResearchQuery]) -> List[ResearchQuery]:
        """
        Sort ResearchQuery list by priority: High -> Medium -> Low
        
        Args:
            research_queries: List of ResearchQuery objects
            
        Returns:
            List of ResearchQuery objects sorted by priority
        """
        # Define priority order mapping
        priority_order = {
            'High': 1,
            'Medium': 2, 
            'Low': 3
        }
        
        # Sort using the priority mapping
        sorted_queries = sorted(
            research_queries, 
            key=lambda query: priority_order.get(query.priority, 4)  # Default to 4 for unknown priorities
        )
        
        return sorted_queries

    async def _execute_queries_concurrently(self, queries: List[ResearchQuery], research_id: str) -> Dict[str, List[ResearchResults]]:
        """
        Execute research queries with controlled concurrency.
        
        Args:
            queries: List of research queries to execute
            research_id: Research identifier for metadata
            
        Returns:
            Dictionary mapping query text to ResearchResults
        """
        results: Dict[str, List[ResearchResults]] = {}
        
        # Process queries in batches to respect concurrency limits
        batch_size = self.config.max_concurrent_queries
        
        queries = self._sort_research_queries_by_priority(queries)

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} queries")
            
            # Execute batch concurrently using await
            batch_results = await self._execute_query_batch(batch, research_id)
            results.update(batch_results)
            
            # Rate limiting between batches
            if i + batch_size < len(queries):
                await asyncio.sleep(self.config.rate_limit_delay)  # Use async sleep
        
        return results


    def _execute_queries_concurrently_old(self, queries: List[ResearchQuery], research_id: str) -> Dict[str, ResearchResults]:
        """
        Execute research queries with controlled concurrency.
        
        Args:
            queries: List of research queries to execute
            research_id: Research identifier for metadata
            
        Returns:
            Dictionary mapping query text to ResearchResults
        """
        results = {}
        
        # Process queries in batches to respect concurrency limits
        batch_size = self.config.max_concurrent_queries
        
        queries = self._sort_research_queries_by_priority(queries)

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} queries")
            
            # Execute batch concurrently
            batch_results = asyncio.run(self._execute_query_batch(batch, research_id))
            results.update(batch_results)
            
            # Rate limiting between batches
            if i + batch_size < len(queries):
                time.sleep(self.config.rate_limit_delay)
        
        return results
    
    async def _execute_query_batch(self, queries: List[ResearchQuery], research_id: str) -> Dict[str, List[ResearchResults]]:
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
        batch_results:Dict[str, List[ResearchResults]] = {}
        try:
            for results in task_results:
                if isinstance(results, Exception):
                    logger.error(f"Query failed: {results}")
                    self.stats['queries_failed'] += 1
                else:
                    if isinstance(results, list):
                        for result in cast(Iterable[ResearchResults], results):
                            batch_results.setdefault(result.query, []).append(result)
                            self.stats['queries_processed'] += 1
        except Exception as e:
            logger.error(f'Failed to batch process: {str(e)}')
        return batch_results
    
    async def _research_single_query(self, query: ResearchQuery, research_id: str) -> List[ResearchResults]:
        """
        Research a single query with retries.
        
        Args:
            query: Research query to execute
            research_id: Research identifier
            
        Returns:
            ResearchResults for the query
        """
        results:List[ResearchResults] = []
        start_time = time.time()
        last_error = None
        main_query:str = query.get_query()
        research_queries:List[str] = query.get_questions()
        research_queries.append(main_query)
        for _query in research_queries:
            for attempt in range(self.config.query_retry_attempts):
                try:
                    logger.info(f"Executing query [{_query}] (attempt {attempt + 1})")
                    
                    # Perform web search
                    search_results = await self._perform_web_search(_query)
                    
                    # Filter and validate results
                    filtered_results = self._filter_search_results(search_results)
                    
                    # Process content and create chunks
                    processed_chunks = await self._process_search_results(filtered_results, _query, research_id)
                    
                    # Store chunks in vector database
                    chunks_stored = await self._store_chunks_in_chroma(processed_chunks, research_id)
                    
                    # Calculate quality statistics
                    quality_stats = self._calculate_quality_stats(processed_chunks)
                    
                    execution_time = time.time() - start_time
                    
                    result = ResearchResults(
                        query=query.category,
                        search_results=filtered_results,
                        processed_chunks=[],
                        total_chunks_stored=chunks_stored,
                        total_sources=len(filtered_results),
                        quality_stats=quality_stats,
                        execution_time=execution_time,
                        status='completed'
                    )
                    
                    logger.info(
                        f"Query [{query.get_query()}]' completed: {chunks_stored} chunks stored, "
                        f"{len(filtered_results)} sources processed in {execution_time:.2f}s"
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Query [{query.get_query()}] attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.config.query_retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay_seconds)

        if len(results) > 0:
            return results

        # All attempts failed
        execution_time = time.time() - start_time
        logger.error(f"Query [{main_query}]' failed after {self.config.query_retry_attempts} attempts")
        
        return [ResearchResults(
            query='; '.join(research_queries),
            search_results=[],
            processed_chunks=[],
            total_chunks_stored=0,
            total_sources=0,
            quality_stats={},
            execution_time=execution_time,
            status='failed',
            error_message=str(last_error)
        )]
    
    async def _perform_web_search(self, query: str) -> List[SearchResult]:
        """
        Perform web search using Tavily API.
        
        Args:
            query: Search query string
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Execute search with Tavily
            search_response = self.tavily_client.search(
                query=query,
                search_depth=self.config.tavily_search_depth,
                max_results=self.config.tavily_max_results,
                include_answer=self.config.tavily_include_answer,
                include_raw_content=self.config.tavily_include_raw_content,
                include_images=self.config.tavily_include_images
            )
            
            # Process search results
            search_results = []
            tavily_answer = search_response.get('answer', '') if self.config.tavily_include_answer else None
            
            for result in search_response.get('results', []):
                # Extract domain
                domain = urlparse(result.get('url', '')).netloc.lower()
                
                search_result = SearchResult(
                    url=result.get('url', ''),
                    title=result.get('title', ''),
                    content=result.get('content', ''),
                    # raw_content=result.get('raw_content', ''),
                    score=result.get('score', 0.0),
                    published_date=result.get('published_date'),
                    domain=domain,
                    query=query,
                    tavily_answer=tavily_answer
                )
                
                search_results.append(search_result)
            
            if self.config.log_search_results:
                logger.info(f"Search for '{query}' returned {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            raise
    
    def _filter_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter search results based on quality criteria.
        
        Args:
            results: List of search results
            query: Original research query
            
        Returns:
            Filtered list of search results
        """
        filtered_results = []
        
        for result in results:
            # Check minimum score threshold
            if result.score > self.config.min_source_score:
                filtered_results.append(result)
                self.processed_urls.add(result.url)
                continue
            
            # Check domain filtering
            if self._is_domain_blocked(result.domain):
                logger.debug(f"Blocked domain: {result.domain}")
                continue
            
            # Check for duplicate URLs
            if result.url in self.processed_urls:
                logger.debug(f"Duplicate URL filtered: {result.url}")
                continue
            
            # Check content quality
            if not self._assess_content_quality(result):
                logger.debug(f"Low quality content filtered: {result.url}")
                continue
            
            filtered_results.append(result)
            self.processed_urls.add(result.url)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} high-quality sources")
        return filtered_results
    
    def _is_domain_blocked(self, domain: str) -> bool:
        """Check if domain is in blocked list."""
        for blocked in self.config.blocked_domains:
            if blocked.lower() in domain.lower():
                return True
        return False
    
    def _is_domain_trusted(self, domain: str) -> bool:
        """Check if domain is in trusted list."""
        for trusted in self.config.trusted_domains:
            if trusted.lower() in domain.lower():
                return True
        return False
    
    def _assess_content_quality(self, result: SearchResult) -> bool:
        """
        Assess content quality based on various criteria.
        
        Args:
            result: Search result to assess
            
        Returns:
            True if content meets quality standards
        """
        if not self.config.enable_content_filtering:
            return True
        
        # Use raw_content if available, otherwise fall back to content
        content = result.tavily_answer or result.content
        
        if not content or len(content.strip()) < self.config.min_chunk_size:
            return False
        
        # Check content length (not too short, not too long)
        if len(content) > self.config.max_content_length:
            content = content[:self.config.max_content_length]
        
        # Basic quality indicators
        quality_score = 0.0
        
        # Domain trust score
        if self._is_domain_trusted(result.domain):
            quality_score += 0.3
        
        # Tavily score contribution
        quality_score += result.score * 0.4
        
        # Content structure indicators
        sentence_count = len(re.findall(r'[.!?]+', content))
        if sentence_count > 3:  # Multiple sentences indicate structured content
            quality_score += 0.2
        else:
            # Count words per sentence
            word_counts = [len(re.findall(r'\w+', s)) for s in content]
            average_words_per_sentence = sum(word_counts) / len(word_counts) if word_counts else 0

            # Adjust score if average sentence length is substantial (e.g., 10+ words)
            if average_words_per_sentence >= 10:
                quality_score += 0.125  # or another weight based on your scoring logic

        # Presence of meaningful content (not just navigation/ads)
        meaningful_words = len(re.findall(r'\b[a-zA-Z]{4,}\b', content))
        if meaningful_words > 750:
            quality_score += 0.1
        
        return quality_score >= self.config.min_content_quality_score
    
    async def _process_search_results(
        self,
        results: List[SearchResult],
        query: ResearchQuery,
        research_id: str
    ) -> List[ProcessedChunk]:
        """
        Process search results into chunks with embeddings.
        
        Args:
            results: Filtered search results
            query: Original research query
            research_id: Research identifier
            
        Returns:
            List of ProcessedChunk objects
        """
        processed_chunks = []
        
        for result in results:
            try:
                # Get content (prefer raw_content)
                content = result.tavily_answer or result.content
                if not content:
                    continue
                
                # Truncate if too long
                if len(content) > self.config.max_content_length:
                    content = content[:self.config.max_content_length]
                
                # Split into chunks
                text_chunks = self.text_splitter.split_text(content)
                
                for i, chunk_text in enumerate(text_chunks):
                    if len(chunk_text.strip()) < self.config.min_chunk_size:
                        continue
                    
                    # Check for content duplication
                    if self.config.filter_duplicate_content:
                        content_hash = self._get_content_hash(chunk_text)
                        if content_hash in self.content_hashes:
                            self.stats['duplicate_content_filtered'] += 1
                            continue
                        self.content_hashes.add(content_hash)
                    
                    # Generate embedding using Ollama
                    embedding = await self.embeddings.aembed_query(chunk_text)
                    
                    # Create unique chunk ID
                    category_hash = sha256(query.category.encode()).hexdigest()
                    chunk_id = f"{research_id}_{category_hash}_{uuid4().hex[:12]}"
                    
                    # Create comprehensive metadata
                    # ChromaDB metadata must be strings, numbers, or booleans - no None values
                    metadata = {
                        'research_id': str(research_id),
                        'query': query.get_query(),
                        'query_type': str(query.query_type),
                        'query_priority': str(query.priority),
                        'source_url': str(result.url),
                        'source_title': str(result.title),
                        'source_domain': str(result.domain),
                        'source_score': float(result.score),
                        'chunk_index': int(i),
                        'chunk_size': int(len(chunk_text)),
                        'published_date': str(result.published_date) if result.published_date is not None else "",
                        'processed_at': str(datetime.now(timezone.utc).isoformat()),
                        'tavily_answer': str(result.tavily_answer[:500]) if result.tavily_answer else "",
                        'total_chunks_from_source': int(len(text_chunks)),
                        'embedding_model': str(self.embeddings.model)  # Track which model was used
                    }
                    
                    # Calculate quality score for this chunk
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
                    
                    if self.config.log_chunk_details:
                        logger.debug(f"Processed chunk {chunk_id}: {len(chunk_text)} chars, quality: {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing result from {result.url}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_chunks)} chunks from {len(results)} search results using Ollama embeddings")
        return processed_chunks
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_chunk_quality_score(self, chunk_text: str, result: SearchResult, query: ResearchQuery) -> float:
        """
        Calculate quality score for a content chunk.
        
        Args:
            chunk_text: Text content of the chunk
            result: Source search result
            query: Original research query
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Base score from Tavily
        score += result.score * 0.3
        
        # Domain trust factor
        if self._is_domain_trusted(result.domain):
            score += 0.2
        
        # Content length factor (optimal range)
        text_length = len(chunk_text)
        if 200 <= text_length <= 800:
            score += 0.15
        elif 100 <= text_length <= 1200:
            score += 0.1
        
        # Query relevance (simple keyword matching)
        query_words = set(query.get_query().lower().split())
        chunk_words = set(chunk_text.lower().split())
        relevance = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
        score += relevance * 0.2
        
        # Content structure indicators
        if len(re.findall(r'[.!?]', chunk_text)) >= 2:  # Multiple sentences
            score += 0.1
        
        # Avoid promotional/spammy content
        spam_indicators = ['click here', 'subscribe now', 'buy now', '!!!', 'free trial']
        if not any(indicator in chunk_text.lower() for indicator in spam_indicators):
            score += 0.05
        
        return min(1.0, score)  # Cap at 1.0
    
    async def _store_chunks_in_chroma(self, chunks: List[ProcessedChunk], research_id: str) -> int:
        """
        Store processed chunks in Chroma vector database.
        
        Args:
            chunks: List of processed chunks to store
            research_id: Research identifier
            
        Returns:
            Number of chunks successfully stored
        """
        if not chunks:
            return 0
        
        try:
            stored_count = 0
            
            # Process chunks in batches
            batch_size = self.config.batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    # Prepare batch data for Chroma
                    ids = [chunk.chunk_id for chunk in batch]
                    documents = [chunk.content for chunk in batch]
                    embeddings = [chunk.embedding for chunk in batch]
                    metadatas = [chunk.metadata for chunk in batch]
                    
                    # Log the research_id being stored for debugging
                    if metadatas:
                        sample_research_ids = set(m.get('research_id') for m in metadatas[:3])
                        logger.info(f"Storing batch with research_ids: {list(sample_research_ids)} (types: {[type(bid) for bid in sample_research_ids]})")
                    
                    # Store batch in Chroma
                    self.chroma_collection.add(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                    
                    stored_count += len(batch)
                    
                    logger.debug(f"Stored batch of {len(batch)} chunks in Chroma")
                    
                    # Small delay between batches to avoid overwhelming the server
                    if i + batch_size < len(chunks):
                        await asyncio.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Failed to store batch starting at index {i}: {e}")
                    # Continue with next batch rather than failing completely
                    continue
            if stored_count > 0:
                logger.info(f"Stored {stored_count} chunks in Chroma for book {research_id}")
            else:
                logger.warning(f"Did not store any chunks in Chroma for book {research_id}")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing chunks in Chroma: {e}")
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
            updated_query = query.copy()
            
            if query.get_query() in results:
                result_list = results[query.get_query()]
                if result_list:
                    # Use the first result for status and aggregate data from all results
                    first_result = result_list[0]
                    updated_query['status'] = first_result.status
                    
                    # Aggregate data from all results
                    total_chunks = sum(r.total_chunks_stored for r in result_list)
                    total_sources = sum(r.total_sources for r in result_list)
                    total_execution_time = sum(r.execution_time for r in result_list)
                    
                    updated_query['results_count'] = total_chunks
                    updated_query['execution_time'] = total_execution_time
                    updated_query['sources_processed'] = total_sources
                    
                    # Combine quality stats from all results
                    combined_quality_stats = {}
                    if result_list[0].quality_stats:
                        combined_quality_stats = result_list[0].quality_stats.copy()
                        # If there are multiple results, you may want to aggregate quality stats
                        # For now, just use the first result's quality stats
                    updated_query['quality_stats'] = combined_quality_stats
                    
                    # Check for any error messages
                    error_messages = [r.error_message for r in result_list if r.error_message]
                    if error_messages:
                        updated_query['error_message'] = '; '.join(error_messages)
            
            updated_queries.append(updated_query)
        
        return updated_queries
    
    def _mark_queries_failed(self, queries: List[ResearchQuery], error_message: str) -> List[ResearchQuery]:
        """Mark all queries as failed with error message."""
        updated_queries = []
        
        for query in queries:
            updated_query = query.copy()
            if updated_query.query_type.is_pending():
                updated_query.query_type = QueryStatus.FAILED
                updated_query['error_message'] = error_message
                updated_query['results_count'] = 0
            updated_queries.append(updated_query)
        
        return updated_queries
    
    def search_similar_content(
        self,
        query_text: str,
        research_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content in the vector database.
        
        Args:
            query_text: Text to search for
            research_id: Research identifier to filter by
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar content chunks with metadata
        """
        try:
            # Generate embedding for query using Ollama
            query_embedding = self.embeddings.embed_query(query_text)
            
            # Search in Chroma
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"research_id": research_id},
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            similar_chunks = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity (assuming cosine distance)
                    similarity = 1 - distance if distance else 0
                    
                    if similarity >= similarity_threshold:
                        similar_chunks.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity,
                            'rank': i + 1
                        })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks for query '{query_text}' using Ollama embeddings")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []
    
    def get_research_summary(self, research_id: str) -> Dict[str, Any]:
        """
        Get comprehensive research summary for a book.
        
        Args:
            research_id: Research identifier
            
        Returns:
            Research summary with statistics and metadata
        """
        try:
            # Query all chunks for this book
            results = self.chroma_collection.get(
                where={"research_id": research_id},
                include=["metadatas"]
            )
            
            if not results['metadatas']:
                return {
                    'research_id': research_id,
                    'total_chunks': 0,
                    'error': 'No research data found'
                }
            
            metadatas = results['metadatas']
            
            # Calculate summary statistics
            total_chunks = len(metadatas)
            
            # Query type distribution
            query_types = {}
            for metadata in metadatas:
                query_type = metadata.get('query_type', 'unknown')
                query_types[query_type] = query_types.get(query_type, 0) + 1
            
            # Source diversity
            unique_domains = len(set(m.get('source_domain', '') for m in metadatas))
            unique_sources = len(set(m.get('source_url', '') for m in metadatas))
            
            # Priority distribution
            priorities = {}
            for metadata in metadatas:
                priority = metadata.get('query_priority', 0)
                priorities[priority] = priorities.get(priority, 0) + 1
            
            # Time range
            processed_times = [m.get('processed_at') for m in metadatas if m.get('processed_at')]
            earliest = min(processed_times) if processed_times else None
            latest = max(processed_times) if processed_times else None
            
            # Embedding model info
            embedding_models = set(m.get('embedding_model', 'unknown') for m in metadatas)
            
            summary = {
                'research_id': research_id,
                'total_chunks': total_chunks,
                'unique_sources': unique_sources,
                'unique_domains': unique_domains,
                'query_type_distribution': query_types,
                'priority_distribution': priorities,
                'research_period': {
                    'earliest': earliest,
                    'latest': latest
                },
                'avg_chunk_size': sum(m.get('chunk_size', 0) for m in metadatas) / total_chunks if total_chunks else 0,
                'embedding_models_used': list(embedding_models)
            }
            
            logger.info(f"Generated research summary for book {research_id}: {total_chunks} chunks from {unique_sources} sources")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating research summary for book {research_id}: {e}")
            return {
                'research_id': research_id,
                'error': str(e)
            }
    
    def cleanup_book_research(self, research_id: str) -> bool:
        """
        Clean up research data for a specific book.
        
        Args:
            research_id: Research identifier
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Get all chunk IDs for this book
            results = self.chroma_collection.get(
                where={"research_id": research_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                chunk_ids = results['ids']
                
                # Delete chunks in batches
                batch_size = self.config.batch_size
                deleted_count = 0
                
                for i in range(0, len(chunk_ids), batch_size):
                    batch_ids = chunk_ids[i:i + batch_size]
                    
                    try:
                        self.chroma_collection.delete(ids=batch_ids)
                        deleted_count += len(batch_ids)
                    except Exception as e:
                        logger.error(f"Failed to delete batch of chunks: {e}")
                
                logger.info(f"Cleaned up {deleted_count} research chunks for book {research_id}")
                return deleted_count == len(chunk_ids)
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up research for book {research_id}: {e}")
            return False
    
    def _get_chroma_storage_info(self, research_id: str) -> Dict[str, Any]:
        """
        Get ChromaDB storage information for the book.
        
        Args:
            research_id: Research identifier
            
        Returns:
            Dictionary with ChromaDB storage details
        """
        try:
            # Get current chunk count for this book
            current_count = 0
            try:
                results = self.chroma_collection.get(
                    where={"research_id": research_id},
                    include=["metadatas"]
                )
                current_count = len(results['ids']) if results['ids'] else 0
            except Exception as e:
                logger.warning(f"Could not get current chunk count for book {research_id}: {e}")
            
            storage_info = {
                'collection_name': self.config.chroma_collection_name,
                'chroma_host': self.config.chroma_host,
                'chroma_port': self.config.chroma_port,
                'research_id': research_id,
                'chunks_in_collection': current_count,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'storage_type': 'chromadb',
                'embedding_model': getattr(self.config, 'embedding_model', 'nomic-embed-text'),
                'ollama_base_url': getattr(self.config, 'ollama_base_url', 'http://localhost:11434')
            }
            
            logger.debug(f"ChromaDB storage info for book {research_id}: {storage_info}")
            return storage_info
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB storage info for book {research_id}: {e}")
            # Return minimal storage info even if there's an error
            return {
                'collection_name': self.config.chroma_collection_name,
                'chroma_host': self.config.chroma_host,
                'chroma_port': self.config.chroma_port,
                'research_id': research_id,
                'chunks_in_collection': 0,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'storage_type': 'chromadb',
                'embedding_model': getattr(self.config, 'embedding_model', 'nomic-embed-text'),
                'ollama_base_url': getattr(self.config, 'ollama_base_url', 'http://localhost:11434'),
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for the researcher agent."""
        current_stats = self.stats.copy()
        
        if current_stats['processing_start_time']:
            current_stats['total_processing_time'] = time.time() - current_stats['processing_start_time']
        
        # Add collection statistics if available
        try:
            collection_count = self.chroma_collection.count()
            current_stats['total_chunks_in_collection'] = collection_count
        except:
            current_stats['total_chunks_in_collection'] = 'unavailable'
        
        # Add embedding model info
        current_stats['embedding_model'] = getattr(self.config, 'embedding_model', 'nomic-embed-text')
        current_stats['ollama_base_url'] = getattr(self.config, 'ollama_base_url', 'http://localhost:11434')
        
        return current_stats
    

async def main():
    import json
    from uuid import uuid4
    from pathlib import Path
    import sys

    project_root = Path(__name__).parent.parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    conf = ResearcherConfig()
    agent = ResearcherAgent(conf)
    with open('musequill/services/backend/outputs/research-20250801-184051.json', "r", encoding='utf-8') as f:
        json_payload = f.read()
        payload = json.loads(json_payload)
        research_id = str(uuid4())
        queries = ResearchQuery.load_research_queries(json_payload)
        results = await agent.execute_research(research_id, queries)
        print('DONE')        

if __name__ == "__main__":
    asyncio.run(main())