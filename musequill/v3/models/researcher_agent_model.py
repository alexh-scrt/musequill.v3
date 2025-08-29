from pydantic import BaseModel, Field, field_validator
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union, cast, Iterable
import json


class QueryStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    @classmethod
    def values(cls):
        return [e.value for e in cls]
    @classmethod
    def default(cls):
        return cls.PENDING
    def is_pending(self):
        return self.value == QueryStatus.PENDING
    def is_completed(self):
        return self.value == QueryStatus.COMPLETED
    def is_failed(self):
        return self.value == QueryStatus.FAILED
    def __str__(self):
        return self.value
    def __repr__(self):
        return self.value
    def __bool__(self):
        return self.value == self.PENDING
    def __getitem__(self, index):
        return self.value[index]
    def display(self):
        return self.value


@dataclass
class SearchResult:
    """Structured search result from Tavily."""
    url: str
    title: str
    content: str
#    raw_content: str
    score: float
    published_date: Optional[str]
    domain: str
    query: str
    tavily_answer: Optional[str] = None


@dataclass
class ProcessedChunk:
    """Processed content chunk ready for vector storage."""
    chunk_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    quality_score: float
    source_info: Dict[str, str]


@dataclass
class ResearchResults:
    """Complete research results for a query."""
    query: str
    search_results: List[SearchResult]
    processed_chunks: List[ProcessedChunk]
    total_chunks_stored: int
    total_sources: int
    quality_stats: Dict[str, QueryStatus]
    execution_time: float
    status: str
    error_message: Optional[str] = None

class ResearchQueryEx(BaseModel):
    query_type: QueryStatus = QueryStatus.default()
    context: Optional[str] = None
    topic: Optional[str] = None
    question: Optional[str] = None
    sources_suggested: Optional[List[str]] = None
    category: Optional[str] = None
    priority: Optional[int] = Field(default=1, description='Research Priority')

    def get_query(self) -> str:
        """Get the query string for Tavily."""
        def build_query(include_sources=True, include_category=True, include_context=True, include_topic=True, question=None):
            parts = []
            if include_context and self.context:
                parts.append(f"**CONTEXT**:[{self.context}].")
            if include_sources and self.sources_suggested:
                parts.append(f"**SOURCES**: [{', '.join(self.sources_suggested)}].")
            if include_category and self.category:
                parts.append(f"**CATEGORY**: [{self.category}].")
            if include_topic and self.topic:
                parts.append(f"**TOPIC**: [{self.topic}].")
            if question or self.question:
                parts.append(f"**WHAT TO RESEARCH**: [{question or self.question}]")
            return "\n".join(parts)

        # Try full query first
        query_str = build_query()
        if len(query_str) <= 400:
            return query_str

        # Drop SOURCES first
        query_str = build_query(include_sources=False)
        if len(query_str) <= 400:
            return query_str

        # Drop CATEGORY next
        query_str = build_query(include_sources=False, include_category=False)
        if len(query_str) <= 400:
            return query_str

        # Drop CONTEXT next
        query_str = build_query(include_sources=False, include_category=False, include_context=False)
        if len(query_str) <= 400:
            return query_str

        # Drop TOPIC next
        query_str = build_query(include_sources=False, include_category=False, include_context=False, include_topic=False)
        if len(query_str) <= 400:
            return query_str

        # Last resort: shorten the question semantically
        if self.question and len(self.question) > 350:  # Leave room for formatting
            # Try to keep the core meaning by taking first part up to last complete sentence/clause
            shortened = self.question[:350]
            # Find the last period, question mark, or exclamation mark
            last_punct = max(shortened.rfind('.'), shortened.rfind('?'), shortened.rfind('!'))
            if last_punct > 100:  # Only truncate at sentence boundary if it's not too short
                shortened = shortened[:last_punct + 1]
            else:
                # Otherwise, truncate at word boundary
                last_space = shortened.rfind(' ')
                if last_space > 50:
                    shortened = shortened[:last_space] + "..."
                else:
                    shortened = shortened + "..."
            
            query_str = build_query(include_sources=False, include_category=False, 
                                  include_context=False, include_topic=False, question=shortened)
        
        return query_str

    def to_dict(self, include_none: bool = False) -> dict:
        """Convert ResearchQueryEx instance to dictionary"""
        result = {}
        for field_name, field_value in [
            ('query_type', self.query_type.value if isinstance(self.query_type, QueryStatus) else self.query_type),
            ('context', self.context),
            ('topic', self.topic),
            ('question', self.question),
            ('sources_suggested', self.sources_suggested),
            ('category', self.category)
        ]:
            if field_value is not None or include_none:
                result[field_name] = field_value
        return result
    
    def to_json(self, include_none: bool = False) -> str:
        """Convert ResearchQueryEx instance to JSON string"""
        return json.dumps(self.to_dict(include_none=include_none), indent=2)

    def __str__(self) -> str:
        """String representation of ResearchQueryEx"""
        return self.to_json(include_none=True)

class ResearchQuery(BaseModel):
    query_type: QueryStatus = QueryStatus.default()
    category: Optional[str] = None
    topic: Optional[str] = None
    description: Optional[str] = None
    queries: List[str] = []
    priority: Optional[str] = None
    estimated_time: Optional[Union[str, int]] = None
    research_methods: Optional[List[str]] = None
    key_questions: Optional[List[str]] = None
    sources_suggested: Optional[List[str]] = None

    @field_validator('estimated_time', mode='before')
    @classmethod
    def validate_estimated_time(cls, v):
        """Accept both string (e.g., '20 hours') and int values for estimated_time"""
        if v is None:
            return v
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return v
        # Try to convert other types to string
        return str(v)

    def get_query(self) -> str:
        """Get the query string for Tavily."""
        return f"""
**CONTEXT**:[TOPIC RESEARCH].
**SOURCES**: [{", ".join(self.sources_suggested)}].
**CATEGORY**: [{self.category}].
**TOPIC**: [{self.topic}].
**WHAT TO RESEARCH**: [{';'.join(self.queries)}]
"""

    def get_questions(self) -> Optional[List[str]]:
        """Get the questions string for Tavily."""
        
        qq: List[str] = self.key_questions or []
        return [
f"""
**TOPIC**: [{self.topic}].
**QUESTION**: [{q}]
"""
        for q in qq]

    @classmethod
    def from_json(cls, json_data: str) -> 'ResearchQuery':
        """Create a ResearchQuery instance from JSON string"""
        data = json.loads(json_data)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ResearchQuery':
        """Create a ResearchQuery instance from dictionary"""
        # Set default query_type if not provided
        if 'query_type' not in data:
            data['query_type'] = QueryStatus.default()
        
        # For Pydantic models, use model_fields instead of __dataclass_fields__
        valid_fields = set(cls.model_fields.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    def to_dict(self, include_none: bool = False) -> dict:
        """Convert ResearchQuery instance to dictionary"""
        result = {}
        for field_name, field_value in [
            ('query_type', self.query_type.value if isinstance(self.query_type, QueryStatus) else self.query_type),
            ('category', self.category),
            ('topic', self.topic),
            ('description', self.description),
            ('priority', self.priority),
            ('estimated_time', self.estimated_time),
            ('research_methods', self.research_methods),
            ('key_questions', self.key_questions),
            ('sources_suggested', self.sources_suggested)
        ]:
            if field_value is not None or include_none:
                result[field_name] = field_value
        return result
    
    def to_json(self, include_none: bool = False) -> str:
        """Convert ResearchQuery instance to JSON string"""
        return json.dumps(self.to_dict(include_none=include_none), indent=2)
    
    @classmethod
    def load_research_queries(cls, json_data: str) -> List[Any]:
        """
        Load a list of ResearchQuery instances from JSON data.
        
        Args:
            json_data: JSON string containing research topics data
            
        Returns:
            List of ResearchQuery instances
            
        Raises:
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If the expected 'research_topics' key is not found
        """
        data = json.loads(json_data)
        
        # Handle case where JSON might be a direct list or wrapped in 'research_topics'
        if isinstance(data, list):
            topics_list = data
        elif 'research_topics' in data:
            topics_list = data['research_topics']
        else:
            raise KeyError("Expected 'research_topics' key in JSON data or direct list of topics")
        
        return [ResearchQuery.from_dict(topic) for topic in topics_list]
