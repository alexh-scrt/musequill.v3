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


class ResearchQuery(BaseModel):
    query_type: QueryStatus = QueryStatus.default()
    category: Optional[str] = None
    topic: Optional[str] = None
    description: Optional[str] = None
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
**WHAT TO RESEARCH**: [{self.description}]
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
            ('query_type', self.query_type),
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


# Example usage:
if __name__ == "__main__":
    # Your complete JSON data
    complete_json = '''
    {
        "category": "World-Building",
        "topic": "Mythical Creatures in Slavic Folklore",
        "description": "Research various mythical creatures from Slavic folklore to incorporate into the story, ensuring cultural accuracy and sensitivity.",
        "priority": "High",
        "estimated_time": 10,
        "research_methods": [
            "Literature review",
            "Online research"
        ],
        "key_questions": [
            "What are the most common mythical creatures in Slavic folklore?",
            "How are these creatures typically depicted and what are their characteristics?",
            "How can these creatures be adapted into the story to enhance world-building?"
        ],
        "sources_suggested": [
            "Folklore texts",
            "Academic articles",
            "Online encyclopedias"
        ]
    }
    '''
    
    # Partial JSON data (missing some fields)
    partial_json = '''
    {
        "category": "World-Building",
        "topic": "Dragon Mythology",
        "priority": "Medium"
    }
    '''
    
    # Create ResearchQuery from complete JSON
    query1 = ResearchQuery.from_json(complete_json)
    print("Complete query:")
    print(f"Topic: {query1.topic}")
    print(f"Priority: {query1.priority}")
    print(f"Estimated time: {query1.estimated_time}")
    
    # Create ResearchQuery from partial JSON
    query2 = ResearchQuery.from_json(partial_json)
    print("\nPartial query:")
    print(f"Topic: {query2.topic}")
    print(f"Priority: {query2.priority}")
    print(f"Description: {query2.description}")  # Will be None
    print(f"Research methods: {query2.research_methods}")  # Will be None
    
    # Convert back to JSON (excluding None values)
    print("\nPartial query back to JSON (no None values):")
    print(query2.to_json())
    
    # Convert back to JSON (including None values)
    print("\nPartial query back to JSON (with None values):")
    print(query2.to_json(include_none=True))