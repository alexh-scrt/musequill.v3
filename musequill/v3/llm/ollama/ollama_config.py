"""
Configuration management for the ollama client
"""

from pydantic import Field
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class OllamaConfig(BaseSettings):
    """Configuration settings for the Ollama client."""

    # Ollama Embeddings settings
    base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Ollama server base URL"
    )

    model_name: str = Field(
        default="llama3.3:70b",
        validation_alias="OLLAMA_MODEL_NAME",
        description="Ollama model"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __str__(self) -> str:
        """User-friendly string (for print)."""
        return f"OllamaConfig(base_url={self.base_url}, model_name={self.model_name})"

    def __repr__(self) -> str:
        """Developer-friendly string (for debugging)."""
        # best practice: make __repr__ unambiguous
        return (
            f"{self.__class__.__name__}"
            f"(base_url={self.base_url!r}, model_name={self.model_name!r})"
        )
