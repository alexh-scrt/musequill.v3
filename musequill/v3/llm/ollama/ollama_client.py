import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
import time
# Import langchain and ollama
from langchain_ollama import OllamaLLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from .ollama_config import OllamaConfig

logger = logging.getLogger(__name__)

# ============================================================================
# LLM Service Integration
# ============================================================================

class LLMService:
    """Service for LLM communication via Ollama."""
    
    def __init__(
        self,
        ollama_config: OllamaConfig,
    ):
        self.base_url: str = ollama_config.base_url
        self.model_name: str = ollama_config.model_name
        self.temperature: float = 0.3
        self.max_tokens: Optional[int] = None
        self.top_p: float = 1.0
        self.top_k: Optional[int] = None
        self.num_ctx: Optional[int] = None
        self.num_predict: Optional[int] = None
        self.repeat_penalty: Optional[float] = None
        self.seed: int = 42
        self.stop: Optional[str] = None
        self.llm: Optional[OllamaLLM] = None

    # ----------------------------
    # String / Debug Representations
    # ----------------------------

    def __str__(self) -> str:
        """User-friendly string (for print/logging)."""
        status = "initialized" if self.llm else "not initialized"
        return (
            f"LLMService(model={self.model_name}, base_url={self.base_url}, "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"max_tokens={self.max_tokens}, status={status})"
        )

    def __repr__(self) -> str:
        """Developer-friendly string (for debugging)."""
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, base_url={self.base_url!r}, "
            f"temperature={self.temperature!r}, max_tokens={self.max_tokens!r}, "
            f"top_p={self.top_p!r}, top_k={self.top_k!r}, "
            f"repeat_penalty={self.repeat_penalty!r}, stop={self.stop!r}, "
            f"num_ctx={self.num_ctx!r}, num_predict={self.num_predict!r}, "
            f"llm={'set' if self.llm else 'None'})"
        )

    async def initialize(self, *args, **kwargs):
        """Initialize LLM connection."""
        try:
            # Prepare initialization parameters
            init_params = {
                "model": self.model_name,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            if self.top_k is not None:
                init_params["top_k"] = self.top_k
            if self.repeat_penalty is not None:
                init_params["repeat_penalty"] = self.repeat_penalty
            if self.stop is not None:
                init_params["stop"] = self.stop
            if self.num_ctx is not None:
                init_params["num_ctx"] = self.num_ctx
            if self.num_predict is not None:
                init_params["num_predict"] = self.num_predict
            if self.seed is not None:
                init_params["seed"] = self.seed

            # Only add max_tokens if it's specified (some models don't support it)
            if self.max_tokens is not None:
                init_params["num_predict"] = self.max_tokens  # Ollama uses 'num_predict'

            # Merge in any extra keyword args (e.g., response_format)
            init_params.update(kwargs)

            self.llm = OllamaLLM(**init_params)
            logger.info(
                f"✅  LLM service initialized with params: {init_params}"
            )
        except Exception as e:
            logger.error(f"🔴  Failed to initialize LLM: {e}")
            raise


    async def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate response from LLM with optional parameter overrides.
        
        Args:
            prompt: The input prompt
            temperature: Override temperature for this request
            max_tokens: Override max_tokens for this request  
            top_p: Override top_p for this request
            
        Returns:
            Dict containing response and metadata
        """
        try:
            if not self.llm:
                logger.error("🔴 LLM is not initialized")
                raise RuntimeError("LLM is not initialized")
            
            # Use per-request parameters if provided, otherwise use instance defaults
            request_temperature = temperature if temperature is not None else self.temperature
            request_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            request_top_p = top_p if top_p is not None else self.top_p
            
            # For per-request parameter changes, we need to update the LLM instance
            # Note: langchain-ollama might not support dynamic parameter changes
            # In that case, you might need to create a new instance or use the Ollama API directly
            if (request_temperature != self.temperature or 
                request_max_tokens != self.max_tokens or 
                request_top_p != self.top_p):
                
                logger.info(f"🟢  Using custom parameters for this request: "
                           f"temperature={request_temperature}, "
                           f"max_tokens={request_max_tokens}, "
                           f"top_p={request_top_p}")
                
                # Create temporary LLM instance with custom parameters
                temp_params = {
                    "model": self.model_name,
                    "base_url": self.base_url,
                    "temperature": request_temperature,
                    "top_p": request_top_p
                }
                
                if request_max_tokens is not None:
                    temp_params["num_predict"] = request_max_tokens
                
                temp_llm = OllamaLLM(**temp_params)
                llm_to_use = temp_llm
            else:
                llm_to_use = self.llm
            
            # Start timing
            start_time = time.perf_counter()
            response = await asyncio.to_thread(llm_to_use.invoke, prompt)
            end_time = time.perf_counter()
            
            # Calculate elapsed time in seconds
            elapsed_time = end_time - start_time
            
            return {
                "response": response,
                "timelapse": elapsed_time,
                "parameters_used": {
                    "base_url": self.base_url,
                    "model": self.model_name,
                    "temperature": request_temperature,
                    "max_tokens": request_max_tokens,
                    "top_p": request_top_p,
                    "top_k": self.top_k,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                    "seed": self.seed,
                    "repeat_penalty": self.repeat_penalty,
                    "stop": self.stop
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"🔴  JSON parsing error in concept analysis: {e}")
            return {
                "error": f"Invalid response: {e}"
            }
        except Exception as e:
            logger.error(f"🔴  Error in concept analysis: {e}")
            return {
                "error": str(e)
            }
    
    async def update_default_parameters(
        self, 
        *args,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[str] = None,
        stop: Optional[str] = None,
        **kwargs
    ):
        """
        Update the default parameters for future requests.
        This will require re-initialization to take effect.
        
        Args:
            temperature: New default temperature
            max_tokens: New default max_tokens
            top_p: New default top_p
        """
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if repeat_penalty is not None:
            self.repeat_penalty = repeat_penalty
        if num_ctx is not None:
            self.num_ctx = min(32768, num_ctx)
        if num_predict is not None:
            self.num_predict = min(4096, num_predict)
        if stop is not None:
            self.stop = stop
        if seed is not None:
            self.seed = seed

        logger.info("Calling initialize() again to apply these changes to the LLM instance")

        # Build kwargs with response_format and any additional kwargs
        extra_kwargs = dict(kwargs)
        if response_format is not None:
            extra_kwargs["response_format"] = response_format

        await self.initialize(**extra_kwargs)
        
        logger.info(f"Updated default parameters: temperature={self.temperature}, "
                   f"max_tokens={self.max_tokens}, top_p={self.top_p}, num_prodict={self.num_predict}, seed={self.seed},"
                   f"top_k={self.top_k}, num_ctx={self.num_ctx} repeat_penalty={self.repeat_penalty}, stop={self.stop}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get the current parameter configuration."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
            "stop": self.stop,
            "is_initialized": self.llm is not None
        }
    
def create_llm_service(ollama_config: Optional[OllamaConfig] = None) -> LLMService:
    """Create and return an LLMService instance."""
    if not ollama_config:
        ollama_config = OllamaConfig()
    return LLMService(ollama_config)
