"""
Component Interface Models

Defines generic interfaces and lifecycle management for all pipeline components
including generators, discriminators, and orchestration elements.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, computed_field
from typing import Dict, List, Optional, Any, TypeVar, Generic, Union
from datetime import datetime
from enum import Enum
import uuid


class ComponentType(str, Enum):
    """Types of components in the pipeline."""
    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator" 
    ORCHESTRATOR = "orchestrator"
    MARKET_INTELLIGENCE = "market_intelligence"
    STORY_STATE = "story_state"
    QUALITY_CONTROLLER = "quality_controller"
    LEARNING_SYSTEM = "learning_system"


class ComponentStatus(str, Enum):
    """Status of component instances."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    RECYCLING = "recycling"
    DISPOSED = "disposed"


class ComponentHealth(str, Enum):
    """Health status of components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# Generic type variables for input/output
InputType = TypeVar('InputType', bound=BaseModel)
OutputType = TypeVar('OutputType', bound=BaseModel)
ConfigType = TypeVar('ConfigType', bound=BaseModel)


class ComponentMetrics(BaseModel):
    """Performance metrics for component instances."""
    
    total_invocations: int = Field(default=0, ge=0)
    successful_invocations: int = Field(default=0, ge=0)
    failed_invocations: int = Field(default=0, ge=0)
    average_execution_time_ms: float = Field(default=0.0, ge=0.0)
    last_execution_time_ms: Optional[float] = Field(default=None, ge=0.0)
    peak_execution_time_ms: float = Field(default=0.0, ge=0.0)
    total_execution_time_ms: float = Field(default=0.0, ge=0.0)
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_invocations == 0:
            return 0.0
        return self.successful_invocations / self.total_invocations
    
    @computed_field
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        return 1.0 - self.success_rate


class ComponentConfiguration(BaseModel, Generic[ConfigType]):
    """Base configuration for all components."""
    
    component_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    component_type: ComponentType = Field(description="Type of component")
    component_name: str = Field(description="Human-readable component name")
    version: str = Field(default="1.0.0", description="Component version")
    
    # Resource management
    max_concurrent_executions: int = Field(default=1, ge=1, le=100)
    execution_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    memory_limit_mb: Optional[int] = Field(default=None, ge=100)
    
    # Lifecycle management
    auto_recycle_after_uses: Optional[int] = Field(default=None, ge=1)
    recycle_on_error_count: Optional[int] = Field(default=5, ge=1)
    
    # Component-specific configuration
    specific_config: ConfigType = Field(description="Component-specific configuration")


class ComponentState(BaseModel):
    """Runtime state of a component instance."""
    
    component_id: str = Field(description="Component identifier")
    status: ComponentStatus = Field(default=ComponentStatus.INITIALIZING)
    health: ComponentHealth = Field(default=ComponentHealth.UNKNOWN)
    
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = Field(default=None)
    
    current_executions: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    use_count: int = Field(default=0, ge=0)
    
    metrics: ComponentMetrics = Field(default_factory=ComponentMetrics)
    
    last_error: Optional[str] = Field(default=None)
    error_history: List[str] = Field(default_factory=list, max_length=10)
    
    runtime_data: Dict[str, Any] = Field(default_factory=dict)
    
    def record_execution_start(self) -> None:
        """Record start of execution."""
        self.current_executions += 1
        self.last_activity = datetime.now()
        self.status = ComponentStatus.BUSY
    
    def record_execution_success(self, execution_time_ms: float) -> None:
        """Record successful execution completion."""
        self.current_executions = max(0, self.current_executions - 1)
        self.use_count += 1
        self.last_activity = datetime.now()
        
        # Update metrics
        self.metrics.total_invocations += 1
        self.metrics.successful_invocations += 1
        self.metrics.last_execution_time_ms = execution_time_ms
        self.metrics.peak_execution_time_ms = max(
            self.metrics.peak_execution_time_ms, execution_time_ms
        )
        self.metrics.total_execution_time_ms += execution_time_ms
        
        # Update average
        if self.metrics.total_invocations > 0:
            self.metrics.average_execution_time_ms = (
                self.metrics.total_execution_time_ms / self.metrics.total_invocations
            )
        
        # Update status
        if self.current_executions == 0:
            self.status = ComponentStatus.READY
    
    def record_execution_failure(self, error_message: str, execution_time_ms: float = 0.0) -> None:
        """Record failed execution."""
        self.current_executions = max(0, self.current_executions - 1)
        self.error_count += 1
        self.last_error = error_message
        self.last_activity = datetime.now()
        
        # Update error history
        self.error_history.append(f"{datetime.now().isoformat()}: {error_message}")
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        
        # Update metrics
        self.metrics.total_invocations += 1
        self.metrics.failed_invocations += 1
        if execution_time_ms > 0:
            self.metrics.total_execution_time_ms += execution_time_ms
            if self.metrics.total_invocations > 0:
                self.metrics.average_execution_time_ms = (
                    self.metrics.total_execution_time_ms / self.metrics.total_invocations
                )
        
        # Update status
        if self.current_executions == 0:
            self.status = ComponentStatus.ERROR if self.error_count >= 3 else ComponentStatus.READY


class BaseComponent(ABC, Generic[InputType, OutputType, ConfigType]):
    """
    Abstract base class for all pipeline components.
    
    Provides lifecycle management, metrics tracking, health monitoring,
    and unified invocation interface for orchestration.
    """
    
    def __init__(self, config: ComponentConfiguration[ConfigType]):
        self.config = config
        self.state = ComponentState(component_id=config.component_id)
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the component.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def process(self, input_data: InputType) -> OutputType:
        """
        Process input data and return output.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed output data
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform health check on component.
        
        Returns:
            True if component is healthy
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup component resources.
        
        Returns:
            True if cleanup successful
        """
        pass
    
    async def invoke(self, input_data: InputType) -> OutputType:
        """
        Unified invocation interface with metrics and error handling.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
            
        Raises:
            ComponentError: If component is not ready or execution fails
        """
        if not self._initialized:
            raise ComponentError(f"Component {self.config.component_id} not initialized")
        
        if self.state.status not in [ComponentStatus.READY, ComponentStatus.BUSY]:
            raise ComponentError(f"Component {self.config.component_id} not ready: {self.state.status}")
        
        if self.state.current_executions >= self.config.max_concurrent_executions:
            raise ComponentError(f"Component {self.config.component_id} at max concurrent executions")
        
        start_time = datetime.now()
        self.state.record_execution_start()
        
        try:
            result = await self.process(input_data)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.state.record_execution_success(execution_time)
            
            # Check if component needs recycling
            await self._check_recycling_conditions()
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Execution failed: {str(e)}"
            self.state.record_execution_failure(error_msg, execution_time)
            
            # Update health status
            if self.state.error_count >= (self.config.recycle_on_error_count or 5):
                self.state.health = ComponentHealth.UNHEALTHY
                self.state.status = ComponentStatus.ERROR
            
            raise ComponentError(error_msg) from e
    
    async def start(self) -> bool:
        """
        Start the component (initialize if needed).
        
        Returns:
            True if start successful
        """
        if not self._initialized:
            success = await self.initialize()
            if success:
                self._initialized = True
                self.state.status = ComponentStatus.READY
                self.state.health = ComponentHealth.HEALTHY
            else:
                self.state.status = ComponentStatus.ERROR
                self.state.health = ComponentHealth.UNHEALTHY
            return success
        return True
    
    async def stop(self) -> bool:
        """
        Stop the component and cleanup resources.
        
        Returns:
            True if stop successful
        """
        self.state.status = ComponentStatus.RECYCLING
        success = await self.cleanup()
        
        if success:
            self.state.status = ComponentStatus.DISPOSED
            self._initialized = False
        else:
            self.state.status = ComponentStatus.ERROR
            
        return success
    
    async def restart(self) -> bool:
        """
        Restart the component (stop then start).
        
        Returns:
            True if restart successful
        """
        await self.stop()
        return await self.start()
    
    async def perform_health_check(self) -> ComponentHealth:
        """
        Perform health check and update component health status.
        
        Returns:
            Current health status
        """
        try:
            is_healthy = await self.health_check()
            
            if is_healthy:
                if self.state.error_count == 0:
                    self.state.health = ComponentHealth.HEALTHY
                elif self.state.error_count <= 2:
                    self.state.health = ComponentHealth.DEGRADED
                else:
                    self.state.health = ComponentHealth.UNHEALTHY
            else:
                self.state.health = ComponentHealth.UNHEALTHY
                
        except Exception as e:
            self.state.health = ComponentHealth.UNHEALTHY
            self.state.last_error = f"Health check failed: {str(e)}"
        
        self.state.last_health_check = datetime.now()
        return self.state.health
    
    async def _check_recycling_conditions(self) -> None:
        """Check if component should be recycled based on configuration."""
        should_recycle = False
        
        # Check use count
        if (self.config.auto_recycle_after_uses and 
            self.state.use_count >= self.config.auto_recycle_after_uses):
            should_recycle = True
        
        # Check error count
        if (self.config.recycle_on_error_count and
            self.state.error_count >= self.config.recycle_on_error_count):
            should_recycle = True
        
        if should_recycle:
            # Signal orchestrator that recycling is needed
            self.state.runtime_data['needs_recycling'] = True
    
    def get_metrics(self) -> ComponentMetrics:
        """Get current component metrics."""
        return self.state.metrics
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            "component_id": self.config.component_id,
            "component_type": self.config.component_type.value,
            "component_name": self.config.component_name,
            "status": self.state.status.value,
            "health": self.state.health.value,
            "uptime_seconds": (datetime.now() - self.state.created_at).total_seconds(),
            "current_executions": self.state.current_executions,
            "total_executions": self.state.metrics.total_invocations,
            "success_rate": self.state.metrics.success_rate,
            "error_count": self.state.error_count,
            "last_error": self.state.last_error,
            "needs_recycling": self.state.runtime_data.get('needs_recycling', False)
        }


class ComponentError(Exception):
    """Exception raised by component operations."""
    
    def __init__(self, message: str, component_id: Optional[str] = None):
        self.message = message
        self.component_id = component_id
        super().__init__(message)


class ComponentRegistry(BaseModel):
    """Registry for managing component instances and types."""
    model_config = {"arbitrary_types_allowed": True}
    registered_types: Dict[str, type] = Field(default_factory=dict)
    active_components: Dict[str, BaseComponent] = Field(default_factory=dict)
    
    def register_component_type(self, component_type: str, component_class: type) -> None:
        """Register a component class type."""
        self.registered_types[component_type] = component_class
    
    def create_component(self, component_type: str, config: ComponentConfiguration) -> str:
        """
        Create new component instance.
        
        Args:
            component_type: Type of component to create
            config: Component configuration
            
        Returns:
            Component ID of created instance
            
        Raises:
            ComponentError: If component type not registered
        """
        if component_type not in self.registered_types:
            raise ComponentError(f"Component type '{component_type}' not registered")
        
        component_class = self.registered_types[component_type]
        component = component_class(config)
        
        self.active_components[config.component_id] = component
        return config.component_id
    
    def get_component(self, component_id: str) -> Optional[BaseComponent]:
        """Get component instance by ID."""
        return self.active_components.get(component_id)
    
    def remove_component(self, component_id: str) -> bool:
        """Remove component from registry."""
        if component_id in self.active_components:
            del self.active_components[component_id]
            return True
        return False
    
    def list_components(self, component_type: Optional[ComponentType] = None) -> List[Dict[str, Any]]:
        """List all components, optionally filtered by type."""
        components = []
        
        for component in self.active_components.values():
            if component_type is None or component.config.component_type == component_type:
                components.append(component.get_status_summary())
        
        return components


# Global component registry instance
component_registry = ComponentRegistry()