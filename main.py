#!/usr/bin/env python3
"""
MuseQuill V3 Main Pipeline Entry Point - Enhanced with Registry Pattern

This enhanced version uses the component registry pattern for better architecture,
maintainability, and configuration management.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

import asyncio
import argparse
import sys
import signal
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
# Import enhanced registry and orchestration
from musequill.v3.components.component_registry import setup_enhanced_component_system, create_enhanced_component_configurations
from musequill.v3.components.base.component_interface import component_registry, ComponentError
from musequill.v3.components.orchestration.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator

# Import pipeline configuration classes (save as separate module first)
from musequill.v3.pipeline_configuration import create_default_pipeline_configuration
from musequill.v3.components.orchestration.enhanced_pipeline_config import (
    EnhancedPipelineOrchestratorConfig,
    create_enhanced_pipeline_configuration_from_dict
)

# Import activity logger (save PipelineActivityLogger as a separate module first)
from musequill.v3.pipeline_activity_logger import PipelineActivityLogger, setup_pipeline_logging

# Import pipeline integration utilities
from musequill.v3.pipeline_integration import save_pipeline_results

# Color support for console output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Create mock color objects
    class MockColor:
        def __getattr__(self, name): return ""
    Fore = Back = Style = MockColor()

load_dotenv(find_dotenv())

class EnhancedPipelineRunner:
    """Enhanced pipeline runner using component registry pattern."""
    
    def __init__(self, config: Dict[str, Any], activity_logger: PipelineActivityLogger):
        self.config = config
        self.activity_logger = activity_logger
        self.orchestrator: Optional[EnhancedPipelineOrchestrator] = None
        self.logger = logging.getLogger('pipeline_runner')
        
    async def initialize_pipeline(self) -> bool:
        """Initialize the complete pipeline using registry pattern."""
        
        try:
            self.logger.info("🔧 Initializing component registry...")
            
            # Setup component system (registers all component types)
            registry_success = setup_enhanced_component_system()
            if not registry_success:
                self.logger.error("Failed to setup component system")
                return False
            
            self.logger.info("✅ Component registry initialized")
            self.activity_logger.log_activity('registry_setup', 'pipeline_runner', {
                'registered_types': list(component_registry.registered_types.keys()),
                'total_types': len(component_registry.registered_types)
            })
            
            # Create pipeline orchestrator configuration
            pipeline_config = self._create_pipeline_configuration()
            
            # Create orchestrator instance using registry  
            self.logger.info("🎭 Creating pipeline orchestrator...")
            
            # Create component configuration wrapper
            from musequill.v3.components.base.component_interface import ComponentConfiguration, ComponentType
            
            orchestrator_component_config = ComponentConfiguration(
                component_type=ComponentType.ORCHESTRATOR,
                component_name="Enhanced Pipeline Orchestrator",
                version="3.0.0",
                max_concurrent_executions=1,
                execution_timeout_seconds=pipeline_config.pipeline_timeout_minutes * 60,
                specific_config=pipeline_config
            )
            
            orchestrator_id = component_registry.create_component(
                'pipeline_orchestrator', 
                orchestrator_component_config
            )

            self.orchestrator = component_registry.get_component(orchestrator_id)
            if not self.orchestrator:
                raise ComponentError(f"Failed to create orchestrator with ID: {orchestrator_id}")
            
            # Initialize the orchestrator using BaseComponent interface
            self.logger.info("🏗️ Initializing pipeline orchestrator...")
            init_success = await self.orchestrator.initialize()
            
            if init_success:
                self.logger.info("✅ Pipeline initialization complete")
                self.activity_logger.log_activity('pipeline_init_success', 'pipeline_runner', {
                    'orchestrator_id': orchestrator_id,
                    'pipeline_strategy': pipeline_config.orchestration_strategy.value,
                    'orchestrator_status': self.orchestrator.state.status.value
                })
                return True
            else:
                self.logger.error("❌ Pipeline initialization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Pipeline initialization error: {e}")
            self.activity_logger.log_error('pipeline_runner', e, {'stage': 'initialization'})
            return False
    
    def _create_pipeline_configuration(self) -> EnhancedPipelineOrchestratorConfig:
        """Create enhanced pipeline orchestrator configuration from main config."""
        return create_enhanced_pipeline_configuration_from_dict(self.config)
    
    async def execute_pipeline(self, story_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        
        if not self.orchestrator:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
        
        self.logger.info("🚀 Starting pipeline execution...")
        self.activity_logger.log_activity('pipeline_execution_start', 'pipeline_runner', {
            'story_title': story_config.get('title', 'Unknown'),
            'estimated_chapters': story_config.get('structure', {}).get('estimated_chapters', 0)
        })
        
        try:
            # Use the enhanced orchestrator's story generation pipeline method
            if hasattr(self.orchestrator, 'execute_story_generation_pipeline'):
                # Use enhanced method with research integration
                results = await self.orchestrator.execute_story_generation_pipeline(
                    story_config=story_config,
                    manual_research_queries=story_config.get('manual_research_queries', [])
                )
            else:
                # Fall back to basic pipeline execution (convert story_config to proper input)
                from musequill.v3.components.orchestration.pipeline_orchestrator import PipelineOrchestratorInput
                orchestrator_input = PipelineOrchestratorInput(
                    chapter_objectives=[],  # Would need to convert story_config to chapter objectives
                    story_state=story_config
                )
                pipeline_result = await self.orchestrator.process(orchestrator_input)
                results = pipeline_result.model_dump() if hasattr(pipeline_result, 'model_dump') else pipeline_result
            
            self.logger.info("✅ Pipeline execution completed successfully")
            self.activity_logger.log_activity('pipeline_execution_success', 'pipeline_runner', {
                'execution_time': results.get('execution_time_seconds', 0),
                'chapters_generated': len(results.get('chapters', [])),
                'quality_score': results.get('final_quality_score', 0)
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.activity_logger.log_error('pipeline_runner', e, {'stage': 'execution'})
            raise
    
    async def cleanup_pipeline(self):
        """Clean up pipeline resources."""
        
        if self.orchestrator:
            try:
                # Use BaseComponent cleanup method
                cleanup_success = await self.orchestrator.cleanup()
                if cleanup_success:
                    self.logger.info("✅ Pipeline cleanup completed")
                    self.activity_logger.log_activity('pipeline_cleanup', 'pipeline_runner', {
                        'cleanup_successful': True
                    })
                else:
                    self.logger.warning("⚠️ Pipeline cleanup completed with warnings")
                    
            except Exception as e:
                self.logger.error(f"Pipeline cleanup error: {e}")
                self.activity_logger.log_error('pipeline_runner', e, {'stage': 'cleanup'})


def setup_enhanced_logging(log_level: str = "INFO", log_dir: Path = Path("logs")) -> PipelineActivityLogger:
    """Setup enhanced logging with registry awareness."""
    return setup_pipeline_logging(log_level, log_dir)


def load_enhanced_configuration(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load enhanced configuration with registry support."""
    
    logger = logging.getLogger('main.config')
    
    if config_path and config_path.exists():
        logger.info(f"Loading configuration from: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Enhance config with registry-specific settings
        config = _enhance_configuration_for_registry(config)
        logger.info("Enhanced configuration loaded successfully")
        return config
    
    else:
        logger.warning("No configuration file provided, using enhanced default configuration")
        return get_enhanced_default_configuration()


def _enhance_configuration_for_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance existing configuration with registry-specific settings."""
    
    # Add component registry settings if not present
    if 'component_registry' not in config:
        config['component_registry'] = {
            'auto_recycle_components': True,
            'max_component_instances': 10,
            'component_health_check_interval': 300,
            'enable_component_metrics': True
        }
    
    # Ensure orchestration settings are present
    if 'orchestration' not in config:
        config['orchestration'] = {
            'strategy': 'balanced',
            'enable_research_integration': True,
            'max_concurrent_operations': 3
        }
    
    # Add enhanced error handling settings
    if 'error_handling' not in config:
        config['error_handling'] = {
            'max_retry_attempts': 3,
            'enable_automatic_recovery': True,
            'fallback_strategies': True
        }
    
    return config


def get_enhanced_default_configuration() -> Dict[str, Any]:
    """Get enhanced default configuration optimized for registry pattern."""
    
    return create_default_pipeline_configuration().model_dump()


def setup_signal_handlers(pipeline_runner: EnhancedPipelineRunner, activity_logger: PipelineActivityLogger):
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger = logging.getLogger('main.signals')
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        activity_logger.log_activity('shutdown_signal_received', 'main', {
            'signal': signum,
            'timestamp': datetime.now().isoformat()
        })
        # Note: Async cleanup will be handled in the main execution loop
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def create_example_story_config() -> Dict[str, Any]:
    """Create an example story configuration for testing."""
    return {
        'title': 'The Quantum Enigma',
        'genre': 'science_fiction',
        'plot_type': 'mystery',
        'target_length': 'novel',
        'target_audience': 'adult',
        'structure': {
            'estimated_chapters': 12,
            'act_structure': '3-act'
        },
        'characters': {
            'protagonist': {
                'name': 'Dr. Sarah Chen',
                'role': 'quantum physicist',
                'motivation': 'uncover scientific conspiracy'
            }
        },
        'setting': {
            'time': '2045',
            'location': 'CERN facility, Geneva'
        },
        'themes': ['scientific ethics', 'reality vs simulation', 'trust'],
        'manual_research_queries': [
            'quantum physics research trends 2024',
            'CERN facility recent discoveries',
            'science fiction thriller market analysis'
        ]
    }


async def main():
    """Main entry point for the enhanced MuseQuill pipeline."""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='MuseQuill V3 Enhanced Pipeline Runner')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--story-config', type=str, help='Path to story configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for log files')
    parser.add_argument('--validate-config', action='store_true', help='Only validate configuration and exit')
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        log_dir = Path(args.log_dir)
        activity_logger = setup_enhanced_logging(args.log_level, log_dir)
        logger = logging.getLogger('main')
        
        logger.info("🚀 Starting MuseQuill V3 Enhanced Pipeline")
        activity_logger.log_activity('pipeline_start', 'main', {
            'version': '3.0.0',
            'config_path': args.config,
            'story_config_path': args.story_config,
            'log_level': args.log_level
        })
        
        # Load configuration
        config = load_enhanced_configuration(Path(args.config) if args.config else None)
        
        # Load story configuration
        if args.story_config and Path(args.story_config).exists():
            logger.info(f"Loading story configuration from: {args.story_config}")
            with open(args.story_config, 'r') as f:
                if args.story_config.endswith('.yaml') or args.story_config.endswith('.yml'):
                    story_config = yaml.safe_load(f)
                else:
                    story_config = json.load(f)
        else:
            logger.info("Using example story configuration")
            story_config = create_example_story_config()
        
        # Validate configuration if requested
        if args.validate_config:
            logger.info("✅ Configuration validation passed")
            return
        
        # Create enhanced pipeline runner
        pipeline_runner = EnhancedPipelineRunner(config, activity_logger)
        
        # Setup signal handlers
        setup_signal_handlers(pipeline_runner, activity_logger)
        
        # Initialize pipeline using registry pattern
        logger.info("🏗️ Initializing enhanced pipeline...")
        init_success = await pipeline_runner.initialize_pipeline()
        
        if not init_success:
            logger.error("❌ Pipeline initialization failed")
            return
        
        # Execute pipeline
        logger.info("🎬 Executing pipeline...")
        results = await pipeline_runner.execute_pipeline(story_config)
        
        # Save results
        logger.info("💾 Saving pipeline results...")
        output_dir = await save_pipeline_results(
            results, 
            story_config, 
            config, 
            activity_logger
        )
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Show summary
        summary = {
            'execution_time': results.get('execution_time_seconds', 0),
            'chapters_generated': len(results.get('chapters', [])),
            'final_quality_score': results.get('final_quality_score', 0),
            'output_directory': str(output_dir)
        }
        
        activity_logger.log_activity('pipeline_completion', 'main', summary)
        
        # Cleanup
        await pipeline_runner.cleanup_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        activity_logger.log_activity('pipeline_interrupted', 'main', {'reason': 'user_interrupt'})
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        activity_logger.log_error('main', e, {'stage': 'main_execution'})
        raise
    finally:
        activity_logger.save_final_report()


if __name__ == "__main__":
    import sys
    import warnings
    
    # Suppress threading cleanup warnings in Python 3.13
    warnings.filterwarnings("ignore", message=".*_DeleteDummyThreadOnDel.*")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        sys.exit(0)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)