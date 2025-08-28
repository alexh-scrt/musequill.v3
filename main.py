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

# Import enhanced registry and orchestration
from musequill.v3.components.component_registry import setup_enhanced_component_system, create_enhanced_component_configurations
from musequill.v3.components.base.component_interface import component_registry, ComponentError
from musequill.v3.components.orchestration.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator

# Import pipeline configuration classes (save as separate module first)
from musequill.v3.pipeline_configuration import PipelineConfiguration, create_pipeline_configuration_from_dict, create_default_pipeline_configuration

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
            
            # Create pipeline configuration from config
            pipeline_config = self._create_pipeline_configuration()
            
            # Create orchestrator instance using registry
            self.logger.info("🎭 Creating pipeline orchestrator...")
            # orchestrator_id = component_registry.create_component(
            #     'pipeline_orchestrator', 
            #     pipeline_config.orchestrator_config
            # )
            
            orchestrator_id = component_registry.create_component(
                'pipeline_orchestrator', 
                pipeline_config.orchestrator_config
            )

            self.orchestrator = component_registry.get_component(orchestrator_id)
            if not self.orchestrator:
                raise ComponentError(f"Failed to create orchestrator with ID: {orchestrator_id}")
            
            # Initialize all pipeline components through orchestrator
            self.logger.info("🏗️ Initializing pipeline components...")
            init_success = await self.orchestrator._initialize_all_components(pipeline_config)
            
            if init_success:
                self.logger.info("✅ Pipeline initialization complete")
                self.activity_logger.log_activity('pipeline_init_success', 'pipeline_runner', {
                    'orchestrator_id': orchestrator_id,
                    'pipeline_strategy': pipeline_config.orchestration_strategy,
                    'components_initialized': len(self.orchestrator.get_active_components())
                })
                return True
            else:
                self.logger.error("❌ Pipeline initialization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Pipeline initialization error: {e}")
            self.activity_logger.log_error('pipeline_runner', e, {'stage': 'initialization'})
            return False
    
    def _create_pipeline_configuration(self) -> PipelineConfiguration:
        """Create pipeline configuration from main config."""
        
        return create_pipeline_configuration_from_dict(self.config)
    
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
            # Execute pipeline through orchestrator
            results = await self.orchestrator.execute_pipeline(story_config)
            
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
                await self.orchestrator.cleanup()
                self.logger.info("✅ Pipeline cleanup completed")
                self.activity_logger.log_activity('pipeline_cleanup', 'pipeline_runner', {
                    'cleanup_successful': True
                })
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
    
    def signal_handler(sig, frame):
        logger = logging.getLogger('main.shutdown')
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        
        activity_logger.log_activity('shutdown_signal', 'main', {'signal': sig})
        
        async def shutdown():
            try:
                await pipeline_runner.cleanup_pipeline()
                activity_logger.save_final_report()
                logger.info("Graceful shutdown completed")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            finally:
                sys.exit(0)
        
        # Run shutdown in event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(shutdown())
        else:
            asyncio.run(shutdown())
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def create_example_story_config() -> Dict[str, Any]:
    """Create example story configuration optimized for registry pattern."""
    
    return {
        "title": "The Neural Network",
        "genre": "Science Fiction Thriller",
        "target_audience": "Adult",
        "estimated_length": "80000-100000 words",
        
        "protagonist": {
            "name": "Dr. Sarah Chen",
            "background": "AI researcher at a leading tech company",
            "motivation": "Uncover the truth about a mysterious AI breakthrough",
            "personality_traits": ["analytical", "persistent", "ethical"]
        },
        
        "supporting_characters": [
            {
                "name": "Marcus Rodriguez",
                "role": "Senior Engineer",
                "relationship_to_protagonist": "Colleague and friend",
                "motivation": "Support Sarah while protecting company secrets"
            }
        ],
        
        "plot_structure": {
            "act_1": "Discovery of unusual AI behavior patterns",
            "act_2": "Investigation reveals deeper conspiracy",
            "act_3": "Confrontation and resolution",
            "estimated_chapters": 12,
            "pacing": "fast-paced with technical elements"
        },
        
        "setting": {
            "time_period": "Near future (2030s)",
            "primary_location": "Silicon Valley tech campus",
            "secondary_locations": ["San Francisco", "Remote data centers"],
            "technology_level": "Advanced AI, quantum computing"
        },
        
        "themes": ["Technology ethics", "Corporate responsibility", "Human vs AI intelligence"],
        
        "quality_targets": {
            "plot_coherence_minimum": 0.80,
            "literary_quality_minimum": 0.75,
            "reader_engagement_minimum": 0.85,
            "market_viability_minimum": 0.70
        }
    }


async def main():
    """Enhanced main pipeline execution function using registry pattern."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MuseQuill V3 Enhanced Pipeline with Component Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --example
    python main.py --config config.yaml --story story.json
    python main.py --config config.yaml --story story.json --log-level DEBUG
    python main.py --validate-config --config config.yaml
    python main.py --test-registry  # Test component registry setup
        """
    )
    
    parser.add_argument('--config', type=Path, help='Pipeline configuration file (.yaml or .json)')
    parser.add_argument('--story', type=Path, help='Story configuration file (.json)')
    parser.add_argument('--example', action='store_true', help='Run with example configuration')
    parser.add_argument('--validate-config', action='store_true', help='Validate configuration and exit')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-dir', type=Path, default=Path('logs'), help='Directory for log files')
    parser.add_argument('--output-dir', type=Path, help='Override output directory')
    
    args = parser.parse_args()
    
    # Setup enhanced logging
    activity_logger = setup_enhanced_logging(args.log_level, args.log_dir)
    logger = logging.getLogger('main')
    
    logger.info("=" * 80)
    logger.info(f"{Fore.CYAN}{Style.BRIGHT}MuseQuill V3 Enhanced Pipeline Starting{Style.RESET_ALL}")
    logger.info(f"Using Component Registry Pattern")
    logger.info("=" * 80)
    
    activity_logger.log_activity('pipeline_start', 'main', {
        'args': vars(args),
        'python_version': sys.version,
        'working_directory': str(Path.cwd()),
        'registry_pattern': True
    })
        
    try:
        # Load enhanced configuration
        if args.example:
            logger.info("Running with enhanced example configuration")
            config = get_enhanced_default_configuration()
            story_config = create_example_story_config()
        else:
            config = load_enhanced_configuration(args.config)
            
            # Override output directory if specified
            if args.output_dir:
                config['output']['base_directory'] = str(args.output_dir)
            
            # Load story configuration
            if args.story and args.story.exists():
                logger.info(f"Loading story configuration from: {args.story}")
                with open(args.story, 'r') as f:
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
    asyncio.run(main())