"""
MuseQuill V3 Pipeline Launcher
Main entry point for the adversarial book generation system with comprehensive logging.

Usage:
    python main.py --config config.yaml --story story_config.json [options]
    python main.py --example  # Run with example configuration
    python main.py --validate-config  # Validate configuration files
"""

import asyncio
import logging
import sys
import argparse
import json
import yaml
import signal
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import colorama
from colorama import Fore, Back, Style
import os

# Configure colorama for cross-platform colored output
colorama.init(autoreset=True)

# Import your pipeline components
from musequill.v3.components.orchestration.pipeline_orchestrator import (
    PipelineOrchestrator, PipelineOrchestratorConfig, PipelineOrchestratorInput,
    PipelineState, OrchestrationStrategy
)
from musequill.v3.components.base.component_interface import (
    ComponentConfiguration, ComponentType, component_registry
)
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.chapter_objective import ChapterObjective

# Import enhanced researcher integration
from musequill.v3.components.orchestration.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator, PipelineResearcherConfig


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE,
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


class PipelineActivityLogger:
    """Specialized logger for tracking pipeline activities and metrics."""
    
    def __init__(self, log_dir: Path = Path("logs")):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Activity log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.activity_log_file = self.log_dir / f"pipeline_activity_{timestamp}.json"
        self.performance_log_file = self.log_dir / f"performance_metrics_{timestamp}.json"
        
        # In-memory activity tracking
        self.activities: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'pipeline_stages': [],
            'component_metrics': {},
            'research_activities': [],
            'error_events': [],
            'quality_assessments': []
        }
        
        # Setup activity logger
        self.logger = logging.getLogger('pipeline.activity')
        self.logger.setLevel(logging.INFO)
        
        # File handler for activity logs
        activity_handler = logging.FileHandler(self.activity_log_file)
        activity_handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "%(name)s", "message": %(message)s}'
        ))
        self.logger.addHandler(activity_handler)
    
    def log_activity(self, activity_type: str, component: str, details: Dict[str, Any]):
        """Log a pipeline activity with structured data."""
        activity = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'activity_type': activity_type,
            'component': component,
            'details': details
        }
        
        self.activities.append(activity)
        self.logger.info(json.dumps(activity, default=str))
    
    def log_stage_start(self, stage_name: str, input_data: Dict[str, Any]):
        """Log the start of a pipeline stage."""
        stage_info = {
            'stage': stage_name,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'input_summary': self._summarize_input(input_data)
        }
        
        self.performance_metrics['pipeline_stages'].append(stage_info)
        self.log_activity('stage_start', 'pipeline', stage_info)
    
    def log_stage_complete(self, stage_name: str, output_data: Dict[str, Any], execution_time: float):
        """Log the completion of a pipeline stage."""
        # Find the matching stage entry
        for stage in reversed(self.performance_metrics['pipeline_stages']):
            if stage['stage'] == stage_name and 'end_time' not in stage:
                stage['end_time'] = datetime.now(timezone.utc).isoformat()
                stage['execution_time_seconds'] = execution_time
                stage['output_summary'] = self._summarize_output(output_data)
                break
        
        self.log_activity('stage_complete', 'pipeline', {
            'stage': stage_name,
            'execution_time': execution_time,
            'success': True
        })
    
    def log_component_performance(self, component_name: str, metrics: Dict[str, Any]):
        """Log component performance metrics."""
        self.performance_metrics['component_metrics'][component_name] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **metrics
        }
        
        self.log_activity('component_performance', component_name, metrics)
    
    def log_research_activity(self, research_type: str, query: str, results: Dict[str, Any]):
        """Log research activities."""
        research_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'research_type': research_type,
            'query': query,
            'results_summary': self._summarize_research_results(results)
        }
        
        self.performance_metrics['research_activities'].append(research_entry)
        self.log_activity('research', 'researcher', research_entry)
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any]):
        """Log error events with full context."""
        error_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.performance_metrics['error_events'].append(error_entry)
        self.log_activity('error', component, error_entry)
    
    def log_quality_assessment(self, chapter_id: str, assessment: Dict[str, Any]):
        """Log quality assessment results."""
        quality_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'chapter_id': chapter_id,
            'assessment': assessment
        }
        
        self.performance_metrics['quality_assessments'].append(quality_entry)
        self.log_activity('quality_assessment', 'quality_controller', quality_entry)
    
    def save_final_report(self):
        """Save final performance report to file."""
        self.performance_metrics['end_time'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate summary statistics
        self.performance_metrics['summary'] = self._calculate_summary_stats()
        
        # Save to file
        with open(self.performance_log_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        logging.getLogger('main').info(f"Final performance report saved to: {self.performance_log_file}")
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of input data for logging."""
        return {
            'keys': list(input_data.keys()),
            'data_types': {k: type(v).__name__ for k, v in input_data.items()},
            'size_estimate': len(str(input_data))
        }
    
    def _summarize_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of output data for logging."""
        return {
            'keys': list(output_data.keys()),
            'data_types': {k: type(v).__name__ for k, v in output_data.items()},
            'size_estimate': len(str(output_data))
        }
    
    def _summarize_research_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of research results."""
        return {
            'status': results.get('status', 'unknown'),
            'sources_found': results.get('sources_found', 0),
            'execution_time': results.get('execution_time', 0),
            'has_summary': bool(results.get('summary'))
        }
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics for the pipeline run."""
        total_stages = len(self.performance_metrics['pipeline_stages'])
        completed_stages = len([s for s in self.performance_metrics['pipeline_stages'] if 'end_time' in s])
        total_errors = len(self.performance_metrics['error_events'])
        total_research = len(self.performance_metrics['research_activities'])
        
        # Calculate total execution time
        start_time = datetime.fromisoformat(self.performance_metrics['start_time'].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(self.performance_metrics['end_time'].replace('Z', '+00:00'))
        total_time = (end_time - start_time).total_seconds()
        
        return {
            'total_execution_time_seconds': total_time,
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'stage_completion_rate': completed_stages / max(total_stages, 1),
            'total_errors': total_errors,
            'total_research_activities': total_research,
            'components_used': list(self.performance_metrics['component_metrics'].keys()),
            'quality_assessments_performed': len(self.performance_metrics['quality_assessments'])
        }


def setup_logging(log_level: str = "INFO", log_dir: Path = Path("logs")) -> PipelineActivityLogger:
    """Setup comprehensive logging system."""
    
    # Create logs directory
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} | '
        f'%(levelname)s | '
        f'{Fore.MAGENTA}%(name)s{Style.RESET_ALL} | '
        f'%(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Create activity logger
    activity_logger = PipelineActivityLogger(log_dir)
    
    logging.getLogger('main').info(f"Logging initialized - Console: {log_level}, File: {log_file}")
    
    return activity_logger


def load_configuration(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load pipeline configuration from file or create default."""
    
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
        
        logger.info("Configuration loaded successfully")
        return config
    
    else:
        logger.warning("No configuration file provided, using default configuration")
        return get_default_configuration()


def get_default_configuration() -> Dict[str, Any]:
    """Get default pipeline configuration."""
    return {
        'pipeline': {
            'max_generation_attempts': 5,
            'max_revision_cycles': 3,
            'orchestration_strategy': 'balanced',
            'parallel_variant_evaluation': True,
            'enable_market_intelligence_refresh': True,
            'market_refresh_interval_hours': 24,
            'component_health_check_interval': 300,
            'enable_adaptive_orchestration': True,
            'pipeline_timeout_minutes': 60,
            'enable_comprehensive_logging': True,
            'fallback_on_component_failure': True
        },
        'researcher': {
            'tavily_api_key': os.getenv('TAVILY_API_KEY', ''),
            'max_concurrent_requests': 3,
            'enable_caching': True,
            'cache_ttl_hours': 12,
            'quality_threshold': 0.7,
            'auto_trigger_conditions': {
                'market_data_age_hours': 6,
                'plot_inconsistency_threshold': 0.8,
                'character_development_gaps': True,
                'trend_analysis_frequency': 'daily'
            }
        },
        'components': {
            'chapter_generator': {
                'model_name': 'claude-3-sonnet',
                'temperature': 0.8,
                'max_tokens': 4000
            },
            'plot_coherence_critic': {
                'strictness_level': 'moderate'
            },
            'literary_quality_critic': {
                'style_analysis_depth': 'comprehensive'
            },
            'reader_engagement_critic': {
                'target_audience': 'general_adult'
            },
            'quality_controller': {
                'quality_threshold': 0.75,
                'enable_detailed_feedback': True
            },
            'market_intelligence_engine': {
                'update_frequency_hours': 24,
                'search_depth': 'advanced',
                'enable_competitive_analysis': True
            }
        },
        'output': {
            'base_directory': './output',
            'save_intermediate_results': True,
            'export_formats': ['json', 'text', 'epub']
        },
        'logging': {
            'level': 'INFO',
            'log_directory': './logs',
            'enable_activity_tracking': True,
            'enable_performance_metrics': True
        }
    }


def create_example_story_config() -> Dict[str, Any]:
    """Create example story configuration for testing."""
    return {
        'title': 'The Digital Shadows',
        'genre': 'thriller',
        'subgenre': 'techno-thriller',
        'target_audience': 'adult',
        'target_length': 'novel',
        'estimated_chapters': 20,
        'premise': 'A cybersecurity expert discovers a conspiracy that threatens global digital infrastructure.',
        'main_character': {
            'name': 'Dr. Sarah Chen',
            'role': 'protagonist',
            'background': 'Former NSA analyst turned private cybersecurity consultant',
            'motivation': 'Protect digital privacy rights and expose corporate surveillance'
        },
        'themes': ['technology vs privacy', 'corporate power', 'individual agency'],
        'tone': 'suspenseful',
        'pacing': 'fast',
        'setting': {
            'primary_location': 'San Francisco',
            'time_period': 'contemporary',
            'tech_level': 'near-future'
        },
        'market_requirements': {
            'trend_awareness': True,
            'competitive_positioning': 'differentiated',
            'reader_engagement_focus': 'high'
        }
    }


async def validate_configuration(config: Dict[str, Any]) -> bool:
    """Validate pipeline configuration."""
    
    logger = logging.getLogger('main.validation')
    logger.info("Validating configuration...")
    
    required_sections = ['pipeline', 'components', 'output', 'logging']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        logger.error(f"Missing required configuration sections: {missing_sections}")
        return False
    
    # Validate researcher configuration if present
    if 'researcher' in config:
        researcher_config = config['researcher']
        if not researcher_config.get('tavily_api_key'):
            logger.warning("Tavily API key not configured - research functionality will be limited")
    
    # Validate component configurations
    required_components = [
        'chapter_generator', 'plot_coherence_critic', 'literary_quality_critic',
        'reader_engagement_critic', 'quality_controller', 'market_intelligence_engine'
    ]
    
    components_config = config.get('components', {})
    missing_components = [comp for comp in required_components if comp not in components_config]
    
    if missing_components:
        logger.warning(f"Missing component configurations (will use defaults): {missing_components}")
    
    logger.info("Configuration validation completed")
    return True


async def initialize_pipeline(config: Dict[str, Any], activity_logger: PipelineActivityLogger) -> EnhancedPipelineOrchestrator:
    """Initialize the enhanced pipeline orchestrator with all components."""
    
    logger = logging.getLogger('main.init')
    logger.info("Initializing enhanced pipeline orchestrator...")
    
    activity_logger.log_activity('pipeline_init_start', 'main', {'config_keys': list(config.keys())})
    
    try:
        # Create enhanced orchestrator with research capabilities
        orchestrator = EnhancedPipelineOrchestrator(config)
        
        # Initialize the orchestrator
        await orchestrator.initialize()
        
        activity_logger.log_activity('pipeline_init_complete', 'main', {
            'orchestrator_type': 'enhanced',
            'research_enabled': config.get('researcher', {}).get('tavily_api_key', '') != ''
        })
        
        logger.info("Pipeline orchestrator initialized successfully")
        return orchestrator
        
    except Exception as e:
        activity_logger.log_error('main', e, {'stage': 'pipeline_initialization'})
        raise


async def execute_story_generation(
    orchestrator: EnhancedPipelineOrchestrator,
    story_config: Dict[str, Any],
    config: Dict[str, Any],
    activity_logger: PipelineActivityLogger
) -> Dict[str, Any]:
    """Execute the complete story generation pipeline."""
    
    logger = logging.getLogger('main.execution')
    logger.info(f"Starting story generation for: {story_config.get('title', 'Untitled')}")
    
    activity_logger.log_stage_start('story_generation', story_config)
    
    start_time = datetime.now()
    
    try:
        # Prepare upfront research queries if research is enabled
        upfront_research = []
        if config.get('researcher', {}).get('tavily_api_key'):
            genre = story_config.get('genre', 'general')
            upfront_research = [
                f"{genre} fiction market trends 2025 reader preferences",
                f"{genre} successful plot structures techniques",
                f"{genre} character archetypes reader engagement",
                f"current {genre} publishing industry analysis"
            ]
            logger.info(f"Prepared {len(upfront_research)} upfront research queries")
        
        # Execute the pipeline
        logger.info("Executing story generation pipeline...")
        final_story_state = await orchestrator.execute_story_generation_pipeline(
            story_config=story_config,
            manual_research_queries=upfront_research
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        activity_logger.log_stage_complete('story_generation', final_story_state, execution_time)
        
        logger.info(f"Story generation completed successfully in {execution_time:.2f} seconds")
        
        # Log summary statistics
        research_count = len(final_story_state.get('research_history', {}))
        chapters_generated = len(final_story_state.get('chapters', {}))
        
        activity_logger.log_activity('generation_summary', 'main', {
            'execution_time_seconds': execution_time,
            'research_activities': research_count,
            'chapters_generated': chapters_generated,
            'final_state_keys': list(final_story_state.keys())
        })
        
        return final_story_state
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        activity_logger.log_error('main', e, {
            'stage': 'story_generation',
            'execution_time': execution_time
        })
        raise


async def save_results(
    story_state: Dict[str, Any],
    story_config: Dict[str, Any],
    config: Dict[str, Any],
    activity_logger: PipelineActivityLogger
):
    """Save generation results in multiple formats."""
    
    logger = logging.getLogger('main.output')
    output_config = config.get('output', {})
    base_dir = Path(output_config.get('base_directory', './output'))
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    story_title = story_config.get('title', 'untitled').replace(' ', '_').lower()
    output_dir = base_dir / f"{story_title}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to: {output_dir}")
    
    activity_logger.log_activity('save_start', 'output', {'output_directory': str(output_dir)})
    
    saved_files = []
    
    try:
        # Save complete story state as JSON
        json_file = output_dir / "complete_story_state.json"
        with open(json_file, 'w') as f:
            json.dump(story_state, f, indent=2, default=str)
        saved_files.append(str(json_file))
        
        # Save story configuration
        config_file = output_dir / "story_configuration.json"
        with open(config_file, 'w') as f:
            json.dump(story_config, f, indent=2, default=str)
        saved_files.append(str(config_file))
        
        # Save individual chapters as text files
        chapters = story_state.get('chapters', {})
        if chapters:
            chapters_dir = output_dir / "chapters"
            chapters_dir.mkdir(exist_ok=True)
            
            for chapter_id, chapter_data in chapters.items():
                chapter_file = chapters_dir / f"chapter_{chapter_id}.txt"
                with open(chapter_file, 'w') as f:
                    f.write(f"Chapter {chapter_id}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(chapter_data.get('content', 'No content available'))
                saved_files.append(str(chapter_file))
        
        # Save research summary if available
        research_history = story_state.get('research_history', {})
        if research_history:
            research_file = output_dir / "research_summary.json"
            with open(research_file, 'w') as f:
                json.dump(research_history, f, indent=2, default=str)
            saved_files.append(str(research_file))
        
        # Save market intelligence if available
        market_intel = story_state.get('market_intelligence', {})
        if market_intel:
            market_file = output_dir / "market_intelligence.json"
            with open(market_file, 'w') as f:
                json.dump(market_intel, f, indent=2, default=str)
            saved_files.append(str(market_file))
        
        # Save quality assessments if available
        quality_assessments = story_state.get('quality_assessments', {})
        if quality_assessments:
            quality_file = output_dir / "quality_assessments.json"
            with open(quality_file, 'w') as f:
                json.dump(quality_assessments, f, indent=2, default=str)
            saved_files.append(str(quality_file))
        
        activity_logger.log_activity('save_complete', 'output', {
            'files_saved': len(saved_files),
            'output_directory': str(output_dir),
            'saved_files': saved_files
        })
        
        logger.info(f"Successfully saved {len(saved_files)} files to {output_dir}")
        return output_dir
        
    except Exception as e:
        activity_logger.log_error('output', e, {'output_directory': str(output_dir)})
        raise


def setup_signal_handlers(orchestrator: EnhancedPipelineOrchestrator, activity_logger: PipelineActivityLogger):
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(sig, frame):
        logger = logging.getLogger('main.shutdown')
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        
        activity_logger.log_activity('shutdown_signal', 'main', {'signal': sig})
        
        # Create shutdown task
        async def shutdown():
            try:
                await orchestrator.shutdown()
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


async def main():
    """Main pipeline execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MuseQuill V3 Adversarial Book Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --example
    python main.py --config config.yaml --story story.json
    python main.py --config config.yaml --story story.json --log-level DEBUG
    python main.py --validate-config --config config.yaml
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
    
    # Setup logging
    activity_logger = setup_logging(args.log_level, args.log_dir)
    logger = logging.getLogger('main')
    
    logger.info("=" * 80)
    logger.info(f"{Fore.CYAN}{Style.BRIGHT}MuseQuill V3 Pipeline Starting{Style.RESET_ALL}")
    logger.info("=" * 80)
    
    activity_logger.log_activity('pipeline_start', 'main', {
        'args': vars(args),
        'python_version': sys.version,
        'working_directory': str(Path.cwd())
    })
    
    try:
        # Load configuration
        if args.example:
            logger.info("Running with example configuration")
            config = get_default_configuration()
            story_config = create_example_story_config()
        else:
            config = load_configuration(args.config)
            
            # Override output directory if specified
            if args.output_dir:
                config['output']['base_directory'] = str(args.output_dir)
            
            # Load story configuration
            if args.story and args.story.exists():
                logger.info(f"Loading story configuration from: {args.story}")
                with open(args.story, 'r') as f:
                    story_config = json.load(f)
            else:
                logger.info("No story configuration provided, using example")
                story_config = create_example_story_config()
        
        # Validate configuration
        if not await validate_configuration(config):
            logger.error("Configuration validation failed")
            return 1
        
        if args.validate_config:
            logger.info("Configuration validation successful")
            return 0
        
        # Initialize pipeline
        orchestrator = await initialize_pipeline(config, activity_logger)
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(orchestrator, activity_logger)
        
        # Execute story generation
        logger.info(f"Generating story: '{story_config.get('title', 'Untitled')}'")
        story_state = await execute_story_generation(orchestrator, story_config, config, activity_logger)
        
        # Save results
        output_dir = await save_results(story_state, story_config, config, activity_logger)
        
        # Final success message
        logger.info("=" * 80)
        logger.info(f"{Fore.GREEN}{Style.BRIGHT}Pipeline Execution Completed Successfully!{Style.RESET_ALL}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
        # Save final activity report
        activity_logger.save_final_report()
        
        # Clean shutdown
        await orchestrator.shutdown()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        activity_logger.log_activity('user_interrupt', 'main', {})
        return 130
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        activity_logger.log_error('main', e, {'stage': 'main_execution'})
        return 1
        
    finally:
        # Ensure activity logger saves final report
        try:
            activity_logger.save_final_report()
        except Exception as e:
            logger.error(f"Failed to save final activity report: {e}")


if __name__ == "__main__":
    # Ensure proper async execution
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Pipeline interrupted by user{Style.RESET_ALL}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        sys.exit(1)