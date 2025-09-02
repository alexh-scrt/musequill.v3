"""
Pipeline Integration Module

Provides integration utilities for the MuseQuill V3 pipeline, including
result saving, configuration validation, and pipeline orchestration helpers.
"""

import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import yaml

# Import activity logger
from musequill.v3.pipeline_activity_logger import PipelineActivityLogger

from musequill.v3.utils.envs import substitute_env_vars

async def save_pipeline_results(
    pipeline_results: Dict[str, Any],
    story_config: Dict[str, Any],
    config: Dict[str, Any],
    activity_logger: PipelineActivityLogger
) -> Path:
    """
    Save complete pipeline results to organized output directory.
    
    Args:
        pipeline_results: Results from pipeline execution
        story_config: Original story configuration
        config: Pipeline configuration
        activity_logger: Activity logger for tracking
        
    Returns:
        Path to the output directory
    """
    logger = logging.getLogger('pipeline_integration.save')
    output_config = config.get('output', {})
    od = output_config.get('base_directory')
    if od is None:
        od = './output'
    else:
        od = substitute_env_vars(od)
    base_dir = Path(od)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    story_title = story_config.get('title', 'untitled').replace(' ', '_').lower()
    output_dir = base_dir / f"{story_title}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving pipeline results to: {output_dir}")
    activity_logger.log_activity('save_start', 'output', {'output_directory': str(output_dir)})
    
    saved_files = []
    
    try:
        # Save complete pipeline results as JSON
        results_file = output_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        saved_files.append(str(results_file))
        
        # Save story configuration
        config_file = output_dir / "story_configuration.json"
        with open(config_file, 'w') as f:
            json.dump(story_config, f, indent=2, default=str)
        saved_files.append(str(config_file))
        
        # Save pipeline configuration
        pipeline_config_file = output_dir / "pipeline_configuration.json"
        with open(pipeline_config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        saved_files.append(str(pipeline_config_file))
        
        # Save individual chapters as text files
        chapters = pipeline_results.get('chapters', {})
        if chapters:
            chapters_dir = output_dir / "chapters"
            chapters_dir.mkdir(exist_ok=True)
            
            for chapter_id, chapter_data in chapters.items():
                # Handle different chapter data formats
                if isinstance(chapter_data, dict):
                    content = chapter_data.get('content', str(chapter_data))
                    title = chapter_data.get('title', f'Chapter {chapter_id}')
                else:
                    content = str(chapter_data)
                    title = f'Chapter {chapter_id}'
                
                chapter_file = chapters_dir / f"chapter_{chapter_id:02d}.txt"
                with open(chapter_file, 'w', encoding='utf-8') as f:
                    f.write(f"{title}\n")
                    f.write("=" * len(title) + "\n\n")
                    f.write(content)
                    f.write("\n")
                saved_files.append(str(chapter_file))
        
        # Save research summary if available
        research_history = pipeline_results.get('research_history', {})
        if research_history:
            research_file = output_dir / "research_summary.json"
            with open(research_file, 'w') as f:
                json.dump(research_history, f, indent=2, default=str)
            saved_files.append(str(research_file))
        
        # Save market intelligence if available
        market_intel = pipeline_results.get('market_intelligence', {})
        if market_intel:
            market_file = output_dir / "market_intelligence.json"
            with open(market_file, 'w') as f:
                json.dump(market_intel, f, indent=2, default=str)
            saved_files.append(str(market_file))
        
        # Save quality assessments if available
        quality_assessments = pipeline_results.get('quality_assessments', {})
        if quality_assessments:
            quality_file = output_dir / "quality_assessments.json"
            with open(quality_file, 'w') as f:
                json.dump(quality_assessments, f, indent=2, default=str)
            saved_files.append(str(quality_file))
        
        # Save component performance metrics if available
        component_metrics = pipeline_results.get('component_metrics', {})
        if component_metrics:
            metrics_file = output_dir / "component_performance.json"
            with open(metrics_file, 'w') as f:
                json.dump(component_metrics, f, indent=2, default=str)
            saved_files.append(str(metrics_file))
        
        # Save story state progression if available
        story_states = pipeline_results.get('story_state_history', [])
        if story_states:
            states_file = output_dir / "story_state_progression.json"
            with open(states_file, 'w') as f:
                json.dump(story_states, f, indent=2, default=str)
            saved_files.append(str(states_file))
        
        # Create a summary file
        summary = create_output_summary(pipeline_results, story_config, saved_files)
        summary_file = output_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files.append(str(summary_file))
        
        # Create human-readable summary
        readme_content = create_readme_content(pipeline_results, story_config, summary)
        readme_file = output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        saved_files.append(str(readme_file))
        
        activity_logger.log_activity('save_complete', 'output', {
            'files_saved': len(saved_files),
            'output_directory': str(output_dir),
            'saved_files': saved_files
        })
        
        logger.info(f"Successfully saved {len(saved_files)} files to {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error saving pipeline results: {e}")
        activity_logger.log_error('output', e, {'output_directory': str(output_dir)})
        raise


def create_output_summary(
    pipeline_results: Dict[str, Any], 
    story_config: Dict[str, Any], 
    saved_files: List[str]
) -> Dict[str, Any]:
    """Create a summary of the pipeline execution and outputs."""
    
    chapters = pipeline_results.get('chapters', {})
    
    summary = {
        'execution_timestamp': datetime.now().isoformat(),
        'story_info': {
            'title': story_config.get('title', 'Unknown'),
            'genre': story_config.get('genre', 'Unknown'),
            'target_audience': story_config.get('target_audience', 'Unknown'),
            'estimated_length': story_config.get('estimated_length', 'Unknown')
        },
        'pipeline_results': {
            'chapters_generated': len(chapters),
            'execution_time_seconds': pipeline_results.get('execution_time_seconds', 0),
            'final_quality_score': pipeline_results.get('final_quality_score', 0),
            'research_activities': len(pipeline_results.get('research_history', {})),
            'quality_assessments': len(pipeline_results.get('quality_assessments', {})),
            'pipeline_status': pipeline_results.get('status', 'unknown')
        },
        'output_files': {
            'total_files_saved': len(saved_files),
            'files_by_type': {
                'chapters': len([f for f in saved_files if 'chapter_' in f]),
                'metadata': len([f for f in saved_files if f.endswith('.json')]),
                'documentation': len([f for f in saved_files if f.endswith('.md')])
            }
        },
        'component_summary': extract_component_summary(pipeline_results)
    }
    
    return summary


def extract_component_summary(pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary information about component performance."""
    
    component_metrics = pipeline_results.get('component_metrics', {})
    
    summary = {
        'components_used': list(component_metrics.keys()),
        'total_components': len(component_metrics),
        'performance_overview': {}
    }
    
    for component_name, metrics in component_metrics.items():
        summary['performance_overview'][component_name] = {
            'execution_time': metrics.get('execution_time', 0),
            'success_rate': metrics.get('success_rate', 0),
            'error_count': metrics.get('error_count', 0)
        }
    
    return summary


def create_readme_content(
    pipeline_results: Dict[str, Any], 
    story_config: Dict[str, Any], 
    summary: Dict[str, Any]
) -> str:
    """Create human-readable README content for the output directory."""
    
    story_info = summary['story_info']
    pipeline_info = summary['pipeline_results']
    
    content = f"""# {story_info['title']}

**Genre**: {story_info['genre']}  
**Target Audience**: {story_info['target_audience']}  
**Estimated Length**: {story_info['estimated_length']}

## Pipeline Execution Summary

- **Execution Date**: {summary['execution_timestamp'][:10]}
- **Chapters Generated**: {pipeline_info['chapters_generated']}
- **Execution Time**: {pipeline_info['execution_time_seconds']:.1f} seconds
- **Final Quality Score**: {pipeline_info['final_quality_score']:.2f}
- **Pipeline Status**: {pipeline_info['pipeline_status']}

## Generated Content

### Chapters
Generated {pipeline_info['chapters_generated']} chapters. Individual chapter files are available in the `chapters/` directory.

### Research Activities
Conducted {pipeline_info['research_activities']} research activities to inform content generation.

### Quality Assessments
Performed {pipeline_info['quality_assessments']} quality assessments throughout the generation process.

## Files in This Directory

- `pipeline_results.json` - Complete pipeline execution results
- `story_configuration.json` - Original story configuration
- `pipeline_configuration.json` - Pipeline configuration used
- `execution_summary.json` - Summary of execution metrics
- `chapters/` - Individual chapter text files
"""

    # Add optional sections if data is available
    if pipeline_results.get('research_history'):
        content += "- `research_summary.json` - Summary of research activities\n"
    
    if pipeline_results.get('market_intelligence'):
        content += "- `market_intelligence.json` - Market research and trends\n"
    
    if pipeline_results.get('quality_assessments'):
        content += "- `quality_assessments.json` - Detailed quality assessment results\n"
    
    if pipeline_results.get('component_metrics'):
        content += "- `component_performance.json` - Component performance metrics\n"
    
    content += f"""
## Component Performance

"""
    
    # Add component performance summary
    component_summary = summary.get('component_summary', {})
    if component_summary.get('performance_overview'):
        for component_name, metrics in component_summary['performance_overview'].items():
            content += f"- **{component_name}**: {metrics['execution_time']:.1f}s execution, {metrics['success_rate']:.1%} success rate\n"
    
    content += f"""
---

*Generated by MuseQuill V3 Adversarial Book Generation Pipeline*
"""
    
    return content


async def validate_pipeline_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate pipeline configuration for completeness and correctness.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    logger = logging.getLogger('pipeline_integration.validation')
    
    required_sections = ['pipeline', 'components', 'output']
    missing_sections = []
    
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        logger.error(f"Missing required configuration sections: {missing_sections}")
        return False
    
    # Validate pipeline section
    pipeline_config = config['pipeline']
    required_pipeline_keys = ['orchestration_strategy', 'max_generation_attempts']
    for key in required_pipeline_keys:
        if key not in pipeline_config:
            logger.error(f"Missing required pipeline configuration key: {key}")
            return False
    
    # Validate components section
    components_config = config['components']
    required_components = ['chapter_generator', 'quality_controller']
    for component in required_components:
        if component not in components_config:
            logger.warning(f"Missing component configuration: {component}")
    
    # Validate output section
    output_config = config['output']
    if 'base_directory' not in output_config:
        logger.warning("No output base_directory specified, using default")
        config['output']['base_directory'] = './output'
    
    logger.info("Pipeline configuration validation passed")
    return True


async def validate_story_configuration(story_config: Dict[str, Any]) -> bool:
    """
    Validate story configuration for completeness.
    
    Args:
        story_config: Story configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    logger = logging.getLogger('pipeline_integration.validation')
    
    required_keys = ['title', 'genre', 'protagonist']
    missing_keys = []
    
    for key in required_keys:
        if key not in story_config:
            missing_keys.append(key)
    
    if missing_keys:
        logger.error(f"Missing required story configuration keys: {missing_keys}")
        return False
    
    # Validate protagonist structure
    protagonist = story_config['protagonist']
    if not isinstance(protagonist, dict):
        logger.error("Protagonist configuration must be a dictionary")
        return False
    
    required_protagonist_keys = ['name', 'background', 'motivation']
    for key in required_protagonist_keys:
        if key not in protagonist:
            logger.warning(f"Missing protagonist key: {key}")
    
    logger.info("Story configuration validation passed")
    return True


def load_configuration_file(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file format is unsupported
        FileNotFoundError: If file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def merge_configurations(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configurations(merged[key], value)
        else:
            merged[key] = value
    
    return merged


async def setup_output_directory(output_config: Dict[str, Any]) -> Path:
    """
    Setup and prepare output directory structure.
    
    Args:
        output_config: Output configuration dictionary
        
    Returns:
        Path to the prepared output directory
    """
    od = output_config.get('base_directory')
    if od is None:
        od = './output'
    else:
        od = substitute_env_vars(od)
    base_dir = Path(od)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories if specified
    subdirs = output_config.get('create_subdirectories', [])
    for subdir in subdirs:
        (base_dir / subdir).mkdir(exist_ok=True)
    
    return base_dir


def get_pipeline_version_info() -> Dict[str, str]:
    """
    Get version information for the pipeline.
    
    Returns:
        Dictionary containing version information
    """
    return {
        'pipeline_version': '3.0.0',
        'component_interface_version': '1.0.0',
        'registry_version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }


class PipelineConfigurationError(Exception):
    """Exception raised for pipeline configuration errors."""
    pass


class PipelineExecutionError(Exception):
    """Exception raised for pipeline execution errors.""" 
    pass


# Export main functions for easy importing
__all__ = [
    'save_pipeline_results',
    'validate_pipeline_configuration', 
    'validate_story_configuration',
    'load_configuration_file',
    'merge_configurations',
    'setup_output_directory',
    'get_pipeline_version_info',
    'PipelineConfigurationError',
    'PipelineExecutionError'
]