"""
Pipeline Activity Logger Module

Provides specialized logging capabilities for tracking pipeline activities,
performance metrics, and comprehensive monitoring of the MuseQuill V3 pipeline.
"""

import logging
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any


class PipelineActivityLogger:
    """
    Specialized logger for tracking pipeline activities and metrics.
    
    This class provides comprehensive logging and monitoring capabilities for
    the MuseQuill pipeline, including activity tracking, performance metrics,
    error logging, and final report generation.
    """
    
    def __init__(self, log_dir: Path = Path("logs")):
        """
        Initialize the pipeline activity logger.
        
        Args:
            log_dir: Directory where log files will be stored
        """
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
        
        # Ensure no duplicate handlers
        if not self.logger.handlers:
            # File handler for activity logs
            activity_handler = logging.FileHandler(self.activity_log_file)
            activity_handler.setFormatter(logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "%(name)s", "message": %(message)s}'
            ))
            self.logger.addHandler(activity_handler)
    
    def log_activity(self, activity_type: str, component: str, details: Dict[str, Any]):
        """
        Log a pipeline activity with structured data.
        
        Args:
            activity_type: Type of activity being logged
            component: Component generating the activity
            details: Additional details about the activity
        """
        activity = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'activity_type': activity_type,
            'component': component,
            'details': details
        }
        
        self.activities.append(activity)
        self.logger.info(json.dumps(activity, default=str))
    
    def log_stage_start(self, stage_name: str, input_data: Dict[str, Any]):
        """
        Log the start of a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            input_data: Input data for the stage
        """
        stage_info = {
            'stage': stage_name,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'input_summary': self._summarize_input(input_data)
        }
        
        self.performance_metrics['pipeline_stages'].append(stage_info)
        self.log_activity('stage_start', 'pipeline', stage_info)
    
    def log_stage_complete(self, stage_name: str, output_data: Dict[str, Any], execution_time: float):
        """
        Log the completion of a pipeline stage.
        
        Args:
            stage_name: Name of the completed stage
            output_data: Output data from the stage
            execution_time: Time taken to execute the stage
        """
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
    
    def log_stage_error(self, stage_name: str, error: Exception, execution_time: float):
        """
        Log an error during a pipeline stage.
        
        Args:
            stage_name: Name of the stage where error occurred
            error: The exception that occurred
            execution_time: Time elapsed before error
        """
        # Find the matching stage entry
        for stage in reversed(self.performance_metrics['pipeline_stages']):
            if stage['stage'] == stage_name and 'end_time' not in stage:
                stage['end_time'] = datetime.now(timezone.utc).isoformat()
                stage['execution_time_seconds'] = execution_time
                stage['error'] = {
                    'type': type(error).__name__,
                    'message': str(error)
                }
                break
        
        self.log_activity('stage_error', 'pipeline', {
            'stage': stage_name,
            'execution_time': execution_time,
            'error_type': type(error).__name__,
            'error_message': str(error)
        })
    
    def log_component_performance(self, component_name: str, metrics: Dict[str, Any]):
        """
        Log component performance metrics.
        
        Args:
            component_name: Name of the component
            metrics: Performance metrics dictionary
        """
        self.performance_metrics['component_metrics'][component_name] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **metrics
        }
        
        self.log_activity('component_performance', component_name, metrics)
    
    def log_research_activity(self, research_type: str, query: str, results: Dict[str, Any]):
        """
        Log research activities.
        
        Args:
            research_type: Type of research performed
            query: Research query used
            results: Results from the research
        """
        research_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'research_type': research_type,
            'query': query,
            'results_summary': self._summarize_research_results(results)
        }
        
        self.performance_metrics['research_activities'].append(research_entry)
        self.log_activity('research', 'researcher', research_entry)
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any]):
        """
        Log error events with full context.
        
        Args:
            component: Component where error occurred
            error: The exception that occurred
            context: Additional context about the error
        """
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
        """
        Log quality assessment results.
        
        Args:
            chapter_id: ID of the chapter being assessed
            assessment: Quality assessment results
        """
        quality_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'chapter_id': chapter_id,
            'assessment': assessment
        }
        
        self.performance_metrics['quality_assessments'].append(quality_entry)
        self.log_activity('quality_assessment', 'quality_controller', quality_entry)
    
    def log_component_lifecycle(self, component_name: str, lifecycle_event: str, details: Dict[str, Any]):
        """
        Log component lifecycle events (start, stop, recycle, etc.).
        
        Args:
            component_name: Name of the component
            lifecycle_event: Type of lifecycle event
            details: Additional details about the event
        """
        self.log_activity('component_lifecycle', component_name, {
            'lifecycle_event': lifecycle_event,
            **details
        })
    
    def log_registry_operation(self, operation: str, component_type: str, details: Dict[str, Any]):
        """
        Log component registry operations.
        
        Args:
            operation: Type of registry operation
            component_type: Type of component being operated on
            details: Additional operation details
        """
        self.log_activity('registry_operation', 'component_registry', {
            'operation': operation,
            'component_type': component_type,
            **details
        })
    
    def save_final_report(self):
        """Save final performance report to file."""
        self.performance_metrics['end_time'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate summary statistics
        self.performance_metrics['summary'] = self._calculate_summary_stats()
        
        # Save performance metrics
        with open(self.performance_log_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        # Save activity log
        activity_summary = {
            'total_activities': len(self.activities),
            'activity_types': list(set(a['activity_type'] for a in self.activities)),
            'components': list(set(a['component'] for a in self.activities)),
            'activities': self.activities
        }
        
        activity_file = self.log_dir / f"final_activity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(activity_file, 'w') as f:
            json.dump(activity_summary, f, indent=2, default=str)
        
        logging.getLogger('main').info(f"Final reports saved:")
        logging.getLogger('main').info(f"  Performance: {self.performance_log_file}")
        logging.getLogger('main').info(f"  Activities: {activity_file}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get current performance summary.
        
        Returns:
            Dictionary containing current performance metrics
        """
        return {
            'total_stages': len(self.performance_metrics['pipeline_stages']),
            'completed_stages': len([s for s in self.performance_metrics['pipeline_stages'] if 'end_time' in s]),
            'total_errors': len(self.performance_metrics['error_events']),
            'total_research': len(self.performance_metrics['research_activities']),
            'active_components': list(self.performance_metrics['component_metrics'].keys()),
            'quality_assessments': len(self.performance_metrics['quality_assessments'])
        }
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of input data for logging.
        
        Args:
            input_data: Input data to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            'keys': list(input_data.keys()),
            'data_types': {k: type(v).__name__ for k, v in input_data.items()},
            'size_estimate': len(str(input_data))
        }
    
    def _summarize_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of output data for logging.
        
        Args:
            output_data: Output data to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            'keys': list(output_data.keys()),
            'data_types': {k: type(v).__name__ for k, v in output_data.items()},
            'size_estimate': len(str(output_data))
        }
    
    def _summarize_research_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of research results.
        
        Args:
            results: Research results to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            'status': results.get('status', 'unknown'),
            'sources_found': results.get('sources_found', 0),
            'execution_time': results.get('execution_time', 0),
            'has_summary': bool(results.get('summary')),
            'result_size': len(str(results))
        }
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the pipeline run.
        
        Returns:
            Summary statistics dictionary
        """
        total_stages = len(self.performance_metrics['pipeline_stages'])
        completed_stages = len([s for s in self.performance_metrics['pipeline_stages'] if 'end_time' in s and 'error' not in s])
        error_stages = len([s for s in self.performance_metrics['pipeline_stages'] if 'error' in s])
        total_errors = len(self.performance_metrics['error_events'])
        total_research = len(self.performance_metrics['research_activities'])
        
        # Calculate total execution time if available
        total_time = 0
        if 'end_time' in self.performance_metrics:
            try:
                start_time = datetime.fromisoformat(self.performance_metrics['start_time'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(self.performance_metrics['end_time'].replace('Z', '+00:00'))
                total_time = (end_time - start_time).total_seconds()
            except:
                total_time = 0
        
        return {
            'total_execution_time_seconds': total_time,
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'error_stages': error_stages,
            'stage_completion_rate': completed_stages / max(total_stages, 1),
            'stage_error_rate': error_stages / max(total_stages, 1),
            'total_errors': total_errors,
            'total_research_activities': total_research,
            'components_used': list(self.performance_metrics['component_metrics'].keys()),
            'quality_assessments_performed': len(self.performance_metrics['quality_assessments']),
            'total_activities_logged': len(self.activities)
        }


# Convenience functions for creating activity loggers

def create_activity_logger(log_dir: Path = Path("logs")) -> PipelineActivityLogger:
    """
    Create a new pipeline activity logger instance.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        New PipelineActivityLogger instance
    """
    return PipelineActivityLogger(log_dir)


def setup_pipeline_logging(log_level: str = "INFO", log_dir: Path = Path("logs")) -> PipelineActivityLogger:
    """
    Setup comprehensive pipeline logging with activity tracking.
    
    This function creates both standard Python logging and the specialized
    activity logger for pipeline monitoring.
    
    Args:
        log_level: Standard logging level
        log_dir: Directory for log files
        
    Returns:
        Configured PipelineActivityLogger instance
    """
    # Create log directory
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create and return activity logger
    activity_logger = PipelineActivityLogger(log_dir)
    
    logging.getLogger('main').info(f"Pipeline logging initialized - Logs in: {log_dir}")
    
    return activity_logger