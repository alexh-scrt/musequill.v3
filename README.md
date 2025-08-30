# MuseQuill V3: Adversarial Book Generation Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Pipeline Status](https://img.shields.io/badge/pipeline-production--ready-brightgreen.svg)]()

> *Transform book writing with AI-powered adversarial generation*

MuseQuill V3 is a sophisticated adversarial book generation system that uses Generator-Discriminator architecture inspired by GANs to create high-quality, commercially viable literature. The system addresses critical issues in AI-generated content: repetitive prose, shallow character development, formulaic language, and poor pacing.

## 🚀 Key Features

### ✨ **Adversarial Quality Control**
- **Multi-Layered Critics**: Plot Coherence, Literary Quality, Reader Engagement, and LLM Discriminator
- **Dynamic Quality Gates**: Adaptive thresholds based on story position and market intelligence
- **Intelligent Revision Cycles**: Targeted feedback with progressive refinement

### 🧠 **Market Intelligence Integration**
- **Real-Time Research**: Tavily API integration for current market trends
- **Commercial Viability**: Reader preference analysis and engagement prediction
- **Competitive Intelligence**: Genre-specific success pattern identification

### 📚 **Advanced Story Management**
- **Dynamic State Tracking**: Plot threads, character arcs, world consistency
- **Multi-Variant Generation**: 2-3 approaches per chapter with best selection
- **Character Development**: Voice consistency and relationship dynamics

### 🔧 **Production-Ready Architecture**
- **Component-Based Design**: Modular, scalable, and testable architecture
- **Health Monitoring**: Comprehensive component lifecycle management
- **Research Integration**: Intelligent research triggers and caching

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB available space for models and cache
- **API Access**: Tavily API key for market research

## ⚡ Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd musequill-v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Configure API keys (required)
nano .env
```

**Required Environment Variables:**
```bash
TAVILY_API_KEY=your_tavily_api_key_here
# Optional: Add other service API keys
```

### 3. Run Your First Book Generation

```bash
# Quick start with example configuration
python main.py --example

# Custom configuration
python main.py --config config.yaml --story your_story_config.json

# Validate configuration before running
python main.py --validate-config --config config.yaml

# Debug mode with detailed logging
python main.py --example --log-level DEBUG
```

## 🎯 Usage Examples

### Basic Book Generation

```python
from musequill.v3.pipeline_runner import EnhancedPipelineRunner

# Initialize pipeline
runner = EnhancedPipelineRunner(config_path="config.yaml")
await runner.initialize_pipeline()

# Define your story
story_config = {
    "title": "The Neural Awakening",
    "genre": "science_fiction",
    "target_audience": "adult",
    "estimated_length": "novel",
    "protagonist": {
        "name": "Dr. Sarah Chen",
        "motivation": "Discover the truth about AI consciousness"
    },
    "themes": ["artificial intelligence", "consciousness", "identity"]
}

# Generate the book
results = await runner.execute_pipeline(story_config)
```

### Advanced Configuration

```python
# Custom orchestration strategy
config = {
    "pipeline_orchestrator": {
        "orchestration_strategy": "quality_first",
        "max_revision_cycles": 5,
        "enable_market_intelligence_refresh": True
    },
    "research": {
        "enable_research_integration": True,
        "market_refresh_interval_hours": 12,
        "max_research_queries_per_session": 20
    },
    "quality_control": {
        "plot_coherence_threshold": 0.75,
        "literary_quality_threshold": 0.70,
        "reader_engagement_threshold": 0.80
    }
}
```

## 📁 Project Structure

```
musequill-v3/
├── musequill/
│   └── v3/
│       ├── components/              # Core pipeline components
│       │   ├── generators/          # Chapter generation
│       │   ├── discriminators/      # Quality critics
│       │   ├── market_intelligence/ # Market research
│       │   ├── quality_control/     # Quality management
│       │   ├── orchestration/       # Pipeline orchestration
│       │   └── character_developer/ # Character development
│       ├── models/                  # Data models and schemas
│       ├── pipeline_runner.py       # Main pipeline execution
│       ├── pipeline_integration.py  # Integration utilities
│       └── pipeline_configuration.py # Configuration management
├── config.yaml                     # Default pipeline configuration
├── main.py                         # Command-line interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## ⚙️ Command Line Interface

```bash
usage: main.py [options]

Options:
  --config CONFIG       Pipeline configuration file (.yaml or .json)
  --story STORY         Story configuration file (.json)
  --example             Run with built-in example configuration
  --validate-config     Validate configuration and exit
  --log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
  --log-dir DIR         Directory for log files (default: logs/)
  --output-dir DIR      Override output directory
```

## 🔧 Configuration

### Pipeline Configuration (config.yaml)

The main configuration controls all pipeline aspects:

```yaml
# Core Pipeline Settings
pipeline_orchestrator:
  orchestration_strategy: "balanced"  # quality_first, speed_optimized, balanced, experimental
  max_revision_cycles: 3
  enable_market_intelligence_refresh: true
  component_health_check_interval: 300

# Research Integration
research:
  enable_research_integration: true
  market_refresh_interval_hours: 24
  enable_automatic_research_triggers: true
  max_research_queries_per_session: 15
  research_cache_ttl_hours: 48

# Component Configurations
chapter_generator:
  max_variants_per_chapter: 3
  enable_adaptive_parameters: true
  generation_temperature: 0.8

quality_control:
  plot_coherence_threshold: 0.7
  literary_quality_threshold: 0.6
  reader_engagement_threshold: 0.75
  enable_dynamic_thresholds: true

# Output Settings
output:
  base_directory: "./output"
  save_intermediate_results: true
  create_timestamped_directories: true
```

### Story Configuration

Define your story parameters:

```json
{
  "title": "Your Story Title",
  "genre": "thriller",
  "target_audience": "adult",
  "estimated_length": "novel",
  "protagonist": {
    "name": "Character Name",
    "motivation": "Character's driving motivation",
    "background": "Character background"
  },
  "supporting_characters": [
    {
      "name": "Supporting Character",
      "role": "ally",
      "relationship_to_protagonist": "mentor"
    }
  ],
  "plot_structure": {
    "act_one_length": 0.25,
    "act_two_length": 0.5,
    "act_three_length": 0.25
  },
  "themes": ["primary theme", "secondary theme"],
  "settings": [
    {
      "name": "Primary Location",
      "time_period": "contemporary",
      "description": "Setting description"
    }
  ]
}
```

## 🔍 Research Integration

### Automatic Research Triggers
- **Market Staleness**: Triggers when market data exceeds configured age
- **Quality Issues**: Activates when quality metrics drop below thresholds
- **Plot Consistency**: Engages when story logic inconsistencies are detected
- **Character Development**: Initiates when character arcs show gaps

### Manual Research
```python
# Quick targeted research
response = await orchestrator.manual_research(
    query="contemporary thriller market trends 2025",
    component_name="market_analyzer"
)

# Deep research with specific questions
response = await orchestrator.researcher.deep_research(
    topic="character development in thrillers",
    specific_questions=[
        "What makes thriller protagonists memorable?",
        "How do readers connect with complex characters?",
        "What character flaws drive engagement in thrillers?"
    ]
)
```

## 📊 Output Structure

Generated content is organized in timestamped directories:

```
output/
├── your_story_title_20250829_143022/
│   ├── pipeline_results.json          # Complete pipeline state
│   ├── story_configuration.json       # Original story config
│   ├── execution_summary.json         # Execution metrics
│   ├── chapters/                      # Individual chapter files
│   │   ├── chapter_01.txt
│   │   ├── chapter_02.txt
│   │   └── ...
│   ├── research_summary.json          # Research findings
│   ├── market_intelligence.json       # Market analysis
│   ├── quality_assessments.json       # Quality metrics
│   ├── component_performance.json     # Component metrics
│   └── README.md                      # Human-readable summary
```

## 📈 Performance Monitoring

### Quality Metrics Tracking
- **Repetition Reduction**: Target 70% decrease in repetitive content
- **Character Consistency**: Achieve 85% voice consistency across chapters
- **Pacing Optimization**: 60% improvement in story momentum tracking
- **Language Quality**: 50% reduction in formulaic expressions

### System Monitoring
- **Component Health**: Real-time health checks and performance tracking
- **Resource Utilization**: Memory and CPU usage monitoring
- **API Usage**: Research API call tracking and optimization
- **Error Rates**: Comprehensive error analysis and recovery metrics

## 🧪 Development and Testing

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_components/ -v        # Component tests
pytest tests/test_integration/ -v       # Integration tests
pytest tests/test_pipeline/ -v          # Pipeline tests

# Run with coverage
pytest tests/ --cov=musequill --cov-report=html
```

### Code Quality
```bash
# Format code
black musequill/ main.py
isort musequill/ main.py

# Type checking
mypy musequill/

# Linting
pylint musequill/
```

### Component Development
```python
from musequill.v3.components.base.component_interface import BaseComponent

class CustomComponent(BaseComponent):
    async def initialize(self) -> bool:
        # Initialize your component
        return True
    
    async def process(self, input_data):
        # Process input and return results
        pass
    
    async def health_check(self):
        # Return health status
        return ComponentHealth.HEALTHY
    
    async def cleanup(self) -> bool:
        # Cleanup resources
        return True
```

## 🔧 Troubleshooting

### Common Issues

**API Configuration**:
```bash
# Error: Tavily API key not configured
# Solution: Set TAVILY_API_KEY in .env file
echo "TAVILY_API_KEY=your_key_here" >> .env
```

**Component Initialization**:
```bash
# Check component status
python main.py --example --log-level DEBUG
# Review logs/pipeline_*.log for detailed error information
```

**Memory Issues**:
```yaml
# Reduce concurrent processing in config.yaml
performance:
  max_concurrent_chapters: 1
  memory_limit_gb: 4
  component_recycling_threshold: 50
```

**Quality Issues**:
```yaml
# Adjust quality thresholds
quality_control:
  plot_coherence_threshold: 0.6  # Lower for more permissive
  enable_fallback_strategies: true
```

### Log Analysis
```bash
# Real-time monitoring
tail -f logs/pipeline_*.log

# Error analysis
grep -i error logs/pipeline_*.log | tail -20

# Performance metrics
python -c "
import json
from pathlib import Path
log_files = list(Path('logs').glob('performance_metrics_*.json'))
if log_files:
    with open(log_files[-1]) as f:
        data = json.load(f)
        print(f'Average execution time: {data[\"average_execution_time\"]}s')
        print(f'Success rate: {data[\"success_rate\"]:.1%}')
"
```

## 🚀 Advanced Usage

### Custom Orchestration Strategies
```python
from musequill.v3.components.orchestration.pipeline_orchestrator import (
    OrchestrationStrategy, PipelineOrchestratorConfig
)

# Create custom orchestration configuration
config = PipelineOrchestratorConfig(
    orchestration_strategy=OrchestrationStrategy.EXPERIMENTAL,
    max_revision_cycles=5,
    enable_market_intelligence_refresh=True,
    parallel_variant_evaluation=True,
    component_health_check_interval=60
)
```

### Research Rule Customization
```python
# Add custom research triggers
from musequill.v3.models.research_models import ResearchRule, ResearchTrigger

custom_rule = ResearchRule(
    trigger=ResearchTrigger.CUSTOM,
    condition=lambda state: state.current_chapter > 10,
    query_template="late-stage {genre} plot development strategies",
    priority=ResearchPriority.HIGH
)

orchestrator.add_research_rule(custom_rule)
```

### Quality Threshold Adaptation
```python
# Dynamic quality adjustment
def custom_threshold_calculator(story_position, genre, market_data):
    base_thresholds = {"plot_coherence": 0.7, "literary_quality": 0.6}
    
    if story_position > 0.8:  # Climax chapters
        base_thresholds["reader_engagement"] = 0.9
    
    return base_thresholds

orchestrator.set_threshold_calculator(custom_threshold_calculator)
```

## 🤝 Contributing

We welcome contributions to MuseQuill V3! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes with tests
5. Submit a pull request

### Areas for Contribution
- **New Critics**: Additional quality assessment components
- **Market Intelligence**: Enhanced research and trend analysis
- **Performance Optimization**: Speed and resource improvements
- **Documentation**: Examples, tutorials, and guides

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Tavily API**: For providing real-time market research capabilities
- **ChromaDB**: For efficient vector storage and retrieval
- **Pydantic**: For robust data validation and serialization
- **The AI Writing Community**: For inspiration and feedback

## 📞 Support

- **Documentation**: Full documentation available at [docs link]
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join our community discussions for help and ideas
- **Email**: Contact us at [support email] for technical support

---

*MuseQuill V3 - Where AI meets creative storytelling* ✨