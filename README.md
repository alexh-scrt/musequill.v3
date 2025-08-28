# ===== README.md =====

# MuseQuill V3 Pipeline Launcher

A comprehensive adversarial book generation system with integrated research capabilities.

## Features

- **Adversarial Generation**: Generator-Discriminator architecture for high-quality content
- **Integrated Research**: Real-time web search and market intelligence via Tavily
- **Multi-Layered Quality Control**: Plot coherence, literary quality, and reader engagement critics
- **Comprehensive Logging**: Activity tracking, performance metrics, and detailed debugging
- **Flexible Configuration**: YAML/JSON configuration with environment variable support
- **Graceful Shutdown**: Proper cleanup and final report generation

## Quick Start

### 1. Installation

```bash
# Clone the repository
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

# Edit .env with your API keys
nano .env  # Add your TAVILY_API_KEY and other credentials
```

### 3. Run Examples

```bash
# Run with built-in example
python main.py --example

# Run with custom configuration
python main.py --config config.yaml --story example_story_config.json

# Validate configuration
python main.py --validate-config --config config.yaml

# Debug mode with verbose logging
python main.py --example --log-level DEBUG
```

## Command Line Options

```
usage: main.py [options]

Options:
  --config CONFIG       Pipeline configuration file (.yaml or .json)
  --story STORY         Story configuration file (.json)
  --example             Run with example configuration
  --validate-config     Validate configuration and exit
  --log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
  --log-dir DIR         Directory for log files
  --output-dir DIR      Override output directory
```

## Configuration

### Pipeline Configuration (config.yaml)

The main configuration file controls all aspects of the pipeline:

- **Pipeline Settings**: Generation attempts, revision cycles, orchestration strategy
- **Research Settings**: Tavily integration, caching, automatic triggers
- **Component Settings**: Individual component configurations and thresholds
- **Output Settings**: File formats, directory structure, naming patterns
- **Logging Settings**: Log levels, rotation, activity tracking

### Story Configuration (story_config.json)

Defines the story to generate:

- **Basic Info**: Title, genre, target audience, estimated length
- **Characters**: Protagonist and supporting characters with motivations
- **Plot Structure**: Three-act structure, pacing, themes
- **Settings**: Time period, locations, technology level
- **Quality Targets**: Minimum thresholds for various quality metrics

## Research Integration

The pipeline includes intelligent research capabilities:

### Automatic Research Triggers
- Market data becomes stale (configurable hours)
- Plot consistency drops below threshold
- Character development gaps detected
- Daily trend monitoring

### Manual Research
```python
# Quick search during pipeline execution
response = await orchestrator.manual_research(
    query="thriller market trends 2025",
    component_name="market_analyzer"
)

# Deep research with specific questions
response = await orchestrator.researcher.deep_research(
    topic="character development techniques",
    specific_questions=[
        "What makes characters memorable?",
        "How do readers connect with protagonists?"
    ]
)
```

## Output Structure

Generated content is saved in timestamped directories:

```
output/
├── the_neural_network_20241201_143022/
│   ├── complete_story_state.json      # Full pipeline state
│   ├── story_configuration.json       # Original story config
│   ├── chapters/                      # Individual chapter files
│   │   ├── chapter_01.txt
│   │   └── chapter_02.txt
│   ├── research_summary.json          # Research findings
│   ├── market_intelligence.json       # Market analysis
│   └── quality_assessments.json       # Quality metrics
```

## Logging and Monitoring

### Log Files
- `pipeline_TIMESTAMP.log` - Complete execution log
- `pipeline_activity_TIMESTAMP.json` - Structured activity log
- `performance_metrics_TIMESTAMP.json` - Performance metrics

### Console Output
Color-coded console output with:
- Pipeline stage progress
- Research activities
- Component performance
- Error and warning messages

### Activity Tracking
Detailed tracking of:
- Pipeline stage execution times
- Component performance metrics
- Research query results
- Quality assessment outcomes
- Error events with full context

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black main.py pipeline_integration.py
isort main.py pipeline_integration.py
```

### Type Checking
```bash
mypy main.py
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```bash
   # Error: Tavily API key not configured
   # Solution: Set TAVILY_API_KEY in .env file
   ```

2. **Component Initialization Failures**
   ```bash
   # Check logs for specific component errors
   python main.py --example --log-level DEBUG
   ```

3. **Memory Issues**
   ```bash
   # Reduce concurrent processing in config.yaml
   performance:
     max_concurrent_chapters: 1
     memory_limit_gb: 4
   ```

### Log Analysis
```bash
# View real-time logs
tail -f logs/pipeline_*.log

# Search for errors
grep -i error logs/pipeline_*.log

# Analyze performance metrics
python -c "import json; print(json.load(open('logs/performance_metrics_*.json'))['summary'])"
```

## Advanced Usage

### Custom Components
Add custom components by inheriting from `ResearchEnabledComponent`:

```python
class MyCustomComponent(ResearchEnabledComponent):
    async def research_before_execution(self, input_data):
        return await self.quick_research("custom research query")
    
    async def main_execute(self, input_data):
        # Your component logic here
        pass
```

### Pipeline Orchestration
Customize orchestration behavior:

```python
# Add custom research rules
orchestrator.add_research_rule(ResearchRule(
    trigger=ResearchTrigger.CUSTOM,
    condition=lambda state: your_condition(state),
    query_template="your research query template",
    scope=ResearchScope.TARGETED
))
```