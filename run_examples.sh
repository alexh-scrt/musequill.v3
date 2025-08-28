# ===== run_examples.sh =====
#!/bin/bash
# Example execution scripts for MuseQuill V3 Pipeline

echo "MuseQuill V3 Pipeline Examples"
echo "==============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "requirements.installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch requirements.installed
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from template..."
    cp .env.example .env
    echo "Please edit .env with your API keys before continuing."
    exit 1
fi

echo ""
echo "Available Examples:"
echo "1. Basic Example (built-in story)"
echo "2. Custom Configuration"
echo "3. Debug Mode"
echo "4. Configuration Validation"
echo "5. Quick Test Run"
echo ""

read -p "Choose an example (1-5): " choice

case $choice in
    1)
        echo "Running basic example..."
        python main.py --example
        ;;
    2)
        echo "Running with custom configuration..."
        python main.py --config config.yaml --story example_story_config.json
        ;;
    3)
        echo "Running in debug mode..."
        python main.py --example --log-level DEBUG
        ;;
    4)
        echo "Validating configuration..."
        python main.py --validate-config --config config.yaml
        ;;
    5)
        echo "Quick test run..."
        python main.py --example --log-level INFO
        ;;
    *)
        echo "Invalid choice. Please select 1-5."
        exit 1
        ;;
esac

echo ""
echo "Example completed. Check the output/ and logs/ directories for results."