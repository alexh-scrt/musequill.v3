from datetime import datetime
import os

def generate_filename(directory: str, prefix: str, extension: str) -> str:
    """
    Generate a filename with timestamp in the format: dir/prefix-{datetime}.extension
    
    Args:
        directory: Directory path where the file will be located
        prefix: Prefix for the filename
        extension: File extension (with or without leading dot)
    
    Returns:
        Complete file path as string
    """
    # Get current timestamp in a filesystem-friendly format
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Ensure extension has a leading dot
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Create filename
    filename = f"{prefix}-{timestamp}{extension}"
    
    # Join with directory path
    return os.path.join(directory, filename)

# Example usage:
if __name__ == "__main__":
    # Examples
    print(generate_filename("./logs", "app_log", "txt"))
    # Output: ./logs/app_log-20250731-143052.txt
    
    print(generate_filename("/tmp/data", "experiment", ".json"))
    # Output: /tmp/data/experiment-20250731-143052.json
    
    print(generate_filename("output", "model_results", "csv"))
    # Output: output/model_results-20250731-143052.csv