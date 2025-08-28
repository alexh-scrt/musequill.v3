def seconds_to_time_string(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS string format.
    
    Args:
        seconds: Time in seconds (float)
    
    Returns:
        Time string in HH:MM:SS format
    """
    # Convert to integer seconds for time calculation
    total_seconds = int(seconds)
    
    # Calculate hours, minutes, and remaining seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    # Format as HH:MM:SS
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Example usage:
if __name__ == "__main__":
    print(seconds_to_time_string(3661.5))   # Output: 01:01:01
    print(seconds_to_time_string(125.7))    # Output: 00:02:05
    print(seconds_to_time_string(7200))     # Output: 02:00:00
    print(seconds_to_time_string(45.3))     # Output: 00:00:45