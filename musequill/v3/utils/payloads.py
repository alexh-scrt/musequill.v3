import json
import re
from typing import Optional, Dict, Any, Union


def extract_json_from_response(input_str: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from a string response.
    
    This function handles various cases:
    - JSON wrapped in markdown code blocks
    - JSON with leading/trailing text
    - Multiple JSON objects (returns the first valid one)
    - Malformed JSON with common issues
    
    Args:
        input_str: The input string that may contain JSON
        
    Returns:
        Dictionary containing the extracted JSON object, or None if no valid JSON found
        
    Examples:
        >>> extract_json_from_response('```json\n{"key": "value"}\n```')
        {'key': 'value'}
        
        >>> extract_json_from_response('Here is the result: {"status": "success"}')
        {'status': 'success'}
    """
    if not input_str or not isinstance(input_str, str):
        return None
    
    # Trim whitespace
    input_str = input_str.strip()
    
    if not input_str:
        return None

    # Try multiple extraction strategies
    strategies = [
        _extract_from_code_blocks,
        _extract_from_braces,
        _extract_with_regex,
        _extract_multiline_json
    ]
    
    for strategy in strategies:
        try:
            result = strategy(input_str)
            if result is not None:
                return result
        except Exception:
            continue
    
    return None


def _extract_from_code_blocks(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from markdown code blocks."""
    # Look for ```json or ``` code blocks
    patterns = [
        r'```json\s*(\{[\s\S]*?\})\s*```',
        r'```\s*\n(.*?)\n```',
        r'`(.*?)`'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                cleaned = match.strip()
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    return json.loads(cleaned)
            except (json.JSONDecodeError, AttributeError):
                continue
    
    return None


def _extract_from_braces(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON by finding the first complete JSON object with braces."""
    # Find the first opening brace
    start = text.find('{')
    if start == -1:
        return None
    
    # Find the matching closing brace
    brace_count = 0
    end = start
    
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    
    if brace_count != 0:
        # Try to find the last closing brace
        end = text.rfind('}') + 1
        if end <= start:
            return None
    
    try:
        json_str = text[start:end]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _extract_with_regex(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON using regex patterns."""
    # Pattern to match JSON objects (basic)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    return None


def _extract_multiline_json(text: str) -> Optional[Dict[str, Any]]:
    """Handle multiline JSON with proper brace matching."""
    lines = text.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        line = line.strip()
        
        # Start collecting when we see an opening brace
        if '{' in line and not in_json:
            in_json = True
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
        elif in_json:
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            # Stop when braces are balanced
            if brace_count == 0:
                break
    
    if json_lines and brace_count == 0:
        try:
            json_str = '\n'.join(json_lines)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


# Additional utility functions
def extract_json_array_from_response(input_str: str) -> Optional[list]:
    """
    Extract JSON array from a string response.
    
    Args:
        input_str: The input string that may contain a JSON array
        
    Returns:
        List containing the extracted JSON array, or None if no valid JSON array found
    """
    if not input_str or not isinstance(input_str, str):
        return None
    
    input_str = input_str.strip()
    
    # Find array brackets
    start = input_str.find('[')
    end = input_str.rfind(']') + 1
    
    if start >= 0 and end > start:
        try:
            json_str = input_str[start:end]
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    
    return None


def is_valid_json(text: str) -> bool:
    """
    Check if a string contains valid JSON.
    
    Args:
        text: String to validate
        
    Returns:
        True if the string contains valid JSON, False otherwise
    """
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, AttributeError):
        return False


def clean_json_string(text: str) -> str:
    """
    Clean a JSON string by removing common formatting issues.
    
    Args:
        text: Raw JSON string to clean
        
    Returns:
        Cleaned JSON string
    """
    if not text:
        return text
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*\n?', '', text)
    text = re.sub(r'^`|`$', '', text)
    
    # Remove common prefixes
    prefixes_to_remove = [
        'Here is the JSON:',
        'Here\'s the JSON:',
        'JSON:',
        'Response:',
        'Result:'
    ]
    
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    
    return text


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Regular JSON
        '{"key": "value", "number": 42}',
        
        # JSON in code blocks
        '```json\n{"status": "success", "data": [1, 2, 3]}\n```',
        
        # JSON with surrounding text
        'Here is the result: {"message": "Hello World"} and that\'s it.',
        
        # Multiline JSON
        '''```json
        {
            "name": "Peter",
            "type": "bunny",
            "adventures": [
                "forest exploration",
                "meeting mythical creatures"
            ]
        }
        ```''',
        
        # Empty or invalid cases
        '',
        'No JSON here',
        'Just text without any braces',
    ]
    
    print("Testing extract_json_from_response function:\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Input: {repr(test_case[:50] + '...' if len(test_case) > 50 else test_case)}")
        
        result = extract_json_from_response(test_case)
        print(f"Output: {result}")
        print("-" * 50)