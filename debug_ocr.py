
import re

def parse_metadata_text(text: str):
    print(f"Testing text: '{text}'")
    result = {
        'temperature': None,
        'date': None,
        'time': None
    }
    
    # Original Patterns
    # temp_pattern = r'(\d+\.?\d*)\s*[°]?[CF]'
    # time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)'

    # Proposed Patterns
    # Allow ' " . ° before C/F
    temp_pattern = r'(\d+\.?\d*)\s*[°\'"\.]?[CF]' 
    
    # Allow . or : as separator
    time_pattern = r'(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s*[AP]M)?)' 

    date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})'
    
    # Extract temperature
    temp_match = re.search(temp_pattern, text, re.IGNORECASE)
    if temp_match:
        result['temperature'] = temp_match.group(0)
    
    # Extract date
    date_match = re.search(date_pattern, text)
    if date_match:
        result['date'] = date_match.group(1)
    
    # Extract time
    time_match = re.search(time_pattern, text, re.IGNORECASE)
    if time_match:
        result['time'] = time_match.group(1)
    
    print("Result:", result)
    return result

text = "Meidase M 16 'C 03/12/2025 00.37:05 0003"
parse_metadata_text(text)
