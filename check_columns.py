# demographic_mappings.py
"""
You need to fill in these mappings based on what the codes actually mean.
"""

DEMOGRAPHIC_MAPPINGS = {
    # Example - you need to provide the actual mappings
    'marital_status': {
        '1': 'Single',
        '2': 'Married', 
        '3': 'Divorced',
        '4': 'Widowed',
        '': 'Not Specified'
    },
    
    'dw_channel_key': {
        '0': 'Unknown',
        '1': 'Mobile App',  # You need to specify what '1' actually means
        '2': 'Website',
        '3': 'Branch',
        '4': 'Agent',
        '': 'Not Specified'
    },
    
    'employment_status': {
        '1': 'Employed',
        '2': 'Self-Employed',
        '3': 'Unemployed',
        '4': 'Student',
        '': 'Not Specified'
    },
    
    'purpose': {
        '1': 'Business',
        '2': 'Education',
        '3': 'Medical',
        '4': 'Personal',
        '5': 'Agriculture',
        '13': 'Home Improvement',  # You need to specify what '13' means
        '': 'Not Specified'
    }
}

def decode_demographic(field_name: str, code: str) -> str:
    """Decode a demographic code to its actual meaning."""
    if field_name in DEMOGRAPHIC_MAPPINGS:
        mapping = DEMOGRAPHIC_MAPPINGS[field_name]
        return mapping.get(str(code).strip(), f"{field_name}: {code}")
    return f"{field_name}: {code}"

def parse_combination(combination_str: str) -> str:
    """Parse a combination string like 'marital_status: 1 | dw_channel_key: 1 | ...'"""
    if not combination_str:
        return ""
    
    parts = combination_str.split(' | ')
    decoded_parts = []
    
    for part in parts:
        if ': ' in part:
            field, value = part.split(': ', 1)
            decoded = decode_demographic(field.strip(), value.strip())
            decoded_parts.append(decoded)
        else:
            decoded_parts.append(part)
    
    return ' | '.join(decoded_parts)