#!/usr/bin/env python3
"""
Text file cleaner that removes all characters except alphanumeric and basic punctuation.
Outputs cleaned file with '_cleaned.txt' suffix in the same directory.
"""

import re
import sys
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean text by keeping only alphanumeric characters and basic punctuation.
    
    Allowed characters:
    - Alphanumeric (a-z, A-Z, 0-9)
    - Basic punctuation: . , ! " ( )
    - Whitespace (spaces, tabs, newlines)
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Keep alphanumeric, whitespace, and specified punctuation: . , ! " ( )
    # Pattern explanation: [^...] means "NOT these characters"
    # So we remove everything that's NOT in the specified set
    pattern = r'[^a-zA-Z0-9\s.,!()\"]'
    cleaned = re.sub(pattern, '', text)
    
    # Optionally clean up multiple consecutive spaces/newlines
    # Remove multiple spaces (but keep single spaces)
    cleaned = re.sub(r' +', ' ', cleaned)
    # Remove multiple newlines (keep max 2 for paragraph breaks)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned


def clean_file(input_path: str) -> None:
    """
    Clean a text file and save the result with '_cleaned.txt' suffix.
    
    Args:
        input_path: Path to the input text file
    """
    input_file = Path(input_path)
    
    # Validate input file exists
    if not input_file.exists():
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)
    
    if not input_file.is_file():
        print(f"Error: '{input_path}' is not a file.")
        sys.exit(1)
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Create output file path
    # Remove existing extension and add '_cleaned.txt'
    output_file = input_file.parent / f"{input_file.stem}_cleaned.txt"
    
    # Write the cleaned text
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"✓ Cleaned file saved to: {output_file}")
        print(f"  Original size: {len(text)} characters")
        print(f"  Cleaned size: {len(cleaned_text)} characters")
        print(f"  Removed: {len(text) - len(cleaned_text)} characters")
    except Exception as e:
        print(f"Error writing file: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python clean_text.py <input_file>")
        print("\nExample:")
        print("  python clean_text.py data/self_reliance.txt")
        sys.exit(1)
    
    input_path = sys.argv[1]
    clean_file(input_path)


if __name__ == "__main__":
    main()

