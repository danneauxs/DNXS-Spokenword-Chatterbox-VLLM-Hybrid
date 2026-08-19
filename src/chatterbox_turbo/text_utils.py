"""
Text processing utilities for Chatterbox TTS
"""
import re
from typing import List


def chunk_text(text: str, max_chars: int = 300) -> List[str]:
    """
    Split text into chunks at sentence boundaries, respecting max_chars limit.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (default 300)
        
    Returns:
        List of text chunks
        
    Examples:
        >>> chunk_text("Hello. World.", max_chars=10)
        ['Hello.', 'World.']
        
        >>> chunk_text("This is a test. Another sentence here.", max_chars=20)
        ['This is a test.', 'Another sentence here.']
    """
    if len(text) <= max_chars:
        return [text]
    
    # Split on sentence boundaries (., !, ?)
    # Keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence exceeds max_chars, split it further
        if len(sentence) > max_chars:
            # If we have accumulated text, save it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence at comma or space boundaries
            sub_chunks = _split_long_sentence(sentence, max_chars)
            chunks.extend(sub_chunks)
        else:
            # Check if adding this sentence would exceed limit
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """
    Split a long sentence at comma or space boundaries.
    
    Args:
        sentence: Long sentence to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of sentence fragments
    """
    # Try splitting at commas first
    if ',' in sentence:
        parts = re.split(r'(,\s*)', sentence)
        # Recombine with commas
        fragments = []
        current = ""
        for i in range(0, len(parts), 2):
            part = parts[i]
            comma = parts[i + 1] if i + 1 < len(parts) else ""
            
            if len(current) + len(part) + len(comma) > max_chars:
                if current:
                    fragments.append(current.strip())
                current = part + comma
            else:
                current += part + comma
        
        if current:
            fragments.append(current.strip())
        
        # Check if any fragment still exceeds limit
        final_fragments = []
        for frag in fragments:
            if len(frag) > max_chars:
                final_fragments.extend(_split_at_spaces(frag, max_chars))
            else:
                final_fragments.append(frag)
        
        return final_fragments
    else:
        # Fall back to splitting at spaces
        return _split_at_spaces(sentence, max_chars)


def _split_at_spaces(text: str, max_chars: int) -> List[str]:
    """
    Split text at space boundaries.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text fragments
    """
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
