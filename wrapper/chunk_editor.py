def update_chunk(chunk, boundary_type=None, pause_duration=None, sentiment_score=None):
    """Update a chunk with boundary type, pause duration, and sentiment score.
    Args:
    - chunk (dict): The chunk to update.
    - boundary_type (str, optional): The new boundary type for the chunk.
    - pause_duration (int, optional): The new pause duration in seconds.
    - sentiment_score (float, optional): The new sentiment score of the chunk.
    Returns:
    - dict: The updated chunk.
    """
    if boundary_type is not None:
        chunk["boundary_type"] = boundary_type
    if pause_duration is not None:
        chunk["pause_duration"] = pause_duration
    if sentiment_score is not None:
        chunk["sentiment_score"] = sentiment_score
    return chunk
