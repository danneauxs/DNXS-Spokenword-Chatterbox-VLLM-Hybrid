def search_chunks(chunks, query):
    """Searches through a list of chunks to find those containing a specific query.
    Args:
    chunks (list): A list of dictionaries, each containing a "text" key.
    query (str): The substring to search for within the chunk's "text".
    Returns:
    list: A list of chunks that contain the query.
    """
    results = []
    query_lower = query.lower()

    for chunk in chunks:
        if query_lower in chunk["text"].lower():
            results.append(chunk)

    return results
