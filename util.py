from langgraph.types import CachePolicy


def create_cache_policy(ttl: int, static_key: str | None = None) -> CachePolicy:
    """Util to create cache policies for research agents.

    Args:
        ttl: Time to live in seconds
        static_key: If provided, uses a static key. Otherwise, uses ticker.

    Returns:
        CachePolicy instance
    """
    if static_key:
        return CachePolicy(key_func=lambda x: static_key.encode(), ttl=ttl)

    # Handle both dict and object state representations
    # graph drawing in langsmith requires dict representation
    # graph execution requires pydantic object representation
    def key_func(x):
        if isinstance(x, dict):
            # Handle missing keys (e.g., when drawing graph without state)
            ticker = x.get("ticker", "default")
            return f"{ticker}".encode()
        return f"{x.ticker}".encode()

    return CachePolicy(key_func=key_func, ttl=ttl)
