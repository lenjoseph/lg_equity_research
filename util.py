from langgraph.types import CachePolicy


def create_cache_policy(ttl: int, static_key: str | None = None) -> CachePolicy:
    """Util to create cache policies for research agents.

    Args:
        ttl: Time to live in seconds
        static_key: If provided, uses a static key. Otherwise, uses ticker+duration pattern.

    Returns:
        CachePolicy instance
    """
    if static_key:
        return CachePolicy(key_func=lambda x: static_key.encode(), ttl=ttl)
    return CachePolicy(
        key_func=lambda x: f"{x.ticker}_{x.trade_duration}".encode(), ttl=ttl
    )
