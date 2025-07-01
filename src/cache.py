from joblib import Memory
from src.config import get_settings

settings = get_settings()
# Set up caching directory - uses ".joblib_cache" if caching is enabled,
# otherwise None
cache_dir = ".joblib_cache" if settings.CACHE_RESULTS else None
# Initialize joblib Memory object for function result caching
memory = Memory(location=cache_dir, verbose=10)