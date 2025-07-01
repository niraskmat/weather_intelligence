from src.config import get_settings
import os

settings = get_settings()
print(settings.CACHE_RESULTS)
print(os.getenv("CACHE_RESULTS"))