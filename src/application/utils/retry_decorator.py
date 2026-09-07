from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import logging

def with_retry(func):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectError))
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logging.info(f"Calling the function {func.__name__} - active retry attempt")
        return await func(*args, **kwargs)
    return wrapper
