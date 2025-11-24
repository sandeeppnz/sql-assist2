# openai_retry.py
import time
import random
from typing import Callable, Any
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from tqdm import tqdm

MAX_RETRIES = 6
BASE_DELAY = 0.8       # seconds
MAX_DELAY = 8.0        # seconds


def _sleep_with_backoff(retry: int):
    delay = min(MAX_DELAY, BASE_DELAY * (2 ** retry))
    delay = delay * (0.8 + random.random() * 0.4)   # jitter
    time.sleep(delay)


def openai_with_retry(func: Callable, *args, **kwargs) -> Any:
    """
    Calls an OpenAI function with full retry protection:
    - RateLimitError
    - APIError
    - Timeout
    - 5xx server-side failures
    """

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)

        except RateLimitError as e:
            tqdm.write(f"[OpenAI] Rate limit hit. Retry {attempt+1}/{MAX_RETRIES}")
            _sleep_with_backoff(attempt)

        except APITimeoutError as e:
            tqdm.write(f"[OpenAI] Timeout. Retry {attempt+1}/{MAX_RETRIES}")
            _sleep_with_backoff(attempt)

        except APIError as e:
            # Retry only on server-side 5xx
            if e.status_code and 500 <= e.status_code < 600:
                tqdm.write(f"[OpenAI] Server error {e.status_code}. Retry {attempt+1}/{MAX_RETRIES}")
                _sleep_with_backoff(attempt)
            else:
                raise

        except Exception as e:
            # Non-retriable
            tqdm.write("[OpenAI] Non-retryable error:", repr(e))
            raise

    raise RuntimeError("OpenAI API failed after maximum retries.")
# openai_retry.py
import time
import random
from typing import Callable, Any
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

MAX_RETRIES = 6
BASE_DELAY = 0.8       # seconds
MAX_DELAY = 8.0        # seconds


def _sleep_with_backoff(retry: int):
    delay = min(MAX_DELAY, BASE_DELAY * (2 ** retry))
    delay = delay * (0.8 + random.random() * 0.4)   # jitter
    time.sleep(delay)


def openai_with_retry(func: Callable, *args, **kwargs) -> Any:
    """
    Calls an OpenAI function with full retry protection:
    - RateLimitError
    - APIError
    - Timeout
    - 5xx server-side failures
    """

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)

        except RateLimitError as e:
            tqdm.write(f"[OpenAI] Rate limit hit. Retry {attempt+1}/{MAX_RETRIES}")
            _sleep_with_backoff(attempt)

        except APITimeoutError as e:
            tqdm.write(f"[OpenAI] Timeout. Retry {attempt+1}/{MAX_RETRIES}")
            _sleep_with_backoff(attempt)

        except APIError as e:
            # Retry only on server-side 5xx
            if e.status_code and 500 <= e.status_code < 600:
                tqdm.write(f"[OpenAI] Server error {e.status_code}. Retry {attempt+1}/{MAX_RETRIES}")
                _sleep_with_backoff(attempt)
            else:
                raise

        except Exception as e:
            # Non-retriable
            tqdm.write("[OpenAI] Non-retryable error:", repr(e))
            raise

    raise RuntimeError("OpenAI API failed after maximum retries.")
