# monitoring.py
import time

def monitor_llm_call(action_name: str, model_name: str, func, *args, **kwargs):
    """
    Wrap an LLM call and log:
    - latency
    - token usage (OpenAI only)
    - model name
    - success/failure
    """

    start = time.time()

    try:
        response = func(*args, **kwargs)
        duration_ms = round((time.time() - start) * 1000)

        # Token usage: OpenAI returns usage metadata; Ollama returns None
        usage = getattr(response, "usage", None)

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        else:
            prompt_tokens = completion_tokens = total_tokens = None

        print(f"[MONITOR] model={model_name} action={action_name} "
              f"took={duration_ms}ms "
              f"tokens={total_tokens} (prompt={prompt_tokens}, completion={completion_tokens})")

        return response

    except Exception as e:
        duration_ms = round((time.time() - start) * 1000)
        print(f"[MONITOR] ERROR action={action_name} model={model_name} "
              f"took={duration_ms}ms error={repr(e)}")
        raise
