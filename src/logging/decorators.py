"""Decorador @audit_llm_call para instrumentar agentes automaticamente."""
import functools
import time
import uuid
from src.logging.audit import log_event, init_db
from src.logging.schemas import LLMCallEvent


def audit_llm_call(agent_name: str, model: str = "biomistral-medquad-lora"):
    """
    Decorador que automaticamente loga toda chamada LLM.

    Uso:
        @audit_llm_call("triagem")
        def triar(relato, llm_client) -> dict:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            init_db()

            session_id = kwargs.get("session_id", str(uuid.uuid4())[:8])
            user_id = kwargs.get("user_id", "anonymous")

            # Capturar input (primeiro argumento str ou dict)
            input_data = args[0] if args else kwargs.get("relato", "")

            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = int((time.time() - t0) * 1000)
                error = None
            except Exception as e:
                latency_ms = int((time.time() - t0) * 1000)
                result = {"error": str(e)}
                error = str(e)

            # Estimar tokens (~4 chars/token)
            tokens_in = len(str(input_data)) // 4
            tokens_out = len(str(result)) // 4

            event = LLMCallEvent(
                event_type="llm_call",
                session_id=session_id,
                user_id=user_id,
                agent=agent_name,
                model=model,
                input_preview=str(input_data)[:500],
                output_preview=str(result)[:500],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                estimated_cost_usd=(tokens_in + tokens_out) * 0.000001,
                metadata={"error": error} if error else {},
            )
            log_event(event)

            if error:
                raise
            return result
        return wrapper
    return decorator