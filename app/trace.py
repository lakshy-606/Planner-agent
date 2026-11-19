"""@traced: turns each node's optional `_trace` info into a timed trace event."""
from __future__ import annotations

import functools
import time


def traced(step: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state):
            t0 = time.perf_counter()
            update = fn(state) or {}
            info = update.pop("_trace", {})
            event = {
                "step": step,
                "status": info.get("status", "ok"),
                "source": info.get("source"),
                "summary": info.get("summary", ""),
                "ms": int((time.perf_counter() - t0) * 1000),
            }
            if info.get("details"):
                event["details"] = info["details"]
            update["trace"] = [event]
            return update
        return wrapper
    return deco
