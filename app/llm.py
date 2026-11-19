"""OpenAI access via langchain-openai, plus a heuristic parser used when the LLM is unavailable."""
from __future__ import annotations

import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from . import rules
from .data_sources import get_secret, load_mock
from .models import Preferences

T = TypeVar("T", bound=BaseModel)
DEFAULT_MODEL = "gpt-4.1-mini"


class LLMUnavailable(Exception):
    pass


def model_name() -> str:
    return get_secret("OPENAI_MODEL") or DEFAULT_MODEL


def structured_call(schema: Type[T], system: str, user: str, temperature: float = 0.2) -> T:
    key = get_secret("OPENAI_API_KEY")
    if not key:
        raise LLMUnavailable("OPENAI_API_KEY not set")
    from langchain_openai import ChatOpenAI

    name = model_name()
    kwargs = dict(model=name, api_key=key, timeout=30, max_retries=1)
    # reasoning models (o-series, gpt-5*) only accept the default temperature
    if not (name.startswith("o") or name.startswith("gpt-5")):
        kwargs["temperature"] = temperature
    llm = ChatOpenAI(**kwargs).with_structured_output(schema)
    try:
        out = llm.invoke([("system", system), ("user", user)])
    except Exception as e:  # auth, network, rate limit, bad JSON
        msg = f"{type(e).__name__}: {str(e)[:160]}"
        if any(s in msg for s in ("AuthenticationError", "APIConnectionError", "RateLimitError",
                                  "APITimeoutError", "InternalServerError", "PermissionDenied")):
            raise LLMUnavailable(msg) from e
        raise
    if out is None:
        raise ValueError("empty structured output")
    return out  # type: ignore[return-value]


# ---------------------------------------------------------------- heuristic parsing (no LLM)

def normalise_interests(items: list[str]) -> list[str]:
    out: list[str] = []
    for raw in items:
        s = raw.lower().strip()
        for interest, words in rules.INTEREST_SYNONYMS.items():
            if s == interest or any(w in s for w in words):
                if interest not in out:
                    out.append(interest)
    return out


def parse_duration(text: str) -> Optional[int]:
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr|h)\b", t)
    if m:
        return int(float(m.group(1)) * 60)
    m = re.search(r"(\d+)\s*(?:minutes|mins|min)\b", t)
    if m:
        return int(m.group(1))
    if "half day" in t or "half-day" in t:
        return 240
    if "whole day" in t or "full day" in t or "all day" in t:
        return 600
    return None


def _hour(h: str, mm: Optional[str], ampm: Optional[str]) -> int:
    hour = int(h)
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm is None and 1 <= hour <= 9:  # "from 4 to 9" on a Saturday almost always means pm
        hour += 12
    return hour * 60 + int(mm or 0)


def parse_time_range(text: str) -> Optional[tuple[str, int]]:
    m = re.search(r"(?:from\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-|till|until)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
                  text.lower())
    if not m:
        return None
    a = _hour(m.group(1), m.group(2), m.group(3) or m.group(6))
    b = _hour(m.group(4), m.group(5), m.group(6))
    if b <= a:
        return None
    return f"{a // 60:02d}:{a % 60:02d}", b - a


def heuristic_parse(raw) -> Preferences:
    """Rough regex parser so the app still works when OpenAI is down."""
    if isinstance(raw, dict):
        text = " ".join(str(v) if not isinstance(v, list) else " ".join(v) for v in raw.values())
    else:
        text = str(raw)
    t = text.lower()
    p = Preferences(assumptions=["Parsed with simple rules because the language model was unavailable."])

    if isinstance(raw, dict) and raw.get("city"):
        p.city = str(raw["city"]).strip()
    else:
        for key, city in load_mock()["cities"].items():
            if any(re.search(rf"\b{n}\b", t) for n in [key, *city.get("aliases", [])]):
                p.city = key.title()
        if not p.city:
            m = re.search(r"\b(?:in|at)\s+([A-Z][a-zA-Z]+)", text)
            if m:
                p.city = m.group(1)

    if isinstance(raw, dict) and raw.get("budget") not in (None, ""):
        p.budget_inr = int(float(raw["budget"]))
    else:
        m = re.search(r"(?:₹|rs\.?|inr)\s*(\d[\d,]*)|(\d[\d,]*)\s*(?:rs|rupees|inr|₹)", t)
        if m:
            p.budget_inr = int((m.group(1) or m.group(2)).replace(",", ""))

    rng = parse_time_range(t)
    if rng:
        p.start_time, p.duration_min = rng
    else:
        p.duration_min = parse_duration(str(raw.get("available_time", "")) if isinstance(raw, dict) else t)
    if isinstance(raw, dict) and raw.get("start_time"):
        p.start_time = raw["start_time"]

    p.mood = raw.get("mood") if isinstance(raw, dict) else None
    for word, energy in rules.MOOD_TO_ENERGY.items():
        if word in t:
            p.energy = energy  # type: ignore[assignment]
            p.mood = p.mood or word
            break

    if isinstance(raw, dict) and raw.get("interests"):
        p.interests = normalise_interests(list(raw["interests"]))
    else:
        p.interests = normalise_interests(re.findall(r"[a-z]+(?: [a-z]+)?", t))
    if "vegan" in t:
        p.diet = "vegan"
    elif re.search(r"\bveg(etarian)?\b", t) and "non-veg" not in t and "non veg" not in t:
        p.diet = "vegetarian"
    p.avoid_crowds = bool(re.search(r"crowd|quiet|calm|peaceful", t))
    return p
