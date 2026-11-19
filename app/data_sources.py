"""External data: config, Geoapify REST client (with TTL cache), mock data loader, normalisation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from . import rules
from .models import Place

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_data.json"
GEOAPIFY_BASE = "https://api.geoapify.com"
HTTP_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
CACHE_TTL_S = 24 * 3600
# Costs, crowd rules and mock data are India-specific, so geocoding is restricted to India.
SUPPORTED_COUNTRY = "in"


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Env var first (covers .env and Streamlit Cloud root secrets), then st.secrets."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st  # imported lazily so tools stay testable without Streamlit
        return st.secrets.get(name, default)  # type: ignore[no-any-return]
    except Exception:
        return default


class DataSourceError(Exception):
    """Raised when a live API call fails; callers switch to mock data."""


# ---------------------------------------------------------------- cache

_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


# ---------------------------------------------------------------- Geoapify

def _api_key() -> str:
    key = get_secret("GEOAPIFY_API_KEY")
    if not key:
        raise DataSourceError("GEOAPIFY_API_KEY not set")
    return key


def _request(method: str, path: str, params: dict, body: Optional[dict] = None) -> dict:
    params = {**params, "apiKey": _api_key()}
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.request(method, f"{GEOAPIFY_BASE}{path}", params=params, json=body)
    except httpx.HTTPError as e:
        raise DataSourceError(f"network error: {type(e).__name__}") from e
    if resp.status_code >= 400:
        raise DataSourceError(f"HTTP {resp.status_code} from {path}")
    return resp.json()


def geoapify_geocode(city: str) -> Optional[dict]:
    """Returns {'lat', 'lon', 'name'} or None if the city isn't found."""
    def call():
        data = _request("GET", "/v1/geocode/search", {
            "text": city, "type": "city", "limit": 1, "format": "json",
            "filter": f"countrycode:{SUPPORTED_COUNTRY}",
        })
        results = data.get("results") or []
        if not results:
            return None
        r = results[0]
        if r.get("result_type") not in ("city", "county", "district", "suburb"):
            return None  # e.g. "Atlantis" matches a building called Atlantis in Mumbai
        conf = (r.get("rank") or {}).get("confidence", 1)
        if conf is not None and conf < 0.5:
            return None
        return {"lat": r["lat"], "lon": r["lon"], "name": r.get("city") or r.get("name") or city}
    return _cached(f"geo:{city.lower().strip()}", call)


def geoapify_places(lat: float, lon: float, categories: list[str], radius_m: int,
                    limit: int = 30, conditions: Optional[str] = None) -> list[dict]:
    def call():
        params = {
            "categories": ",".join(sorted(set(categories))),
            "filter": f"circle:{lon},{lat},{radius_m}",
            "bias": f"proximity:{lon},{lat}",
            "limit": limit,
        }
        if conditions:
            params["conditions"] = conditions
        return _request("GET", "/v2/places", params).get("features", [])
    key = f"places:{lat:.3f},{lon:.3f}:{sorted(categories)}:{radius_m}:{limit}:{conditions}"
    return _cached(key, call)


def geoapify_route_legs(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drive time (min) and distance (km) for each consecutive leg of `points` (lat, lon)."""
    if len(points) < 2:
        return []
    def call():
        body = {
            "mode": "drive",
            "sources": [{"location": [lon, lat]} for lat, lon in points[:-1]],
            "targets": [{"location": [lon, lat]} for lat, lon in points[1:]],
            "traffic": "approximated",
        }
        data = _request("POST", "/v1/routematrix", {}, body)
        matrix = data["sources_to_targets"]
        legs = []
        for i in range(len(points) - 1):
            cell = matrix[i][i]
            if cell.get("time") is None:
                raise DataSourceError("route not found")
            legs.append((cell["time"] / 60.0, cell["distance"] / 1000.0))
        return legs
    return _cached(f"route:{points}", call)


# ---------------------------------------------------------------- opening hours

_DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]


def _days_cover_saturday(spec: str) -> bool:
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = [x.strip()[:2] for x in part.split("-", 1)]
            if a in _DAYS and b in _DAYS:
                ia, ib = _DAYS.index(a), _DAYS.index(b)
                rng = range(ia, ib + 1) if ia <= ib else list(range(ia, 7)) + list(range(0, ib + 1))
                if 5 in rng:
                    return True
        elif part[:2] == "Sa":
            return True
    return False


def parse_saturday_hours(oh: Optional[str]) -> Optional[list[tuple[str, str]]]:
    """Best-effort OSM opening_hours parser for Saturday. Returns None if unparseable, [] if closed."""
    if not oh or not isinstance(oh, str):
        return None
    oh = oh.strip()
    if oh == "24/7":
        return [("00:00", "23:59")]
    result: Optional[list[tuple[str, str]]] = None
    for rule in oh.split(";"):
        rule = rule.strip()
        if not rule:
            continue
        m = re.match(r"^((?:Mo|Tu|We|Th|Fr|Sa|Su)(?:[\s,\-]*(?:Mo|Tu|We|Th|Fr|Sa|Su))*)\s*(.*)$", rule)
        if m:
            days, times = m.group(1), m.group(2)
            if not _days_cover_saturday(days):
                continue
        elif re.match(r"^\d", rule):
            times = rule  # no day spec = every day
        else:
            continue  # PH, months, etc.
        if re.search(r"\b(off|closed)\b", times):
            result = []
            continue
        spans = re.findall(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", times)
        if spans:
            out = []
            for a, b in spans:
                a, b = a.zfill(5), b.zfill(5)
                if b <= a or b > "23:59":  # past midnight
                    b = "23:59"
                out.append((a, b))
            result = out
    return result


# ---------------------------------------------------------------- normalise live data

def _classify(categories: list[str]) -> tuple[str, list[str]]:
    for prefix, sub, interests in rules.GEOAPIFY_SUBCATEGORY:
        if any(c == prefix or c.startswith(prefix + ".") for c in categories):
            return sub, interests
    return "other", []


def _veg_status(categories: list[str], kind: str) -> str:
    if kind != "food":
        return "n/a"
    if any(c in ("vegetarian.only", "vegan.only") for c in categories):
        return "pure_veg"
    if any(c.startswith("vegetarian") or c.startswith("vegan") for c in categories):
        return "veg_options"
    return "limited_veg"


def normalise_feature(feature: dict, kind: str) -> Optional[Place]:
    p = feature.get("properties", {})
    name = p.get("name")
    if not name or p.get("lat") is None:
        return None
    cats = p.get("categories", [])
    sub, interests = _classify(cats)
    if sub == "other":
        return None
    is_food = sub in ("cafe", "restaurant", "fast_food", "ice_cream")
    place_type = "food" if is_food else "activity"
    estimated = ["cost", "crowd"]

    raw_oh = p.get("opening_hours") or (p.get("datasource", {}).get("raw", {}) or {}).get("opening_hours")
    hours = parse_saturday_hours(raw_oh)
    if hours is None:
        hours = rules.DEFAULT_HOURS.get(sub, rules.FALLBACK_HOURS)
        estimated.append("hours")
    if hours == []:  # closed on Saturday
        return None

    cost = rules.COST_BY_SUBCATEGORY.get(sub, rules.DEFAULT_COST[place_type])
    if "no_fee" in cats and not is_food:
        cost = 0
    if "fee" in cats and cost == 0:
        cost = 50

    area = p.get("suburb") or p.get("district") or p.get("neighbourhood") or p.get("city")
    return Place(
        # short ids: the LLM copies these back, long hashes get mangled
        id="g" + hashlib.sha1(str(p.get("place_id", name)).encode()).hexdigest()[:7],
        name=name,
        type=place_type,
        subcategory=sub,
        area=area,
        lat=p["lat"], lon=p["lon"],
        interests=list(dict.fromkeys(interests + (["food"] if is_food else []))),
        cost_per_person_inr=cost,
        energy=rules.ENERGY_BY_SUBCATEGORY.get(sub, "low"),
        crowd=rules.CROWD_RULES.get(sub, rules.DEFAULT_CROWD),
        veg=_veg_status(cats, place_type),
        hours=hours,
        duration_min=rules.TYPICAL_DURATION.get(sub, 60),
        indoor=sub not in ("park", "nature", "beach", "sights"),
        source="live",
        estimated_fields=estimated,
    )


# ---------------------------------------------------------------- mock data

@lru_cache(maxsize=1)
def load_mock() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_mock_city(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.lower().strip()
    for key, city in load_mock()["cities"].items():
        if n == key or n in city.get("aliases", []):
            return key
    return None


def mock_center(city_key: str) -> tuple[float, float]:
    c = load_mock()["cities"][city_key]["center"]
    return c["lat"], c["lon"]


def mock_places(city_key: str, kind: str) -> list[Place]:
    """kind: 'activity' (activities + sample events) or 'food'."""
    city = load_mock()["cities"][city_key]
    rows = [p for p in city["places"] if p["type"] == kind]
    if kind == "activity":
        rows += city.get("sample_events", [])
    out = []
    for r in rows:
        est = ["cost", "crowd", "hours"]
        out.append(Place(**r, source="mock", estimated_fields=est))
    return out


def generic_places(kind: str, center: Optional[tuple[float, float]] = None) -> list[Place]:
    rows = [p for p in load_mock()["fallback_generic"]["places"] if p["type"] == kind]
    lat, lon = center if center else (None, None)
    return [Place(**r, lat=lat, lon=lon, source="generic",
                  estimated_fields=["cost", "crowd", "hours", "place"]) for r in rows]
