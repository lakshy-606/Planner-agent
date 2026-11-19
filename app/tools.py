"""Agent tools: plain functions, no LLM. Each returns data plus enough info for a trace line."""
from __future__ import annotations

import math
from typing import Optional

from . import data_sources as ds
from . import rules
from .models import Issue, Place, Preferences, Stop

DAY_END = 23 * 60 + 59
MAX_LEG_MIN = 60
MIN_STOP_MIN = 30
LOW_BUDGET_INR = 500


# ---------------------------------------------------------------- time helpers

def to_min(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def to_hhmm(m: int) -> str:
    m = max(0, min(int(round(m)), DAY_END))
    return f"{m // 60:02d}:{m % 60:02d}"


def window(prefs: Preferences) -> tuple[int, int]:
    start = to_min(prefs.start_time)
    return start, min(start + (prefs.duration_min or 240), DAY_END)


def max_stops_for(duration_min: int) -> int:
    if duration_min < 90:
        return 1
    if duration_min < 180:
        return 2
    if duration_min < 300:
        return 3
    return 4


def min_stops_for(duration_min: int) -> int:
    return 1 if duration_min < 90 else 2


def min_stop_for(prefs: Preferences) -> int:
    """Shortest acceptable stop: 30 min, or the whole window if it is shorter than that."""
    ws, we = window(prefs)
    return max(10, min(MIN_STOP_MIN, we - ws))


def crowd_at(place: Place, start_min: int) -> str:
    slot = rules.slot_of(to_hhmm(start_min))
    if slot in place.crowd:
        return place.crowd[slot]
    return next(iter(place.crowd.values()), "moderate")


def open_span(place: Place, at_min: int) -> Optional[tuple[int, int]]:
    """The opening interval containing or next after `at_min` (start may be later than at_min)."""
    spans = sorted((to_min(a), to_min(b)) for a, b in place.hours) or [(0, DAY_END)]
    for a, b in spans:
        if b > at_min:
            return max(a, at_min), b
    return None


# ---------------------------------------------------------------- city + places

def resolve_city(city: str) -> dict:
    """Live geocode → mock alias → generic. Returns coords, source, city_key and a user-facing note."""
    mock_key = ds.find_mock_city(city)
    try:
        geo = ds.geoapify_geocode(city)
        if geo:
            return {"coords": (geo["lat"], geo["lon"]), "source": "live", "city_key": mock_key,
                    "name": geo["name"], "note": None, "detail": "geocoded via Geoapify"}
        live_error = "city not found by geocoder"
    except ds.DataSourceError as e:
        live_error = str(e)

    if mock_key:
        return {"coords": ds.mock_center(mock_key), "source": "mock", "city_key": mock_key,
                "name": mock_key.title(), "note": None, "detail": f"live lookup failed ({live_error}); using curated data"}
    return {"coords": None, "source": "generic", "city_key": None, "name": city,
            "note": f"We couldn't find \"{city}\" (we currently cover Indian cities), so this is a "
                    "template plan with place types rather than specific venues.",
            "detail": f"{live_error}; no curated data"}


def _live_places(coords, categories, kind, prefs) -> list[Place]:
    lat, lon = coords
    conditions = "vegetarian" if (kind == "food" and prefs.diet in ("vegetarian", "vegan")) else None
    places: list[Place] = []
    for radius in (8000, 15000):
        # many OSM parks/cafes are unnamed, so fetch generously (100 results ≈ 5 credits)
        feats = ds.geoapify_places(lat, lon, categories, radius, limit=100, conditions=conditions)
        feats = [f for f in feats if "access_limited.private" not in f.get("properties", {}).get("categories", [])]
        places = [p for p in (ds.normalise_feature(f, kind) for f in feats) if p]
        places = [p for p in places if (p.type == "food") == (kind == "food")]
        if len(places) >= 5:
            break
    if conditions:  # we asked for veg-friendly places, so at least veg options are confirmed
        for p in places:
            if p.veg == "limited_veg":
                p.veg = "veg_options"
    # dedupe by name
    seen, out = set(), []
    for p in places:
        if p.name.lower() not in seen:
            seen.add(p.name.lower())
            out.append(p)
    return out


def get_options(kind: str, city: dict, prefs: Preferences) -> tuple[list[Place], str, str]:
    """kind = 'activity' | 'food'. Returns (places, source_used, detail)."""
    if kind == "activity":
        cats = sorted({c for i in prefs.interests for c in rules.INTEREST_CATEGORIES.get(i, [])}) \
            or rules.DEFAULT_ACTIVITY_CATEGORIES
        if prefs.budget_inr is not None and prefs.budget_inr < LOW_BUDGET_INR:
            cats = sorted(set(cats) | {"leisure.park"})  # free options for tight budgets
    else:
        cats = rules.FOOD_CATEGORIES

    if city["source"] == "live":
        try:
            places = _live_places(city["coords"], cats, kind, prefs)
            if len(places) >= 3:
                return places, "live", f"{len(places)} places from Geoapify"
            detail = f"only {len(places)} live results"
        except ds.DataSourceError as e:
            places, detail = [], f"Geoapify error: {e}"
        if city["city_key"]:
            extra = ds.mock_places(city["city_key"], kind)
            return places + extra, "mock", f"{detail}; added {len(extra)} curated places"
        extra = ds.generic_places(kind, city["coords"])
        return places + extra, "generic", f"{detail}; added {len(extra)} generic templates"

    if city["source"] == "mock":
        places = ds.mock_places(city["city_key"], kind)
        return places, "mock", f"{len(places)} curated places"
    places = ds.generic_places(kind, city["coords"])
    return places, "generic", f"{len(places)} generic templates"


# ---------------------------------------------------------------- scoring + filtering

def fit_score(p: Place, prefs: Preferences, at_min: Optional[int] = None) -> float:
    s = 0.0
    matches = len(set(p.interests) & set(prefs.interests))
    s += 2.0 * min(matches, 2)
    if p.type == "food" and "food" in prefs.interests:
        s += 1
    if prefs.energy == "low":
        s += {"low": 1, "medium": -0.5, "high": -5}[p.energy]
    elif prefs.energy == "high":
        s += {"low": -0.5, "medium": 0.5, "high": 1.5}[p.energy]
    crowd = crowd_at(p, at_min) if at_min is not None else max(p.crowd.values(), default="moderate",
                                                               key=["quiet", "moderate", "busy"].index)
    if prefs.avoid_crowds:
        s += {"quiet": 1.5, "moderate": 0, "busy": -3}.get(crowd, 0)
    if prefs.budget_inr:
        if p.cost_per_person_inr > 0.6 * prefs.budget_inr:
            s -= 2
        elif p.cost_per_person_inr == 0:
            s += 0.3
    if p.type == "food" and p.veg == "pure_veg" and prefs.diet != "any":
        s += 1
    if "sample_event" in p.tags:
        s -= 0.5
    return s


def filter_candidates(prefs: Preferences, places: list[Place]) -> tuple[list[Place], dict]:
    """Hard filters (diet, energy, budget, open in window) then rank by fit. Returns kept + drop stats."""
    ws, we = window(prefs)
    stats = {"diet": 0, "energy": 0, "budget": 0, "closed": 0}
    kept = []
    for p in places:
        if p.type in ("food", "event") and prefs.diet != "any" and p.veg not in ("pure_veg", "veg_options"):
            if p.type == "food":
                stats["diet"] += 1
                continue
        if prefs.energy == "low" and p.energy == "high":
            stats["energy"] += 1
            continue
        if prefs.budget_inr is not None and p.cost_per_person_inr > prefs.budget_inr:
            stats["budget"] += 1
            continue
        need = min(p.duration_min, 45, max(we - ws, 10))
        spans = [(max(to_min(a), ws), min(to_min(b), we)) for a, b in p.hours] or [(ws, we)]
        if not any(b - a >= need for a, b in spans):
            stats["closed"] += 1
            continue
        kept.append(p)
    kept.sort(key=lambda p: fit_score(p, prefs), reverse=True)
    acts = [p for p in kept if p.type != "food"][:15]
    food = [p for p in kept if p.type == "food"][:10]
    return acts + food, stats


# ---------------------------------------------------------------- travel, schedule, cost

def haversine_km(a: Place, b: Place) -> Optional[float]:
    if None in (a.lat, a.lon, b.lat, b.lon):
        return None
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dphi, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def estimate_leg(a: Place, b: Place) -> tuple[float, float]:
    """Offline estimate: road distance ≈ 1.35× straight line, ~20 km/h city traffic, +5 min pickup."""
    d = haversine_km(a, b)
    if d is None:
        return 15.0, 5.0  # generic templates: assume "nearby"
    road = d * 1.35
    return 5 + road / 20 * 60, road


def estimate_travel(places: list[Place]) -> tuple[list[tuple[float, float]], str]:
    """Per-leg (minutes, km). Geoapify route matrix when possible, else distance estimate."""
    if len(places) < 2:
        return [], "none"
    pts = [(p.lat, p.lon) for p in places]
    if all(p.source != "generic" for p in places) and None not in [c for pt in pts for c in pt]:
        try:
            legs = ds.geoapify_route_legs(pts)  # type: ignore[arg-type]
            # Routing times are optimistic for Indian city traffic: floor at 20 km/h, +5 min pickup/parking
            return [(max(m, km / 20 * 60) + 5, km) for m, km in legs], "live"
        except ds.DataSourceError:
            pass
    return [estimate_leg(a, b) for a, b in zip(places, places[1:])], "estimate"


def build_schedule(places: list[Place], durations: list[Optional[int]],
                   legs: list[tuple[float, float]], prefs: Preferences) -> list[Stop]:
    """Lay stops on the clock. Waits for opening time; trims a stop to fit closing time / end of window."""
    ws, we = window(prefs)
    min_stop = min_stop_for(prefs)
    t = ws
    stops = []
    for i, p in enumerate(places):
        travel_min, travel_km = legs[i - 1] if i > 0 else (0.0, 0.0)
        arrive = t + travel_min
        span = open_span(p, int(arrive))
        start = span[0] if span else arrive
        dur = durations[i] or p.duration_min
        dur = max(min_stop, min(dur, 180))
        end = start + dur
        if span and end > span[1] and span[1] - start >= min_stop:
            end = span[1]
        if end > we and we - start >= min_stop:
            end = we
        stops.append(Stop(place_id=p.id, start=to_hhmm(start), end=to_hhmm(end),
                          duration_min=int(end - start), travel_min_from_prev=int(round(travel_min)),
                          travel_km_from_prev=round(travel_km, 1)))
        t = end
    return stops


def estimate_cost(stops: list[Stop], by_id: dict[str, Place]) -> dict:
    per_stop = {s.place_id: by_id[s.place_id].cost_per_person_inr for s in stops if s.place_id in by_id}
    km = sum(s.travel_km_from_prev for s in stops)
    return {"per_stop": per_stop, "travel_km": round(km, 1), "total": sum(per_stop.values())}


# ---------------------------------------------------------------- validation

def validate_plan(stops: list[Stop], by_id: dict[str, Place], prefs: Preferences, cost: dict) -> list[Issue]:
    issues: list[Issue] = []
    err = lambda r, m: issues.append(Issue(level="error", rule=r, message=m))
    warn = lambda r, m: issues.append(Issue(level="warning", rule=r, message=m))
    ws, we = window(prefs)
    avail = we - ws

    # 10. stop count + duplicates
    lo, hi = min_stops_for(avail), (1 if avail < 90 else 4)
    if not (lo <= len(stops) <= hi):
        err("stop_count", f"Plan has {len(stops)} stops; need {lo} to {hi} for a {avail}-minute window.")
    ids = [s.place_id for s in stops]
    if len(ids) != len(set(ids)):
        err("duplicates", "The same place appears twice.")

    # 1. ids exist
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        err("unknown_id", f"Unknown place ids: {unknown}. Use only candidate ids.")
        return issues

    # 2. budget
    if prefs.budget_inr is not None:
        total = cost["total"]
        if total > prefs.budget_inr * 1.10:
            err("budget", f"Total ₹{total} exceeds budget ₹{prefs.budget_inr} by more than 10%. Pick cheaper stops.")
        elif total > prefs.budget_inr:
            warn("budget", f"Total ₹{total} is slightly over the ₹{prefs.budget_inr} budget.")

    # 3. ends within window
    if stops and to_min(stops[-1].end) > we:
        err("time_window", f"Plan ends at {stops[-1].end}, after the {to_hhmm(we)} end of the window. Drop or shorten a stop.")

    total_travel = 0
    for i, s in enumerate(stops):
        p = by_id[s.place_id]
        a, b = to_min(s.start), to_min(s.end)
        # 4. open for whole slot
        if p.hours and not any(to_min(o) <= a and b <= to_min(c) for o, c in p.hours):
            hrs = ", ".join(f"{o}-{c}" for o, c in p.hours)
            covering = [(o, c) for o, c in p.hours if to_min(o) <= a < to_min(c)]
            why = (f"closes at {covering[0][1]} but the stop would run until {s.end}" if covering
                   else f"is closed when you'd arrive at {s.start}")
            err("hours", f"{p.name} {why} (Saturday hours {hrs}). Visit it earlier in the order or replace it.")
        if b - a < min_stop_for(prefs):
            err("too_short", f"{p.name} only gets {b - a} minutes.")
        # 5. travel
        if s.travel_min_from_prev > MAX_LEG_MIN:
            err("travel_leg", f"Getting to {p.name} takes {s.travel_min_from_prev} min (> {MAX_LEG_MIN}). Choose stops closer together.")
        total_travel += s.travel_min_from_prev
        # 6. diet
        if p.type == "food" and prefs.diet != "any" and p.veg not in ("pure_veg", "veg_options"):
            err("diet", f"{p.name} is not reliably {prefs.diet}.")
        # 7. crowds
        if prefs.avoid_crowds and crowd_at(p, a) == "busy":
            warn("crowds", f"{p.name} is usually busy around {s.start} on Saturdays.")
        # 8. energy
        if prefs.energy == "low" and p.energy == "high":
            err("energy", f"{p.name} needs high energy, but the user is low on energy.")
        # gaps (soft)
        if i > 0:
            gap = a - to_min(stops[i - 1].end) - s.travel_min_from_prev
            if gap > 30:
                warn("idle_gap", f"{gap} minutes of waiting before {p.name} opens.")

    if total_travel > 0.30 * avail:
        err("travel_total", f"Total travel {total_travel} min is over 30% of the {avail}-minute window.")

    # 9. interest coverage
    covered = {i for s in stops for i in by_id[s.place_id].interests}
    available = {i for p in by_id.values() for i in p.interests}
    missing = [i for i in prefs.interests[:3] if i in available and i not in covered]
    if missing:
        warn("interests", f"No stop covers: {', '.join(missing)}.")
    return issues


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.level == "error"]


# ---------------------------------------------------------------- full evaluation + fallback

def evaluate(places: list[Place], durations: list[Optional[int]], prefs: Preferences,
             by_id: dict[str, Place]) -> tuple[list[Stop], dict, list[Issue], str]:
    legs, travel_src = estimate_travel(places)
    stops = build_schedule(places, durations, legs, prefs)
    cost = estimate_cost(stops, by_id)
    issues = validate_plan(stops, by_id, prefs, cost)
    return stops, cost, issues, travel_src


def greedy_plan(prefs: Preferences, cands: list[Place], respect_budget: bool = True) -> list[Place]:
    ws, we = window(prefs)
    budget = prefs.budget_inr if (respect_budget and prefs.budget_inr is not None) else 10 ** 9
    max_stops = max_stops_for(we - ws)
    chosen: list[Place] = []
    t, spent, covered = float(ws), 0.0, set()
    min_stop = min_stop_for(prefs)
    while len(chosen) < max_stops and we - t >= max(min_stop, min(40, we - ws)):
        best, best_score, best_end = None, -1e9, 0.0
        for p in cands:
            if p in chosen:
                continue
            leg_min, _ = estimate_leg(chosen[-1], p) if chosen else (0.0, 0.0)
            if leg_min > MAX_LEG_MIN:
                continue
            span = open_span(p, int(t + leg_min))
            if not span:
                continue
            start = span[0]
            end = min(start + p.duration_min, span[1], we)
            if end - start < min_stop:
                continue
            cost = p.cost_per_person_inr
            if spent + cost > budget:
                continue
            score = fit_score(p, prefs, int(start))
            score += 1.5 * len((set(p.interests) & set(prefs.interests)) - covered)
            score -= leg_min / 15 + (start - t - leg_min) / 30
            if chosen and chosen[-1].type == "food" and p.type == "food":
                score -= 3
            if not respect_budget:
                score -= cost / 300
            if score > best_score:
                best, best_score, best_end = p, score, end
                best_cost = cost
        if best is None:
            break
        chosen.append(best)
        covered |= set(best.interests)
        spent += best_cost
        t = best_end
    return chosen


def fallback_plan(prefs: Preferences, cands: list[Place]) -> tuple[list[Place], bool]:
    """Deterministic plan. Returns (places, within_budget)."""
    plan = greedy_plan(prefs, cands)
    if plan:
        return plan, True
    return greedy_plan(prefs, cands, respect_budget=False), False
