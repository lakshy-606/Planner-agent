"""LangGraph nodes. Each node reads state, calls tools/LLM, returns a partial update + a trace line."""
from __future__ import annotations

import json
import re

from . import rules, tools
from .data_sources import generic_places
from .llm import LLMUnavailable, heuristic_parse, normalise_interests, parse_duration, structured_call
from .models import DraftPlan, FinalNarrative, Issue, Place, PlannerState, Preferences, Stop
from .trace import traced

MAX_RETRIES = 2
DEFAULT_BUDGET = 1500
DEFAULT_DURATION = 240
DEFAULT_INTERESTS = ["food", "walks"]


def _by_id(state: PlannerState) -> dict[str, Place]:
    return {p.id: p for p in state.get("candidates", [])}


# ================================================================ parse_preferences

PARSE_SYSTEM = f"""You extract Saturday-outing preferences for a city in India into the given schema.
Rules:
- Never invent a city or budget. Leave them null if absent.
- budget_inr: per-person budget in rupees, integer.
- duration_min: total available minutes ("4 hours" -> 240). "Free from 4 to 9" means start_time "16:00" and duration_min 300. Null if not stated.
- start_time: 24h "HH:MM". If no start time is given use "15:00". Do NOT write assumptions about the start time.
- energy: derive from mood. Hints: {json.dumps(rules.MOOD_TO_ENERGY)}. "tired but wants fun" -> low. Default medium.
- interests: only values from {rules.KNOWN_INTERESTS}. Map synonyms (museum -> culture, park -> nature, coffee -> food, gig -> music).
- diet: "vegetarian" or "vegan" only if stated, else "any".
- avoid_crowds: true if they want to avoid crowds or want somewhere quiet.
- assumptions: short notes for anything else you had to assume."""


TIME_HINT = re.compile(
    r"\d\s*(am|pm)\b"                              # 5pm, 5 pm
    r"|\d[:.]\d\d"                                  # 17:00, 5.30
    r"|\b([01]?\d|2[0-3])[0-5]\d\s*(hrs?|hours|h)\b"  # 1700 hrs, 0930h
    r"|\bfrom\s+\d|\b\d{1,2}\s*(to|-|till|until)\s*\d"  # from 4 to 9
    r"|\b(at|start\w*(\s+time)?(\s+(at|is|from))?)\s*:?\s*\d"  # at 5, start time 17
    r"|\b(noon|midday)\b",
    re.I)


def _start_given(raw) -> bool:
    """Did the user state a start time? Used only to decide whether to show the 15:00 assumption."""
    if isinstance(raw, dict):
        return bool(raw.get("start_time"))
    return bool(TIME_HINT.search(str(raw)))


@traced("parse_preferences")
def parse_preferences(state: PlannerState) -> dict:
    raw = state["raw_input"]
    status, how = "ok", "LLM"
    try:
        user = json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)
        prefs = structured_call(Preferences, PARSE_SYSTEM, user, temperature=0.2)
    except Exception as e:
        prefs, status, how = heuristic_parse(raw), "fallback", f"rules (LLM unavailable: {str(e)[:80]})"

    # Normalise + trust explicit form fields over the LLM
    prefs.interests = normalise_interests(prefs.interests)
    if isinstance(raw, dict):
        if raw.get("city"):
            prefs.city = str(raw["city"]).strip()
        if raw.get("budget") not in (None, ""):
            prefs.budget_inr = int(float(raw["budget"]))
        if raw.get("start_time"):
            prefs.start_time = str(raw["start_time"])
        d = parse_duration(str(raw.get("available_time", "")))
        if d:
            prefs.duration_min = d
        cons = " ".join(raw.get("constraints") or []).lower()
        if "vegan" in cons:
            prefs.diet = "vegan"
        elif "veg" in cons and "non" not in cons:
            prefs.diet = "vegetarian"
        if "crowd" in cons or "quiet" in cons:
            prefs.avoid_crowds = True
    if not re.fullmatch(r"\d{2}:\d{2}", prefs.start_time or ""):
        prefs.start_time = "15:00"
    # drop LLM notes we generate ourselves (start time, missing fields) to avoid duplicates
    prefs.assumptions = [a for a in prefs.assumptions
                         if not re.search(r"start|15:00|interest|budget|not (specified|mentioned|given|provided)", a, re.I)]
    # Trust the LLM's start time; only flag the 15:00 default when the user really gave none
    if prefs.start_time == "15:00" and not _start_given(raw):
        prefs.assumptions.append("No start time given, so the plan starts at 15:00.")
    if prefs.duration_min:
        start = tools.to_min(prefs.start_time)
        if start + prefs.duration_min > tools.DAY_END:
            prefs.duration_min = tools.DAY_END - start
            prefs.assumptions.append("Your window runs past midnight; the plan ends by 23:59.")
    prefs.assumptions = list(dict.fromkeys(prefs.assumptions))

    summary = (f"via {how}: city={prefs.city}, budget=₹{prefs.budget_inr}, {prefs.duration_min} min from "
               f"{prefs.start_time}, energy={prefs.energy}, interests={prefs.interests}, diet={prefs.diet}, "
               f"avoid_crowds={prefs.avoid_crowds}")
    return {"prefs": prefs, "retries": 0, "notes": [], "_trace": {"status": status, "summary": summary}}


# ================================================================ check_clarity

@traced("check_clarity")
def check_clarity(state: PlannerState) -> dict:
    prefs = state["prefs"].model_copy(deep=True)
    allow_questions = not state.get("is_followup", False)
    questions: list[str] = []
    missing = [k for k, v in (("budget", prefs.budget_inr is not None), ("time", bool(prefs.duration_min)),
                              ("interests", bool(prefs.interests))) if not v]

    if allow_questions and not prefs.city:
        questions.append("Which city will you be in on Saturday?")
    if allow_questions and len(missing) >= 2:
        if "budget" in missing or "time" in missing:
            questions.append("Roughly how much would you like to spend, and when are you free "
                             "(e.g. ₹1500, from 4 to 9pm)?")
        if "interests" in missing and len(questions) < 2:
            questions.append("What are you in the mood for: food, music, walks, art, books, shopping?")
    questions = questions[:2]
    if questions:
        return {"clarifying_questions": questions,
                "_trace": {"status": "needs_input", "summary": f"vague input, missing: {', '.join(missing) or 'city'}"}}

    # Fill single gaps with stated defaults
    if prefs.budget_inr is None:
        prefs.budget_inr = DEFAULT_BUDGET
        prefs.assumptions.append(f"No budget given, so we assumed about ₹{DEFAULT_BUDGET} per person.")
    if not prefs.duration_min:
        prefs.duration_min = DEFAULT_DURATION
        prefs.assumptions.append("No time window given, so we planned for 4 hours.")
    if not prefs.interests:
        prefs.interests = DEFAULT_INTERESTS
        prefs.assumptions.append("No interests given, so we went with food and a relaxed walk.")
    if not prefs.city:
        prefs.city = "your city"
    return {"prefs": prefs, "clarifying_questions": [],
            "_trace": {"summary": "enough detail to plan" + (f" (defaults for: {', '.join(missing)})" if missing else "")}}


@traced("ask_clarification")
def ask_clarification(state: PlannerState) -> dict:
    qs = state["clarifying_questions"]
    return {"final": {"type": "clarify", "questions": qs},
            "_trace": {"status": "needs_input", "summary": f"asked {len(qs)} question(s)"}}


# ================================================================ data gathering

@traced("resolve_city")
def resolve_city(state: PlannerState) -> dict:
    prefs = state["prefs"]
    city = tools.resolve_city(prefs.city or "your city")
    notes = [city["note"]] if city["note"] else []
    coords = f" at {city['coords'][0]:.3f},{city['coords'][1]:.3f}" if city["coords"] else ""
    return {"city": city, "notes": notes,
            "_trace": {"source": city["source"], "status": "ok" if city["source"] != "generic" else "fallback",
                       "summary": f"{city['name']}{coords}: {city['detail']}"}}


def _options_node(kind: str):
    @traced(f"get_{kind}_options")
    def node(state: PlannerState) -> dict:
        places, source, detail = tools.get_options(kind, state["city"], state["prefs"])
        key = "activity_candidates" if kind == "activity" else "food_candidates"
        return {key: places, "_trace": {"source": source, "summary": detail,
                                        "status": "ok" if source == state["city"]["source"] else "fallback"}}
    return node


get_activity_options = _options_node("activity")
get_food_options = _options_node("food")


@traced("enrich_and_filter")
def enrich_and_filter(state: PlannerState) -> dict:
    prefs = state["prefs"]
    raw = state.get("activity_candidates", []) + state.get("food_candidates", [])
    kept, stats = tools.filter_candidates(prefs, raw)
    status, extra = "ok", ""
    if len(kept) < 3 and state["city"]["source"] != "generic":
        more, _ = tools.filter_candidates(prefs, generic_places("activity", state["city"]["coords"])
                                          + generic_places("food", state["city"]["coords"]))
        kept += more
        status, extra = "fallback", f"; too few matches, added {len(more)} generic templates"
    sources = {p.source for p in kept}
    data_source = "live" if "live" in sources else ("mock" if "mock" in sources else "generic")
    dropped = ", ".join(f"{v} {k}" for k, v in stats.items() if v)
    return {"candidates": kept, "data_source": data_source,
            "_trace": {"status": status, "source": data_source,
                       "summary": f"{len(raw)} found → {len(kept)} candidates"
                                  + (f" (dropped: {dropped})" if dropped else "") + extra}}


# ================================================================ planning loop

DRAFT_SYSTEM = """You plan a Saturday outing by choosing places from a candidate list. You never invent places.
Return {lo} to {hi} stops in visiting order.
Hard rules:
- Use only candidate ids, each once.
- Everything, including travel (about 15-30 min per leg; stops in the same or nearby area are faster), must fit between {start} and {end}.
- Each stop must be open for the whole time spent there (check its hours against the time you'll arrive).
- Total spend (sum of stop costs) must stay within ₹{budget}.
- Respect diet, energy and crowd preferences. If the user avoids crowds, prefer stops whose crowd level is 'quiet' at that time of day (morning <12:00, afternoon 12-17, evening >=17:00).
- Cover the user's top interests where possible. Don't put two meals back-to-back.
- Prefer stops that are geographically close (compare lat/lon).
- duration_min is optional; use it to shorten a stop so the plan fits.
If validation issues from a previous attempt are listed, fix every one of them: reorder, shorten or swap stops.
Never return the previous attempt unchanged."""


@traced("draft_plan")
def draft_plan(state: PlannerState) -> dict:
    prefs, cands = state["prefs"], state["candidates"]
    ws, we = tools.window(prefs)
    lo, hi = tools.min_stops_for(we - ws), tools.max_stops_for(we - ws)
    system = DRAFT_SYSTEM.format(lo=lo, hi=max(hi, lo), start=tools.to_hhmm(ws), end=tools.to_hhmm(we),
                                 budget=prefs.budget_inr)
    cand_rows = [{**p.short(), "lat": round(p.lat, 3) if p.lat else None, "lon": round(p.lon, 3) if p.lon else None}
                 for p in cands]
    user = {"user": prefs.model_dump(exclude={"assumptions"}), "candidates": cand_rows}
    errs = tools.errors(state.get("issues", []))
    if errs and state.get("plan"):
        user["previous_attempt"] = [s.model_dump(include={"place_id", "start", "end", "travel_min_from_prev"})
                                    for s in state["plan"]]
        user["validation_issues_to_fix"] = [e.message for e in errs]
    attempt = state.get("retries", 0) + 1
    try:
        draft = structured_call(DraftPlan, system, json.dumps(user, ensure_ascii=False), temperature=0.2)
    except LLMUnavailable as e:
        return {"llm_failed": True, "plan": [],
                "_trace": {"status": "error", "summary": f"LLM unavailable ({str(e)[:80]}), switching to fallback"}}
    except Exception as e:
        return {"plan": [], "plan_rationale": "",
                "_trace": {"status": "error", "summary": f"attempt {attempt}: bad LLM output ({type(e).__name__})"}}
    plan = [Stop(place_id=s.place_id, duration_min=s.duration_min or 0) for s in draft.stops]
    by_id = _by_id(state)
    names = [by_id[s.place_id].name if s.place_id in by_id else f"??{s.place_id}" for s in plan]
    return {"plan": plan, "plan_rationale": draft.rationale, "llm_failed": False,
            "_trace": {"summary": f"attempt {attempt}: {' → '.join(names)}"}}


@traced("estimate_travel_and_cost")
def estimate_travel_and_cost(state: PlannerState) -> dict:
    by_id = _by_id(state)
    plan = state.get("plan", [])
    if not plan or any(s.place_id not in by_id for s in plan):
        return {"cost": {"total": 0, "travel_km": 0, "per_stop": {}},
                "_trace": {"status": "skipped", "summary": "plan empty or has unknown ids"}}
    places = [by_id[s.place_id] for s in plan]
    legs, src = tools.estimate_travel(places)
    stops = tools.build_schedule(places, [s.duration_min or None for s in plan], legs, state["prefs"])
    cost = tools.estimate_cost(stops, by_id)
    legs_txt = ", ".join(f"{int(m)} min/{km:.1f} km" for m, km in legs) or "single stop"
    return {"plan": stops, "cost": cost,
            "_trace": {"source": src, "summary": f"legs: {legs_txt}; total ₹{cost['total']}"}}


@traced("validate_plan")
def validate_plan(state: PlannerState) -> dict:
    issues = tools.validate_plan(state.get("plan", []), _by_id(state), state["prefs"], state.get("cost", {}))
    errs = tools.errors(issues)
    warns = [i for i in issues if i.level == "warning"]
    retries = state.get("retries", 0) + (1 if errs else 0)
    summary = ("valid" if not errs else f"{len(errs)} error(s): " + "; ".join(e.message for e in errs)[:300])
    if warns:
        summary += f" | {len(warns)} warning(s)"
    return {"issues": issues, "retries": retries,
            "_trace": {"status": "ok" if not errs else "retry", "summary": summary}}


@traced("fallback_plan")
def fallback_plan(state: PlannerState) -> dict:
    prefs = state["prefs"]
    cands = list(state.get("candidates", []))
    if not cands:
        cands = generic_places("activity", state["city"].get("coords")) + generic_places("food", state["city"].get("coords"))
    places, within_budget = tools.fallback_plan(prefs, cands)
    by_id = {p.id: p for p in cands}
    stops, cost, issues, src = tools.evaluate(places, [None] * len(places), prefs, by_id)
    notes = []
    if not within_budget and prefs.budget_inr is not None and cost["total"] > prefs.budget_inr:
        gap = cost["total"] - prefs.budget_inr
        free = [p.name for p in cands if p.cost_per_person_inr == 0][:3]
        notes.append(f"Your budget is tight: the cheapest workable plan is ₹{cost['total']}, about ₹{gap} over."
                     + (f" Free options nearby: {', '.join(free)}." if free else ""))
    why = "LLM unavailable" if state.get("llm_failed") else f"validation failed {state.get('retries', 0)} times"
    return {"plan": stops, "cost": cost, "issues": issues, "used_fallback": True,
            "candidates": cands, "notes": state.get("notes", []) + notes,
            "_trace": {"status": "fallback", "source": src,
                       "summary": f"{why}; greedy plan: {' → '.join(p.name for p in places) or 'none'}"
                                  f" (₹{cost['total']}, {len(tools.errors(issues))} unresolved)"}}


# ================================================================ finalize

FINAL_SYSTEM = """You explain a finished Saturday plan to the user. Be warm, concise and honest.
- For each stop write one sentence on why it fits THIS user (mood, energy, interests, diet, crowd level at that time, cost).
- Turn the listed warnings into honest trade-offs (e.g. "slightly over budget, but quieter than the alternatives").
- Mention that prices, crowd levels and some hours are estimates where it matters.
- Do not add, remove or rename places. Do not change any times or prices."""


def _template_reason(p: Place, prefs: Preferences, s: Stop) -> str:
    bits = []
    match = [i for i in p.interests if i in prefs.interests]
    if match:
        bits.append(f"covers your interest in {' and '.join(match[:2])}")
    crowd = tools.crowd_at(p, tools.to_min(s.start))
    if prefs.avoid_crowds and crowd == "quiet":
        bits.append(f"usually quiet around {s.start}")
    if prefs.energy == "low" and p.energy == "low":
        bits.append("easy on energy")
    if p.type == "food" and p.veg == "pure_veg":
        bits.append("fully vegetarian")
    if p.cost_per_person_inr == 0:
        bits.append("free")
    return (bits[0].capitalize() + (", " + ", ".join(bits[1:]) if bits[1:] else "") + ".") if bits \
        else f"A good {p.subcategory.replace('_', ' ')} stop that fits your time window."


@traced("finalize_plan")
def finalize_plan(state: PlannerState) -> dict:
    prefs, by_id, plan, cost = state["prefs"], _by_id(state), state.get("plan", []), state.get("cost", {})
    issues = state.get("issues", [])
    warnings = [i.message for i in issues if i.level == "warning"]
    ws, we = tools.window(prefs)

    stops_out = []
    for s in plan:
        p = by_id[s.place_id]
        stops_out.append({
            "place_id": p.id, "name": p.name, "area": p.area, "type": p.type, "subcategory": p.subcategory,
            "start": s.start, "end": s.end, "duration_min": s.duration_min,
            "travel_min": s.travel_min_from_prev, "travel_km": s.travel_km_from_prev,
            "cost": p.cost_per_person_inr, "crowd": tools.crowd_at(p, tools.to_min(s.start)),
            "veg": p.veg, "energy": p.energy, "source": p.source, "estimated_fields": p.estimated_fields,
            "sample_event": "sample_event" in p.tags, "lat": p.lat, "lon": p.lon,
            "reason": _template_reason(p, prefs, s),
        })

    status, how = "ok", "LLM"
    headline = f"Your Saturday in {state['city'].get('name', prefs.city)}"
    summary = state.get("plan_rationale") or ""
    tradeoffs = list(warnings)
    if stops_out:
        try:
            payload = {"user": prefs.model_dump(), "stops": [{k: v for k, v in s.items() if k not in ("lat", "lon")}
                                                             for s in stops_out],
                       "total_cost": cost.get("total"), "budget": prefs.budget_inr, "warnings": warnings}
            nar = structured_call(FinalNarrative, FINAL_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.6)
            reasons = {r.place_id: r.reason for r in nar.reasons}
            for s in stops_out:
                s["reason"] = reasons.get(s["place_id"], s["reason"])
            headline, summary, tradeoffs = nar.headline, nar.summary, nar.tradeoffs or tradeoffs
        except Exception as e:
            status, how = "fallback", f"template (LLM unavailable: {str(e)[:60]})"
            if not summary:
                summary = (f"{len(stops_out)} stops from {tools.to_hhmm(ws)} to {stops_out[-1]['end']}, "
                           f"around ₹{cost.get('total', 0)} in total.")

    caveats = []
    if any(s["sample_event"] for s in stops_out):
        caveats.append("Events marked 'sample' are illustrative, not real listings. Check local listings.")
    if any(s["source"] == "generic" for s in stops_out):
        caveats.append("Some stops are place types, not specific venues. Pick a nearby one.")
    if any("hours" in s["estimated_fields"] for s in stops_out):
        caveats.append("Opening hours for some places are estimates. Check before you go.")
    caveats.append("Prices and crowd levels are estimates based on place type and time of day.")
    friendly = {"stop_count": "Only a few places are open and suitable in this window, so the plan is shorter than usual.",
                "budget": "Even the cheapest workable plan goes over your budget."}
    caveats += list(dict.fromkeys(friendly.get(i.rule, f"Couldn't fully satisfy: {i.message}")
                                  for i in issues if i.level == "error"))

    notes = list(state.get("notes", []))
    if not stops_out:
        notes.append("Nothing suitable is open in this time window. Try a longer window or an earlier start.")
    final = {
        "type": "plan",
        "headline": headline, "summary": summary,
        "city": state["city"].get("name"), "data_source": state.get("data_source"),
        "planner": "fallback" if state.get("used_fallback") else "llm",
        "stops": stops_out,
        "totals": {"cost": cost.get("total", 0),
                   "travel_min": sum(s["travel_min"] for s in stops_out), "travel_km": cost.get("travel_km", 0), "budget": prefs.budget_inr,
                   "available_min": we - ws, "window": f"{tools.to_hhmm(ws)}–{tools.to_hhmm(we)}",
                   "ends_at": stops_out[-1]["end"] if stops_out else None},
        "tradeoffs": tradeoffs, "caveats": caveats,
        "assumptions": prefs.assumptions, "notes": notes,
    }
    return {"final": final, "_trace": {"status": status, "summary": f"explanations via {how}"}}
