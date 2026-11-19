"""Offline tests: validation rules, fallback planner, parsing, and the full graph without any API keys."""
import pytest

from app import data_sources as ds
from app import tools
from app.graph import run
from app.llm import heuristic_parse, parse_time_range
from app.models import Preferences, Stop


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEOAPIFY_API_KEY", raising=False)
    ds._cache.clear()


def t1_prefs(**kw):
    base = dict(city="Bangalore", budget_inr=2000, duration_min=240, start_time="15:00", energy="low",
                interests=["food", "music", "walks"], diet="vegetarian", avoid_crowds=True)
    return Preferences(**{**base, **kw})


def blr_candidates(prefs):
    city = tools.resolve_city(prefs.city)
    a, _, _ = tools.get_options("activity", city, prefs)
    f, _, _ = tools.get_options("food", city, prefs)
    return a + f


def plan_for(prefs, ids, durations=None):
    cands = blr_candidates(prefs)
    by_id = {p.id: p for p in cands}
    places = [by_id[i] for i in ids]
    return tools.evaluate(places, durations or [None] * len(ids), prefs, by_id)


def rules_of(issues, level="error"):
    return {i.rule for i in issues if i.level == level}


# ---------------------------------------------------------------- validation rules

def test_valid_plan_has_no_errors():
    _, _, issues, _ = plan_for(t1_prefs(), ["blr_cubbon_park", "blr_ngma", "blr_brahmins_coffee"], [60, 60, None])
    assert rules_of(issues) == set()


def test_unknown_id_is_error():
    prefs = t1_prefs()
    by_id = {p.id: p for p in blr_candidates(prefs)}
    issues = tools.validate_plan([Stop(place_id="made_up", start="15:00", end="16:00"),
                                  Stop(place_id="blr_mtr", start="16:30", end="17:30")], by_id, prefs, {"total": 0})
    assert "unknown_id" in rules_of(issues)


def test_duplicates_and_stop_count():
    _, _, issues, _ = plan_for(t1_prefs(), ["blr_cubbon_park"])
    assert "stop_count" in rules_of(issues)
    prefs = t1_prefs()
    by_id = {p.id: p for p in blr_candidates(prefs)}
    dup = [Stop(place_id="blr_mtr", start="15:00", end="16:00"), Stop(place_id="blr_mtr", start="16:00", end="17:00")]
    assert "duplicates" in rules_of(tools.validate_plan(dup, by_id, prefs, {"total": 700}))


def test_budget_error_and_warning():
    prefs = t1_prefs(budget_inr=1000)
    _, cost, issues, _ = plan_for(prefs, ["blr_windmills", "blr_cubbon_park"])  # ₹1500
    assert cost["total"] == 1500 and "budget" in rules_of(issues)
    prefs = t1_prefs(budget_inr=cost["total"] - 50)  # within 10%
    _, _, issues, _ = plan_for(prefs, ["blr_windmills", "blr_cubbon_park"])
    assert "budget" not in rules_of(issues) and "budget" in rules_of(issues, "warning")


def test_closed_place_is_error():
    prefs = t1_prefs(start_time="16:30", duration_min=180)
    by_id = {p.id: p for p in blr_candidates(prefs)}
    stops = [Stop(place_id="blr_ngma", start="16:30", end="18:00"),  # NGMA closes 17:00
             Stop(place_id="blr_mtr", start="18:15", end="19:15")]
    assert "hours" in rules_of(tools.validate_plan(stops, by_id, prefs, {"total": 370}))


def test_scheduler_trims_to_closing_time():
    stops, _, issues, _ = plan_for(t1_prefs(start_time="16:30", duration_min=180), ["blr_ngma", "blr_mtr"], [90, None])
    assert stops[0].end == "17:00" and "hours" not in rules_of(issues)


def test_window_overrun_is_error():
    prefs = t1_prefs(duration_min=100)
    stops = [Stop(place_id="blr_cubbon_park", start="15:00", end="16:00"),
             Stop(place_id="blr_mtr", start="16:10", end="17:10")]
    by_id = {p.id: p for p in blr_candidates(prefs)}
    assert "time_window" in rules_of(tools.validate_plan(stops, by_id, prefs, {"total": 350}))


def test_energy_and_diet_errors():
    prefs = Preferences(city="Mumbai", budget_inr=3000, duration_min=300, energy="low", diet="vegetarian",
                        interests=["nature"])
    city = tools.resolve_city("Mumbai")
    cands = tools.get_options("activity", city, prefs)[0] + tools.get_options("food", city, prefs)[0]
    by_id = {p.id: p for p in cands}
    stops = [Stop(place_id="bom_sgnp", start="15:00", end="16:30"),
             Stop(place_id="bom_swati_snacks", start="17:00", end="18:00")]
    assert "energy" in rules_of(tools.validate_plan(stops, by_id, prefs, {"total": 500}))


def test_avoid_crowds_is_warning():
    prefs = t1_prefs(start_time="17:30", duration_min=180)
    _, _, issues, _ = plan_for(prefs, ["blr_lalbagh", "blr_mtr"])  # Lalbagh is busy on Saturday evenings
    assert "crowds" in rules_of(issues, "warning")


# ---------------------------------------------------------------- fallback planner

def test_fallback_t1_is_valid_and_veg():
    prefs = t1_prefs()
    cands, _ = tools.filter_candidates(prefs, blr_candidates(prefs))
    places, within = tools.fallback_plan(prefs, cands)
    by_id = {p.id: p for p in cands}
    stops, cost, issues, _ = tools.evaluate(places, [None] * len(places), prefs, by_id)
    assert within and 2 <= len(stops) <= 4
    assert cost["total"] <= 2000
    assert all(by_id[s.place_id].veg in ("pure_veg", "veg_options") for s in stops if by_id[s.place_id].type == "food")
    assert rules_of(issues) == set()


def test_fallback_tight_budget_prefers_free():
    prefs = Preferences(city="Mumbai", budget_inr=200, duration_min=180, interests=["walks"])
    city = tools.resolve_city("Mumbai")
    cands, _ = tools.filter_candidates(prefs, tools.get_options("activity", city, prefs)[0]
                                       + tools.get_options("food", city, prefs)[0])
    places, within = tools.fallback_plan(prefs, cands)
    assert places and sum(p.cost_per_person_inr for p in places) <= 200


def test_unknown_city_is_generic():
    city = tools.resolve_city("Atlantis")
    assert city["source"] == "generic" and "Atlantis" in city["note"]


def test_late_start_excludes_museums():
    prefs = t1_prefs(start_time="21:00", duration_min=170, interests=["culture"])
    cands, stats = tools.filter_candidates(prefs, blr_candidates(prefs))
    assert stats["closed"] > 0 and all(p.subcategory != "museum" for p in cands)


# ---------------------------------------------------------------- parsing helpers

def test_opening_hours_parser():
    assert ds.parse_saturday_hours("Mo-Su 10:00-22:00") == [("10:00", "22:00")]
    assert ds.parse_saturday_hours("Mo-Fr 09:00-18:00; Sa 10:00-14:00; Su off") == [("10:00", "14:00")]
    assert ds.parse_saturday_hours("Mo-Fr 09:00-18:00; Sa,Su off") == []
    assert ds.parse_saturday_hours("Tu-Su 11:00-15:00,19:00-23:00") == [("11:00", "15:00"), ("19:00", "23:00")]
    assert ds.parse_saturday_hours("18:00-02:00") == [("18:00", "23:59")]
    assert ds.parse_saturday_hours("sunrise-sunset") is None


def test_heuristic_parse_chat():
    p = heuristic_parse("I'm in Mumbai, ₹1500, free from 4 to 9, want something chill and veg")
    assert (p.city, p.budget_inr, p.start_time, p.duration_min) == ("Mumbai", 1500, "16:00", 300)
    assert p.diet == "vegetarian" and p.energy == "low"
    assert parse_time_range("from 10am to 1pm") == ("10:00", 180)


# ---------------------------------------------------------------- full graph, no keys

def test_graph_offline_t1():
    out = run({"city": "Bangalore", "budget": 2000, "available_time": "4 hours", "start_time": "15:00",
               "mood": "tired but wants to do something fun", "interests": ["food", "music", "walks"],
               "constraints": ["vegetarian", "avoid crowded places"]})
    f = out["final"]
    assert f["type"] == "plan" and f["planner"] == "fallback" and f["data_source"] == "mock"
    assert 2 <= len(f["stops"]) <= 4 and f["totals"]["cost"] <= 2000
    steps = [e["step"] for e in out["trace"]]
    assert {"parse_preferences", "resolve_city", "get_food_options", "fallback_plan", "finalize_plan"} <= set(steps)


def test_graph_offline_vague():
    out = run("something fun this weekend")
    assert out["final"]["type"] == "clarify" and 1 <= len(out["final"]["questions"]) <= 2


def test_graph_followup_never_asks_twice():
    out = run("something fun this weekend", is_followup=True)
    assert out["final"]["type"] == "plan" and out["final"]["data_source"] == "generic"


def test_leg_limit_is_60_minutes():
    prefs = t1_prefs(duration_min=300)
    by_id = {p.id: p for p in blr_candidates(prefs)}
    def legs(travel):
        return [Stop(place_id="blr_cubbon_park", start="15:00", end="16:00"),
                Stop(place_id="blr_bic", start=tools.to_hhmm(960 + travel), end=tools.to_hhmm(1020 + travel),
                     travel_min_from_prev=travel)]
    assert "travel_leg" not in rules_of(tools.validate_plan(legs(55), by_id, prefs, {"total": 0}))
    assert "travel_leg" in rules_of(tools.validate_plan(legs(65), by_id, prefs, {"total": 0}))


def test_zero_budget_is_respected_not_defaulted():
    out = run({"city": "Bangalore", "budget": 0, "available_time": "4 hours", "start_time": "15:00",
               "interests": ["walks", "food"], "constraints": []})
    f = out["final"]
    assert f["totals"]["budget"] == 0 and f["totals"]["cost"] == 0 and f["stops"]
    assert all(s["cost"] == 0 for s in f["stops"])


def test_thirty_minute_window_gets_one_stop():
    out = run({"city": "Bangalore", "budget": 1000, "available_time": "30 minutes", "start_time": "17:00",
               "interests": ["food"], "constraints": []})
    f = out["final"]
    assert len(f["stops"]) == 1 and f["stops"][0]["end"] <= "17:30"
    assert not any("over" in n for n in f["notes"])
