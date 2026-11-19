"""Typed data models shared by every part of the agent."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Energy = Literal["low", "medium", "high"]
Veg = Literal["pure_veg", "veg_options", "limited_veg", "n/a"]
Source = Literal["live", "mock", "generic"]


class Preferences(BaseModel):
    city: Optional[str] = None
    budget_inr: Optional[int] = None
    duration_min: Optional[int] = None
    start_time: str = "15:00"
    mood: Optional[str] = None
    energy: Energy = "medium"
    interests: list[str] = Field(default_factory=list)
    diet: Literal["vegetarian", "vegan", "any"] = "any"
    avoid_crowds: bool = False
    assumptions: list[str] = Field(default_factory=list)


class Place(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    type: Literal["activity", "food", "event"]
    subcategory: str = "event"
    area: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    interests: list[str] = Field(default_factory=list)
    cost_per_person_inr: int = 0
    energy: Energy = "low"
    crowd: dict[str, str] = Field(default_factory=dict)
    veg: Veg = "n/a"
    hours: list[tuple[str, str]] = Field(default_factory=list)
    duration_min: int = 60
    indoor: bool = False
    tags: list[str] = Field(default_factory=list)
    source: Source = "mock"
    estimated_fields: list[str] = Field(default_factory=list)

    def short(self) -> dict[str, Any]:
        """Compact view handed to the LLM."""
        return {
            "id": self.id, "name": self.name, "type": self.type,
            "subcategory": self.subcategory, "area": self.area,
            "interests": self.interests, "cost": self.cost_per_person_inr,
            "energy": self.energy, "crowd": self.crowd, "veg": self.veg,
            "hours": self.hours, "duration_min": self.duration_min,
        }


class Stop(BaseModel):
    place_id: str
    start: str = ""
    end: str = ""
    duration_min: int = 0
    travel_min_from_prev: int = 0
    travel_km_from_prev: float = 0.0
    reason: str = ""


# ---- LLM structured-output schemas ----

class DraftStop(BaseModel):
    place_id: str = Field(description="Must be one of the candidate ids")
    duration_min: Optional[int] = Field(None, description="Minutes to spend; omit for the typical duration")


class DraftPlan(BaseModel):
    stops: list[DraftStop]
    rationale: str = Field("", description="One line on the overall shape of the plan")


class StopReason(BaseModel):
    place_id: str
    reason: str = Field(description="One sentence: why this stop fits this user")


class FinalNarrative(BaseModel):
    headline: str = Field(description="A short, friendly title for the plan")
    summary: str = Field(description="Two sentences describing the afternoon")
    reasons: list[StopReason]
    tradeoffs: list[str] = Field(description="Honest trade-offs and caveats")


class Issue(BaseModel):
    level: Literal["error", "warning"]
    rule: str
    message: str


class PlannerState(TypedDict, total=False):
    raw_input: Any
    is_followup: bool               # answer to clarifying questions: never ask twice
    prefs: Preferences
    clarifying_questions: list[str]
    city: dict                       # coords, source, city_key, name, note (from tools.resolve_city)
    data_source: str
    notes: list[str]                 # user-facing notes (e.g. unknown city)
    activity_candidates: list[Place]
    food_candidates: list[Place]
    candidates: list[Place]
    plan: list[Stop]
    plan_rationale: str
    llm_failed: bool
    cost: dict
    issues: list[Issue]
    retries: int
    used_fallback: bool
    final: dict
    trace: Annotated[list[dict], operator.add]
