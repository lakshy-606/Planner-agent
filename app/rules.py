"""Rule tables used to enrich live data with fields APIs don't provide (cost, crowd, energy)."""
from __future__ import annotations

COST_BY_SUBCATEGORY = {
    "cafe": 400, "restaurant": 600, "fine_dining": 1500, "fast_food": 250,
    "bar": 1200, "pub": 1200, "museum": 100, "park": 0, "garden": 30,
    "bookstore": 300, "shopping_mall": 800, "marketplace": 300,
    "cultural_centre": 0, "theatre": 500, "art_gallery": 0, "sights": 50,
    "nature": 0, "beach": 0, "ice_cream": 150,
}
DEFAULT_COST = {"activity": 200, "food": 500, "event": 500}

CROWD_RULES = {  # Saturday defaults
    "shopping_mall": {"morning": "moderate", "afternoon": "busy", "evening": "busy"},
    "marketplace": {"morning": "moderate", "afternoon": "busy", "evening": "busy"},
    "park": {"morning": "moderate", "afternoon": "quiet", "evening": "moderate"},
    "nature": {"morning": "moderate", "afternoon": "quiet", "evening": "moderate"},
    "beach": {"morning": "moderate", "afternoon": "moderate", "evening": "busy"},
    "museum": {"morning": "quiet", "afternoon": "moderate", "evening": "quiet"},
    "art_gallery": {"morning": "quiet", "afternoon": "quiet", "evening": "quiet"},
    "cafe": {"morning": "quiet", "afternoon": "moderate", "evening": "moderate"},
    "restaurant": {"morning": "quiet", "afternoon": "moderate", "evening": "busy"},
    "bar": {"morning": "quiet", "afternoon": "quiet", "evening": "busy"},
    "pub": {"morning": "quiet", "afternoon": "quiet", "evening": "busy"},
    "bookstore": {"morning": "quiet", "afternoon": "quiet", "evening": "moderate"},
    "sights": {"morning": "moderate", "afternoon": "busy", "evening": "moderate"},
}
DEFAULT_CROWD = {"morning": "quiet", "afternoon": "moderate", "evening": "moderate"}

ENERGY_BY_SUBCATEGORY = {
    "park": "low", "museum": "low", "cafe": "low", "bookstore": "low",
    "art_gallery": "low", "restaurant": "low", "garden": "low",
    "trek": "high", "national_park": "high", "gaming": "medium",
    "shopping_mall": "medium", "marketplace": "medium", "sights": "medium",
    "nature": "medium", "bar": "medium", "pub": "medium", "beach": "low",
}

TYPICAL_DURATION = {
    "cafe": 60, "restaurant": 75, "fast_food": 40, "bar": 90, "pub": 90,
    "museum": 75, "park": 60, "bookstore": 45, "shopping_mall": 90,
    "marketplace": 60, "art_gallery": 60, "cultural_centre": 75, "sights": 45,
    "nature": 60, "beach": 60, "theatre": 120,
}

DEFAULT_HOURS = {
    "park": [("06:00", "19:00")], "nature": [("06:00", "18:30")], "beach": [("05:00", "22:00")],
    "museum": [("10:00", "17:00")], "art_gallery": [("11:00", "19:00")],
    "cafe": [("09:00", "22:00")], "restaurant": [("11:30", "22:30")], "fast_food": [("11:00", "23:00")],
    "bar": [("17:00", "23:30")], "pub": [("12:00", "23:30")],
    "bookstore": [("10:30", "20:30")], "shopping_mall": [("10:00", "22:00")],
    "marketplace": [("10:00", "21:00")], "cultural_centre": [("10:00", "20:00")],
    "sights": [("09:00", "18:00")], "theatre": [("12:00", "22:00")],
}
FALLBACK_HOURS = [("10:00", "20:00")]

MOOD_TO_ENERGY = {  # hint only; the LLM decides
    "tired": "low", "chill": "low", "lazy": "low", "relaxed": "low", "calm": "low",
    "energetic": "high", "adventurous": "high", "active": "high", "pumped": "high",
}

# Interest -> Geoapify categories
INTEREST_CATEGORIES = {
    "walks": ["leisure.park", "natural"],
    "nature": ["leisure.park", "natural", "beach"],
    "art": ["entertainment.museum", "entertainment.culture"],
    "culture": ["entertainment.museum", "entertainment.culture", "tourism.sights"],
    "books": ["commercial.books"],
    "music": ["entertainment.culture", "catering.bar", "catering.pub"],
    "nightlife": ["catering.bar", "catering.pub"],
    "shopping": ["commercial.shopping_mall", "commercial.marketplace"],
}
DEFAULT_ACTIVITY_CATEGORIES = ["leisure.park", "entertainment.museum", "entertainment.culture"]
FOOD_CATEGORIES = ["catering.restaurant", "catering.cafe"]

# Geoapify category prefix -> (our subcategory, interests). Most specific first.
GEOAPIFY_SUBCATEGORY = [
    ("catering.cafe", "cafe", ["food"]),
    ("catering.fast_food", "fast_food", ["food"]),
    ("catering.ice_cream", "ice_cream", ["food"]),
    ("catering.restaurant", "restaurant", ["food"]),
    ("catering.bar", "bar", ["nightlife", "music"]),
    ("catering.pub", "pub", ["nightlife", "music"]),
    ("entertainment.museum", "museum", ["art", "culture"]),
    ("entertainment.culture.theatre", "theatre", ["culture", "music"]),
    ("entertainment.culture.gallery", "art_gallery", ["art", "culture"]),
    ("entertainment.culture", "cultural_centre", ["culture", "music", "art"]),
    ("commercial.books", "bookstore", ["books"]),
    ("commercial.shopping_mall", "shopping_mall", ["shopping"]),
    ("commercial.marketplace", "marketplace", ["shopping"]),
    ("leisure.park", "park", ["walks", "nature"]),
    ("beach", "beach", ["walks", "nature"]),
    ("natural", "nature", ["walks", "nature"]),
    ("tourism.sights", "sights", ["culture"]),
]

KNOWN_INTERESTS = ["food", "music", "walks", "nature", "art", "culture", "books", "shopping", "nightlife"]

# Interest synonyms used by the heuristic (non-LLM) parser and normalisation
INTEREST_SYNONYMS = {
    "food": ["food", "eat", "dinner", "lunch", "snack", "cafe", "coffee", "brunch"],
    "music": ["music", "gig", "concert", "jazz", "live band"],
    "walks": ["walk", "stroll", "wander"],
    "nature": ["nature", "park", "garden", "green", "lake", "beach", "outdoor"],
    "art": ["art", "gallery", "painting"],
    "culture": ["culture", "museum", "heritage", "history", "theatre", "theater"],
    "books": ["book", "reading", "library"],
    "shopping": ["shop", "mall", "market"],
    "nightlife": ["nightlife", "bar", "pub", "drinks", "club"],
}


def slot_of(hhmm: str) -> str:
    h = int(hhmm.split(":")[0])
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    return "evening"
