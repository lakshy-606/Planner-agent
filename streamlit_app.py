"""Perfect Saturday Planner: Streamlit UI with a live agent trace."""
from __future__ import annotations

import datetime as dt

import streamlit as st

from app.data_sources import get_secret
from app.graph import build_graph
from app.llm import model_name
from app.rules import KNOWN_INTERESTS

st.set_page_config(page_title="Perfect Saturday Planner", page_icon="🗓️", layout="centered")

STEP_ICONS = {"ok": "✅", "fallback": "🛟", "retry": "🔁", "error": "⚠️", "needs_input": "❓", "skipped": "⏭️"}
SOURCE_BADGE = {"live": ":green-badge[live data]", "mock": ":blue-badge[curated data]",
                "generic": ":orange-badge[template]"}
EXAMPLE = {
    "city": "Bangalore", "budget": 2000, "available_time": "4 hours", "start_time": "15:00",
    "mood": "tired but wants to do something fun", "interests": ["food", "music", "walks"],
    "constraints": ["vegetarian", "avoid crowded places"],
}

ss = st.session_state
ss.setdefault("result", None)
ss.setdefault("trace", [])
ss.setdefault("last_raw", None)


# ------------------------------------------------------------------ run the agent

def run_agent(raw, is_followup: bool = False) -> None:
    graph = build_graph()
    trace, final = [], None
    with st.status("Planning your Saturday…", expanded=True) as status:
        try:
            for update in graph.stream({"raw_input": raw, "is_followup": is_followup, "trace": []},
                                       stream_mode="updates"):
                for node, upd in update.items():
                    if not upd:
                        continue
                    for e in upd.get("trace", []):
                        trace.append(e)
                        src = f" · `{e['source']}`" if e.get("source") else ""
                        status.write(f"{STEP_ICONS.get(e['status'], '•')} **{e['step']}**{src} · {e['summary']} "
                                     f"<span style='color:gray'>({e['ms']} ms)</span>", unsafe_allow_html=True)
                    if upd.get("final"):
                        final = upd["final"]
        except Exception as e:  # last-resort guard so the reviewer never sees a stack trace
            status.update(label="Something went wrong", state="error")
            final = {"type": "error", "message": f"{type(e).__name__}: {e}"}
        else:
            label = "Need a bit more info" if final and final["type"] == "clarify" else "Plan ready"
            status.update(label=label, state="complete", expanded=False)
    ss.result, ss.trace, ss.last_raw = final, trace, raw


# ------------------------------------------------------------------ rendering

def render_plan(f: dict) -> None:
    st.subheader(f["headline"])
    if f.get("summary"):
        st.write(f["summary"])
    for n in f.get("notes", []):
        st.warning(n)

    t = f["totals"]
    c1, c2, c3 = st.columns(3)
    if t["budget"]:
        delta = t["budget"] - t["cost"]
        c1.metric("Total cost", f"₹{t['cost']:,}", f"₹{abs(delta):,} {'under' if delta >= 0 else 'over'} budget",
                  delta_color="normal" if delta >= 0 else "inverse")
    else:
        c1.metric("Total cost", f"₹{t['cost']:,}")
    c2.metric("Window", t["window"], f"ends {t['ends_at']}" if t["ends_at"] else None, delta_color="off")
    c3.metric("Travel", f"{t['travel_min']} min", f"{t['travel_km']} km", delta_color="off")
    planner = "AI-drafted, code-validated" if f["planner"] == "llm" else "Fallback planner (rule-based)"
    st.caption(f"{planner} · {SOURCE_BADGE.get(f['data_source'], '')}")

    for s in f["stops"]:
        if s["travel_min"]:
            st.caption(f"🚕 ~{s['travel_min']} min · {s['travel_km']} km")
        with st.container(border=True):
            left, right = st.columns([5, 1])
            badges = [SOURCE_BADGE[s["source"]]]
            if s["sample_event"]:
                badges.append(":red-badge[sample event]")
            if s["type"] == "food" and s["veg"] in ("pure_veg", "veg_options"):
                badges.append(":green-badge[pure veg]" if s["veg"] == "pure_veg" else ":gray-badge[veg options]")
            badges.append(f":gray-badge[{s['crowd']} at this time]")
            area = f" · {s['area']}" if s.get("area") else ""
            left.markdown(f"**{s['start']}–{s['end']}** &nbsp; **{s['name']}**{area}  \n" + " ".join(badges))
            right.markdown(f"**{'Free' if s['cost'] == 0 else '₹' + format(s['cost'], ',')}**  \n"
                           f"<span style='color:gray;font-size:0.8em'>approx</span>", unsafe_allow_html=True)
            st.markdown(f"_{s['reason']}_")

    if f.get("tradeoffs"):
        st.markdown("**Trade-offs**")
        st.markdown("\n".join(f"- {x}" for x in f["tradeoffs"]))
    with st.expander("Heads-up and assumptions"):
        for x in f.get("caveats", []) + f.get("assumptions", []):
            st.markdown(f"- {x}")
    pts = [{"lat": s["lat"], "lon": s["lon"]} for s in f["stops"] if s.get("lat") is not None]
    if len(pts) >= 1 and f["data_source"] != "generic":
        with st.expander("Map"):
            st.map(pts, size=60)


def render_trace() -> None:
    if not ss.trace:
        return
    with st.expander(f"Agent trace ({len(ss.trace)} steps)"):
        st.dataframe(
            [{"": STEP_ICONS.get(e["status"], ""), "step": e["step"], "source": e.get("source") or "",
              "summary": e["summary"], "ms": e["ms"]} for e in ss.trace],
            hide_index=True, width="stretch",
            column_config={"summary": st.column_config.TextColumn(width="large")},
        )


# ------------------------------------------------------------------ page

st.title("🗓️ Perfect Saturday Planner")
st.caption("Tell me your city, budget, time and mood. An agent finds real places, checks the timings, "
           "cost and travel, and explains why each stop fits you.")

with st.sidebar:
    st.markdown("### Status")
    st.markdown(f"- LLM: {'`' + model_name() + '`' if get_secret('OPENAI_API_KEY') else '❌ not set (rule-based mode)'}")
    st.markdown(f"- Places: {'Geoapify live' if get_secret('GEOAPIFY_API_KEY') else '❌ curated data only'}")
    st.markdown("### Try")
    st.markdown("- The prefilled form (brief example)\n- *something fun this weekend*\n- City: *Atlantis*\n"
                "- *Mumbai, ₹200, 3 hours, walks*\n- Start *21:00*, interests *culture*")
    if st.button("Clear result"):
        ss.result, ss.trace = None, []

form_tab, chat_tab = st.tabs(["📝 Form", "💬 Describe it"])
with form_tab:
    with st.form("prefs_form"):
        c1, c2 = st.columns(2)
        city = c1.text_input("City", EXAMPLE["city"])
        budget = c2.number_input("Budget per person (₹)", min_value=0, value=EXAMPLE["budget"], step=100)
        c3, c4 = st.columns(2)
        available = c3.text_input("Available time", EXAMPLE["available_time"])
        start = c4.time_input("Start time", dt.time(15, 0), step=1800)
        mood = st.text_input("Mood", EXAMPLE["mood"])
        interests = st.multiselect("Interests", KNOWN_INTERESTS, EXAMPLE["interests"])
        constraints = st.multiselect("Constraints", ["vegetarian", "vegan", "avoid crowded places"],
                                     EXAMPLE["constraints"])
        if st.form_submit_button("Plan my Saturday", type="primary", width="stretch"):
            run_agent({"city": city, "budget": budget, "available_time": available,
                       "start_time": start.strftime("%H:%M"), "mood": mood,
                       "interests": interests, "constraints": constraints})

with chat_tab:
    with st.form("chat_form"):
        text = st.text_area("What's your Saturday looking like?",
                            placeholder="I'm in Mumbai, ₹1500, free from 4 to 9, want something chill and veg")
        if st.form_submit_button("Plan my Saturday", type="primary", width="stretch") and text.strip():
            run_agent(text.strip())

res = ss.result
if res:
    st.divider()
    if res["type"] == "clarify":
        st.info("I need a little more to plan something good:")
        for q in res["questions"]:
            st.markdown(f"- **{q}**")
        with st.form("clarify_form"):
            answer = st.text_input("Your answer", placeholder="Pune, about ₹1000, from 4 to 8pm, food and music")
            if st.form_submit_button("Continue", type="primary") and answer.strip():
                raw = ss.last_raw
                raw = {**raw, "additional_info": answer} if isinstance(raw, dict) else f"{raw}. {answer}"
                run_agent(raw, is_followup=True)
                st.rerun()
    elif res["type"] == "plan":
        render_plan(res)
    else:
        st.error("The planner hit an unexpected error. Please try again.")
        st.caption(res.get("message", ""))
    render_trace()
