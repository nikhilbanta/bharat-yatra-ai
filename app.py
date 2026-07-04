"""
Bharat Yatra AI — GenAI-powered destination discovery & cultural experience
platform for India.

Run locally:
    export GEMINI_API_KEY=your_key_here
    streamlit run app.py
"""

import streamlit as st

from core import db
from core.gemini_client import GeminiClientError, GeminiQuotaError, generate, generate_grounded
from core.prompts import (
    build_events_and_experiences_prompt,
    build_hidden_gems_prompt,
    build_recommendation_prompt,
    build_storytelling_prompt,
)
from core.retrieval import search_destinations, search_experiences

st.set_page_config(page_title="Bharat Yatra AI", page_icon="🧭", layout="wide")

db.init_db()

INTERESTS_OPTIONS = [
    "Heritage & history", "Food & cuisine", "Adventure & trekking",
    "Spiritual & pilgrimage", "Nature & wildlife", "Art & handicrafts",
    "Festivals & music",
]


def _identifier_key() -> str | None:
    """Retrieve the current user identifier from the session state."""
    return st.session_state.get("identifier")


def render_login() -> None:
    """Render the login screen for user authentication."""
    st.title("Bharat Yatra AI")
    st.caption("Discover India's destinations, hidden gems, and living heritage — powered by GenAI.")
    st.info(
        "Enter your name or email to save and revisit your trip history across visits. "
        "This demo uses no password — please don't enter sensitive personal details."
    )
    with st.form("login_form"):
        identifier = st.text_input("Name or email", placeholder="e.g. priya@example.com", help="Enter your name or email to login")
        submitted = st.form_submit_button("Continue", help="Click to continue and login")
        if submitted:
            if not identifier.strip():
                st.error("Please enter a name or email to continue.")
            else:
                st.session_state["identifier"] = identifier.strip().lower()
                st.rerun()


def render_profile_form(existing: dict | None) -> None:
    """Render the profile form in the sidebar to collect user preferences."""
    st.subheader("Tell us about yourself")
    st.caption("All fields are optional except name. Used only to personalize your suggestions.")
    existing = existing or {}
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=existing.get("name", ""), help="Enter your full name")
            age_band = st.selectbox(
                "Age group", ["Prefer not to say", "18-25", "26-40", "41-60", "60+"],
                index=_safe_index(["Prefer not to say", "18-25", "26-40", "41-60", "60+"], existing.get("age_band")),
                help="Select your age group"
            )
            gender = st.selectbox(
                "Gender", ["Prefer not to say", "Female", "Male", "Other"],
                index=_safe_index(["Prefer not to say", "Female", "Male", "Other"], existing.get("gender")),
                help="Select your gender identity"
            )
            location = st.text_input("Home city/state", value=existing.get("location", ""), help="Enter your home city or state")
        with col2:
            companions = st.selectbox(
                "Traveling as", ["Solo", "Couple", "Family", "Friends group"],
                index=_safe_index(["Solo", "Couple", "Family", "Friends group"], existing.get("companions")),
                help="Select who you are traveling with"
            )
            budget = st.selectbox(
                "Budget level", ["Budget", "Mid-range", "Luxury"],
                index=_safe_index(["Budget", "Mid-range", "Luxury"], existing.get("budget")),
                help="Select your preferred budget level"
            )
            language_pref = st.selectbox(
                "Response language", ["English", "Hindi", "English + Hindi"],
                index=_safe_index(["English", "Hindi", "English + Hindi"], existing.get("language_pref")),
                help="Select the language for AI responses"
            )
            interests = st.multiselect("Interests", INTERESTS_OPTIONS, default=existing.get("interests", []), help="Select one or more travel interests")

        submitted = st.form_submit_button("Save profile", help="Save your preferences")
        if submitted:
            profile = {
                "name": name, "age_band": age_band, "gender": gender, "location": location,
                "companions": companions, "interests": interests, "budget": budget,
                "language_pref": language_pref,
            }
            db.save_profile(_identifier_key() or "", profile)
            st.success("Profile saved.")
            st.rerun()


def _safe_index(options: list[str], value: str | None) -> int:
    """Return the index of a value in a list safely, defaulting to 0."""
    try:
        return options.index(value) if value in options else 0
    except Exception:
        return 0


def _show_gemini_error(exc: Exception, action: str) -> None:
    """
    Shows a clean, actionable message for Gemini errors instead of a raw
    stack trace. Quota errors get specific guidance since they're usually
    an account setup issue, not a real bug.
    """
    if isinstance(exc, GeminiQuotaError):
        st.error(
            f"Could not {action} — the Gemini API quota was hit.\n\n"
            "If you're using a fresh API key, your Google project may need billing "
            "linked at [aistudio.google.com](https://aistudio.google.com) to unlock "
            "free-tier access (you won't be charged for free-tier usage — this just "
            "validates the account). Otherwise, please wait a moment and try again."
        )
    else:
        st.error(f"Could not {action} right now. Please try again in a moment.")


def render_recommender(profile: dict) -> None:
    """Render the destination recommendation feature."""
    st.subheader("Destination recommender")
    query = st.text_input(
        "What kind of trip are you looking for?",
        placeholder="e.g. a quiet heritage getaway for a long weekend",
        key="rec_query",
        help="Describe the kind of trip you want to take"
    )
    if st.button("Get recommendations", key="rec_btn", help="Click to get personalized AI recommendations") and query:
        with st.spinner("Finding destinations that match your profile..."):
            try:
                retrieved = search_destinations(query, top_k=6)
                system, user_prompt = build_recommendation_prompt(query, profile, retrieved)
                result = generate(user_prompt, system_instruction=system)
                st.markdown(result.text)
                db.add_history_entry(_identifier_key() or "", "recommendation", query, result.text)
            except GeminiClientError as e:
                st.error(f"Could not generate recommendations right now: {e}")


def render_hidden_gems(profile: dict) -> None:
    """Render the hidden gems discovery feature."""
    st.subheader("Hidden gems finder")
    location = st.text_input(
        "Which place or region should we explore for hidden gems?",
        placeholder="e.g. around Coorg, Karnataka",
        key="gems_query",
        help="Enter a location to find offbeat spots and hidden gems"
    )
    if st.button("Find hidden gems", key="gems_btn", help="Click to search for hidden gems") and location:
        with st.spinner("Searching for lesser-known spots..."):
            try:
                system, user_prompt = build_hidden_gems_prompt(location, profile)
                result = generate_grounded(user_prompt, system_instruction=system)
                st.markdown(result.text)
                if result.sources:
                    with st.expander("Sources"):
                        for s in result.sources:
                            st.write(s)
                db.add_history_entry(_identifier_key() or "", "hidden_gems", location, result.text)
            except GeminiClientError as e:
                st.error(f"Could not search for hidden gems right now: {e}")


def render_storytelling(profile: dict) -> None:
    """Render the immersive storytelling and heritage feature."""
    st.subheader("Immersive storytelling & heritage")
    place = st.text_input(
        "Which place would you like a story about?",
        placeholder="e.g. Hampi, Karnataka",
        key="story_query",
        help="Enter a place to hear an immersive historical or cultural story"
    )
    if st.button("Tell me the story", key="story_btn", help="Click to generate the story") and place:
        with st.spinner("Weaving the story..."):
            try:
                system, user_prompt = build_storytelling_prompt(place, profile)
                result = generate(user_prompt, system_instruction=system)
                st.markdown(result.text)
                db.add_history_entry(_identifier_key() or "", "storytelling", place, result.text)
            except GeminiClientError as e:
                st.error(f"Could not generate the story right now: {e}")


def render_events_experiences(profile: dict) -> None:
    """Render the local events and experiences feature."""
    st.subheader("Local events & authentic experiences")
    location = st.text_input(
        "Which place or region are you interested in?",
        placeholder="e.g. Jaipur, Rajasthan",
        key="events_query",
        help="Enter a location to find local events and cultural experiences"
    )
    if st.button("Find events & experiences", key="events_btn", help="Click to search for events and experiences") and location:
        with st.spinner("Looking up current events and matching experiences..."):
            try:
                retrieved = search_experiences(location, top_k=5)
                system, user_prompt = build_events_and_experiences_prompt(location, profile, retrieved)
                result = generate_grounded(user_prompt, system_instruction=system)
                st.markdown(result.text)
                if result.sources:
                    with st.expander("Sources"):
                        for s in result.sources:
                            st.write(s)
                db.add_history_entry(_identifier_key() or "", "events_experiences", location, result.text)
            except GeminiClientError as e:
                st.error(f"Could not fetch events/experiences right now: {e}")


def render_history() -> None:
    """Render the history of the user's past interactions."""
    st.subheader("Your past history")
    entries = db.get_history(_identifier_key() or "", limit=20)
    if not entries:
        st.caption("No history yet — try one of the features above.")
        return
    for entry in entries:
        with st.expander(f"{entry['feature_type']} — {entry['query_input']} ({entry['created_at'][:19]})"):
            st.markdown(entry["ai_output"])


def main() -> None:
    """Main entry point for the Streamlit app."""
    if not _identifier_key():
        render_login()
        return

    profile = db.get_profile(_identifier_key() or "")

    with st.sidebar:
        st.write(f"Signed in as **{_identifier_key()}**")
        if st.button("Sign out", help="Click to sign out of your profile"):
            del st.session_state["identifier"]
            st.rerun()
        st.divider()
        render_profile_form(profile)

    if not profile or not profile.get("name"):
        st.warning("Please fill in your profile in the sidebar to get personalized suggestions.")

    st.title("Bharat Yatra AI 🧭")
    st.caption("Discover India's destinations, hidden gems, and living heritage — powered by GenAI.")

    tabs = st.tabs(["Recommender", "Hidden gems", "Storytelling", "Events & experiences", "History"])
    with tabs[0]:
        render_recommender(profile or {})
    with tabs[1]:
        render_hidden_gems(profile or {})
    with tabs[2]:
        render_storytelling(profile or {})
    with tabs[3]:
        render_events_experiences(profile or {})
    with tabs[4]:
        render_history()


if __name__ == "__main__":
    main()
