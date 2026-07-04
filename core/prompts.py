"""
Prompt construction for each feature pipeline. Kept separate from the
Gemini client so prompt wording can be iterated on and unit-tested without
touching API logic.

Every builder takes the user `profile` dict and folds it into the prompt so
outputs are genuinely personalized rather than generic.
"""

from __future__ import annotations


def _language_instruction(profile: dict | None) -> str:
    """Determine language instructions based on user profile preferences."""
    lang = (profile or {}).get("language_pref", "English")
    if lang == "Hindi":
        return "Respond entirely in Hindi (Devanagari script)."
    if lang == "English + Hindi":
        return "Respond in English first, then provide a Hindi translation of the same content below it, clearly separated."
    return "Respond in English."


def _profile_summary(profile: dict | None) -> str:
    """Generate a text summary of the user's profile to inject into prompts."""
    if not profile:
        return "No profile details provided; give generally appealing suggestions for an Indian traveler."
    parts = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("age_band"):
        parts.append(f"Age band: {profile['age_band']}")
    if profile.get("gender") and profile["gender"] != "Prefer not to say":
        parts.append(f"Gender: {profile['gender']}")
    if profile.get("location"):
        parts.append(f"Home location: {profile['location']}")
    if profile.get("companions"):
        parts.append(f"Traveling as: {profile['companions']}")
    if profile.get("interests"):
        parts.append(f"Interests: {', '.join(profile['interests'])}")
    if profile.get("budget"):
        parts.append(f"Budget level: {profile['budget']}")
    return "; ".join(parts) if parts else "No profile details provided."


def build_recommendation_prompt(user_query: str, profile: dict, retrieved: list[dict]) -> tuple[str, str]:
    """Returns (system_instruction, user_prompt) for the destination recommender."""
    system = (
        "You are an expert Indian travel advisor who gives warm, specific, and practical "
        "destination recommendations grounded in real places across India. Never invent "
        "destinations that are not in the provided context. Keep the tone friendly and concise."
    )
    context_block = "\n".join(
        f"- {d['name']} ({d['state']}, {d['region']}): {d['description']} "
        f"[best season: {d['best_season']}]"
        for d in retrieved
    )
    user = f"""
Traveler profile: {_profile_summary(profile)}
Traveler's request: "{user_query}"

Candidate destinations (grounding context, choose and elaborate only from these unless
the traveler's request clearly needs a well-known place not listed):
{context_block}

Recommend 3 destinations from the candidates above best suited to this traveler.
For each: a short reason tied to their profile/interests, and one practical tip
(best time to visit, how to get around, or what to pack).
{_language_instruction(profile)}
"""
    return system, user.strip()


def build_hidden_gems_prompt(location_query: str, profile: dict) -> tuple[str, str]:
    """Returns (system_instruction, user_prompt) for the search-grounded hidden gems feature."""
    system = (
        "You are a well-traveled local Indian guide who specializes in lesser-known, "
        "authentic spots that most tourists miss. Use current, real information from "
        "search results. Avoid generic famous landmarks unless the traveler explicitly asks."
    )
    user = f"""
Traveler profile: {_profile_summary(profile)}
Traveler wants hidden gems near or related to: "{location_query}" in India.

Find 3 lesser-known, authentic places, experiences, or local spots (not the obvious
tourist landmarks) relevant to this location and the traveler's interests. For each,
explain briefly why it's special and how a visitor would find/access it.
{_language_instruction(profile)}
"""
    return system, user.strip()


def build_storytelling_prompt(place_name: str, profile: dict) -> tuple[str, str]:
    """Returns (system_instruction, user_prompt) for immersive storytelling + heritage promotion."""
    system = (
        "You are a knowledgeable Indian heritage storyteller, in the voice of a local "
        "historian who has spent a lifetime near this place. You bring history, legend, "
        "and living culture to life vividly but factually, and highlight why the heritage "
        "matters today."
    )
    user = f"""
Traveler profile: {_profile_summary(profile)}
Place: "{place_name}"

Write an immersive, evocative short story (250-350 words) about this place's history and
cultural significance, weaving in a local legend or historical event, one detail about
its architecture or geography, and one living tradition or festival still connected to it.
End with one sentence on why preserving this heritage matters.
{_language_instruction(profile)}
"""
    return system, user.strip()


def build_events_and_experiences_prompt(location_query: str, profile: dict, retrieved_experiences: list[dict]) -> tuple[str, str]:
    """Returns (system_instruction, user_prompt) for local events (search-grounded) + curated experiences."""
    system = (
        "You are a local Indian cultural concierge who connects travelers with real, "
        "current events and authentic hands-on cultural experiences. Use current search "
        "results for events, and personalize the curated experience suggestions to the traveler."
    )
    experiences_block = "\n".join(
        f"- {e['title']} ({e['location']}, category: {e['category']}): {e['description']}"
        for e in retrieved_experiences
    )
    user = f"""
Traveler profile: {_profile_summary(profile)}
Traveler is interested in events/experiences near or related to: "{location_query}" in India.

Part 1 - Current local events: search for any real festivals, fairs, or cultural events
happening in or near this location in the coming weeks. List up to 3 with approximate dates
if available.

Part 2 - Authentic cultural experiences (choose from this curated list, personalize the pitch):
{experiences_block}

Suggest up to 2 experiences from the list above best matched to the traveler's profile,
with a short personalized pitch for each.
{_language_instruction(profile)}
"""
    return system, user.strip()
