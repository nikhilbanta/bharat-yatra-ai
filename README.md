# Bharat Yatra AI

A GenAI-powered platform for discovering Indian destinations, hidden gems,
heritage storytelling, local events, and authentic cultural experiences —
built for the "Destination Discovery & Cultural Experiences" challenge.

## Access

No login/password is required — enter any name or email to get a personalized
session. History and profile are saved locally against that identifier.

**Note on identity**: this demo intentionally uses a name/email as the only
identifier, with no password, to keep the flow frictionless. This means it is
not real access control — don't enter sensitive personal data. This tradeoff
is disclosed to users in the app itself.

## Features

| Feature | What it does | AI technique |
|---|---|---|
| Destination recommender | Personalized picks from a curated pan-India dataset | RAG-lite: keyword/embedding retrieval + Gemini generation |
| Hidden gems finder | Surfaces lesser-known spots near a place | Gemini + Google Search grounding |
| Storytelling & heritage | Immersive short stories about a place's history/culture | Gemini generation, persona-prompted |
| Events & experiences | Current local events + curated craft/homestay experiences | Gemini + Google Search grounding, combined with curated dataset |

All four features make a real, live call to the Gemini API on every use —
nothing is hardcoded or pre-generated.

## Setup

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. `pip install -r requirements.txt`
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
   add your key.
4. Run: `streamlit run app.py`

## Testing

```
pytest tests/
```

Tests cover retrieval logic, prompt construction, and the database layer —
all independent of live network calls, so they run in CI without an API key.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On https://share.streamlit.io, create a new app pointing at `app.py`.
3. In the app's Settings → Secrets, paste the contents of your
   `secrets.toml` (i.e. `GEMINI_API_KEY = "..."`).
4. Deploy.

**Note**: Streamlit Community Cloud's filesystem is ephemeral — the SQLite
file (`app.db`) may reset on redeploys or app restarts. This is acceptable
for a demo; for durable long-term storage, swap `core/db.py` for a hosted
Postgres (e.g. Supabase free tier) using the same function signatures.

## Architecture

```
Streamlit UI (app.py)
   │
   ├── core/db.py            SQLite: profiles + history, keyed by identifier
   ├── core/retrieval.py     Embedding/keyword search over curated datasets
   ├── core/prompts.py       Prompt templates per feature, profile-aware
   └── core/gemini_client.py Gemini API wrapper (plain + search-grounded)
              │
              ├── data/destinations.json   curated pan-India destinations
              └── data/experiences.json    curated pan-India cultural experiences
```

## Security notes

- API key is read from `st.secrets`/environment only — never hardcoded.
- No password-based auth; this is disclosed to users (see above).
- No sensitive data is requested; profile fields are optional beyond name.
- Grounding sources are shown to the user for transparency on AI-generated
  event/hidden-gem claims.
