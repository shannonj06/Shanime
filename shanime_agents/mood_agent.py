from typing import List, Optional

from pydantic import BaseModel, Field
from agents import Agent, Runner
from search.search_engine import FILTERS, search

# Pull the allowed vocabulary from the search engine itself so the agent can only
# emit filter values that apply_filters actually understands.
MEDIA_TYPES = FILTERS["media_type"]["allowed"]
GENRES = FILTERS["genres"]["allowed"]
THEMES = FILTERS["themes"]["allowed"]
MIN_YEAR = FILTERS["year_min"]["allowed"][0]
MAX_YEAR = FILTERS["year_max"]["allowed"][0]


class MoodFilters(BaseModel):
    media_type: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None


class MoodQuery(BaseModel):
    current_mood: str      # how the user says they feel now
    desired_mood: str      # how they want to feel afterwards
    embedding_text: str    # what gets embedded and matched against shows
    filters: MoodFilters


INSTRUCTIONS = f"""You turn a user's emotional state into a structured search query for \
an anime recommendation engine.

The user tells you two things, sometimes only implicitly: how they feel right now, and \
how they want to feel after watching. Example: "I'm burnt out and want to feel hopeful."

Produce:
1. current_mood: a short phrase naming the feeling they are in now.
2. desired_mood: a short phrase naming the feeling they want to end with.
3. embedding_text: 2-3 sentences describing the *emotional experience* of a show that \
would carry someone from current_mood to desired_mood. This text is embedded and matched \
against show descriptions AND against what real viewers wrote in reviews, so describe how \
the show feels to watch and what it leaves you with -- gentle pacing, quiet comfort, a \
cathartic cry, slow-building hope, warmth, momentum. Name the emotional arc, not a title. \
Do NOT put years, scores, or filter words in here.
4. filters: only set a filter when the user clearly asks for it; otherwise leave it null \
(for single values) or empty (for lists). A mood is NOT a filter request -- do not invent \
genres just because they sound emotionally close.

You MUST only use values from these allowed lists. Pick the closest match, or leave the \
field empty if nothing fits. Never invent genre or theme labels.
- media_type: {MEDIA_TYPES} (leave null to search anime by default)
- genres: {GENRES}
- themes: {THEMES}
- year_min / year_max: integers between {MIN_YEAR} and {MAX_YEAR}
- score_min / score_max: floats between 0 and 10

Never recommend specific titles yourself. Your only job is to describe the feeling well \
enough that the search engine can find it.
"""

mood_agent = Agent(
    name="mood_agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=MoodQuery,
)


# the user describes how they feel and how they want to feel afterwards
# ex. "im burnt out and want to feel hopeful"
async def create_mood_query(query, agent=mood_agent) -> MoodQuery:
    result = await Runner.run(agent, query)
    return result.final_output  # -> MoodQuery

#this converts the mood filters object into a dictionary so the search engine can take it in
def _to_filters(mood_filters: MoodFilters):
    # apply_filters skips None/"" but not [], and genres/themes are list_contains
    # (one value per filter), so drop empties and take the first label.
    filters = {}
    for key, value in mood_filters.model_dump().items():
        if value is None or value == "" or value == []:
            continue
        filters[key] = value[0] if isinstance(value, list) else value
    return filters


async def recommend_by_mood(query, top_k=5, agent=mood_agent):
    """Mood text -> structured query -> ranked shows."""
    mood = await create_mood_query(query, agent)
    # Mood matching leans on reviews: how a show made people feel lives in what
    # viewers wrote about it, not in the synopsis.
    results = search(
        mood.embedding_text,
        top_k=top_k,
        meta_weight=.4,
        review_weight=.6,
        filters=_to_filters(mood.filters),
    )
    return mood, results


if __name__ == "__main__":
    import asyncio

    mood, results = asyncio.run(
        recommend_by_mood("im burnt out from finals and i want to feel hopeful again")
    )
    print(f"{mood.current_mood} -> {mood.desired_mood}")
    print(mood.embedding_text)
    print(results[["title", "score", "search_score"]])
