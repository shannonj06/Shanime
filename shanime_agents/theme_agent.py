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


class ThemeQuery(BaseModel):
    desired_themes: str    # the theme they asked for, plus the angle they want on it
    embedding_text: str    # what gets embedded and matched against shows
    filters: MoodFilters


INSTRUCTIONS = f"""You turn a user's request for an anime *theme* into a structured search \
query for an anime recommendation engine.

A theme is what a story is about underneath its plot: grief and learning to move forward, \
identity and self-acceptance, obsession and self-destruction, loneliness hidden behind \
humor, found family and belonging, ambition and what it costs, guilt, revenge, mortality, \
growing up. Example: "anime about being the only one who remembers someone."

Read the request precisely. Near-synonyms are not the same theme -- loneliness is not \
depression, grief is not trauma, dark is not psychologically deep, and a sad story is not \
automatically a story *about* sadness. Also pick up the angle they want: realistic or \
symbolic, hopeful or devastating, sympathetic or critical, and anything they say they \
want to avoid.

Produce:
1. desired_themes: a short phrase naming the central theme and the angle they want on it, \
e.g. "grief and moving forward, quiet and realistic" or "obsession as self-destruction, \
told critically".
2. embedding_text: 2-3 sentences describing a show where this theme is CENTRAL -- it \
drives the main story or a main character's arc, not one scene or one side character. \
Describe how the theme surfaces: what the characters are wrestling with, the perspective \
the story takes on it, and the emotional tone. This text is embedded and matched against \
show descriptions AND against what real viewers wrote in reviews, so name the thematic \
arc, not a title. Do NOT put years, scores, or filter words in here.
3. filters: only set a filter when the user clearly asks for it; otherwise leave it null \
(for single values) or empty (for lists). A theme is NOT a filter request -- do not invent \
genres just because they sound thematically close. The `themes` filter below is the search \
engine's tag vocabulary, not the theme you described above; only use it when a tag is an \
exact match for something the user asked for.

You MUST only use values from these allowed lists. Pick the closest match, or leave the \
field empty if nothing fits. Never invent genre or theme labels.
- media_type: {MEDIA_TYPES} (leave null to search anime by default)
- genres: {GENRES}
- themes: {THEMES}
- year_min / year_max: integers between {MIN_YEAR} and {MAX_YEAR}
- score_min / score_max: floats between 0 and 10

Users often ask about depression, self-harm, addiction, abuse, or suicide. Write about how \
a story portrays those subjects -- "a character living with severe isolation", "a story \
about hopelessness and recovery" -- in neutral, non-romanticizing language. Never diagnose, \
and never assume someone asking for a dark theme is describing their own life.

Never recommend specific titles yourself. Your only job is to describe the theme well \
enough that the search engine can find it.
"""

theme_agent = Agent(
    name="theme_agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ThemeQuery,
)


# the user describes a theme they want a show to be about
# ex. "anime about grief where someone learns to move on"
async def create_theme_query(query, agent=theme_agent) -> ThemeQuery:
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


async def recommend_by_theme(query, top_k=5, agent=theme_agent):
    """Theme text -> structured query -> ranked shows."""
    theme = await create_theme_query(query, agent)
    # Theme matching leans on metadata: what a show is *about* lives in the synopsis
    # and tags, while reviews mostly say how it felt. Reviews still help, since that
    # is where viewers name the theme a synopsis leaves implicit.
    results = search(
        theme.embedding_text,
        top_k=top_k,
        meta_weight=.6,
        review_weight=.4,
        filters=_to_filters(theme.filters),
    )
    return theme, results


if __name__ == "__main__":
    import asyncio

    theme, results = asyncio.run(
        recommend_by_theme("anime about grief where someone learns to move on")
    )
    print(theme.desired_themes)
    print(theme.embedding_text)
    print(results[["title", "score", "search_score"]])
