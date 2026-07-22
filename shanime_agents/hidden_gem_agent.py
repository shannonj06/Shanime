from typing import List, Literal, Optional

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

# What counts as a hidden gem is a data policy, not something to let the model
# improvise per request -- it picks a tier, these numbers do the rest.
# members_max is the "hidden" half, score_min is the "gem" half, and scored_by_min
# is what stops a 9.0 off forty votes from beating a genuinely good obscure show.
# Pools on the current dataset: 605 / 1051 / 1073 anime.
TIERS = {
    "deep_cut": {"members_max": 5000, "score_min": 7.0, "scored_by_min": 500},
    "obscure": {"members_max": 20000, "score_min": 7.2, "scored_by_min": 750},
    "underrated": {"members_max": 100000, "score_min": 7.5, "scored_by_min": 1000},
}


class GemFilters(BaseModel):
    media_type: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None


class GemQuery(BaseModel):
    what_they_want: str    # the kind of show they're after, in a short phrase
    embedding_text: str    # what gets embedded and matched against shows
    obscurity: Literal["deep_cut", "obscure", "underrated"]
    filters: GemFilters


INSTRUCTIONS = f"""You turn a request for an overlooked anime into a structured search \
query for an anime recommendation engine.

The user wants something good that they have not already been told about ten times. Your \
job is to describe what they actually want to watch, and to judge how far off the beaten \
path they are asking you to go. Example: "something underrated and quiet, nothing \
mainstream."

Produce:
1. what_they_want: a short phrase naming the kind of show they're after.
2. embedding_text: 2-3 sentences describing the show itself -- what happens in it, what it \
is about, how it feels to watch. This text is embedded and matched against show \
descriptions AND against what real viewers wrote in reviews, so describe the show, not the \
fact that it is overlooked. Words like "hidden gem", "underrated", or "slept on" say \
nothing about content and will only pull in noise, so keep them out entirely. If they named \
a show they liked, describe what that show is *like* rather than naming it. Do NOT put \
years, scores, or filter words in here.
3. obscurity: how far off the radar they are asking you to go.
   - "underrated": they want something good that is merely under-discussed -- slept on, \
overlooked, something they missed. This is the default when they give you no real signal.
   - "obscure": they explicitly want something most people have not heard of, or they say \
mainstream and popular shows are not what they want.
   - "deep_cut": a heavy watcher asking for the truly buried -- they say they have seen \
everything, or they want the most obscure thing you have. Rare. Do not reach for it just \
because someone said "hidden gem".
4. filters: only set a filter when the user clearly asks for it; otherwise leave it null \
(for single values) or empty (for lists). Do not invent genres just because they sound \
close to what they described.

Do NOT try to make the results obscure through the filters. The search engine already does \
that from the obscurity tier, and a score_min you set on your own to "guarantee quality" \
will override it and work against you. Only set score_min if the user actually named a \
rating they want.

You MUST only use values from these allowed lists. Pick the closest match, or leave the \
field empty if nothing fits. Never invent genre or theme labels.
- media_type: {MEDIA_TYPES} (leave null to search anime by default)
- genres: {GENRES}
- themes: {THEMES}
- year_min / year_max: integers between {MIN_YEAR} and {MAX_YEAR}
- score_min / score_max: floats between 0 and 10

Never recommend specific titles yourself. Your only job is to describe what they want well \
enough that the search engine can find it.
"""

hidden_gem_agent = Agent(
    name="hidden_gem_agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=GemQuery,
)


# the user asks for something good that nobody talks about
# ex. "give me an underrated anime nobody's heard of"
async def create_gem_query(query, agent=hidden_gem_agent) -> GemQuery:
    result = await Runner.run(agent, query)
    return result.final_output  # -> GemQuery


#this converts the gem filters object into a dictionary so the search engine can take it in
def _to_filters(gem_filters: GemFilters):
    # apply_filters skips None/"" but not [], and genres/themes are list_contains
    # (one value per filter), so drop empties and take the first label.
    filters = {}
    for key, value in gem_filters.model_dump().items():
        if value is None or value == "" or value == []:
            continue
        filters[key] = value[0] if isinstance(value, list) else value
    return filters


async def recommend_hidden_gems(query, top_k=5, agent=hidden_gem_agent):
    """Request for something overlooked -> structured query -> ranked shows."""
    gem = await create_gem_query(query, agent)
    # Tier first, then the user's own asks on top: if they named a score they want,
    # that is a real request and it should win over the tier's floor.
    filters = {**TIERS[gem.obscurity], **_to_filters(gem.filters)}
    # Lean on metadata here. Obscure shows are exactly the ones with few or no
    # reviews, and search() already falls back to the metadata score alone for any
    # show that has none.
    results = search(
        gem.embedding_text,
        top_k=top_k,
        meta_weight=.65,
        review_weight=.35,
        filters=filters,
    )
    return gem, results


if __name__ == "__main__":
    import asyncio

    gem, results = asyncio.run(
        recommend_hidden_gems("something underrated and quiet, nothing mainstream")
    )
    print(f"{gem.what_they_want} [{gem.obscurity}]")
    print(gem.embedding_text)
    print(results[["title", "score", "members", "search_score"]])
