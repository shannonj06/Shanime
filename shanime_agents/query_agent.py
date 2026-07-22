from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from agents import Agent, Runner

# Load the filter vocabulary straight from the embedded metadata so the values the
# agent is allowed to emit exactly match what search_engine.apply_filters checks.
PATH = Path(__file__).resolve().parent.parent / "Datasets"
_meta = pd.read_pickle(PATH / "meta_embeddings.pkl")


def _list_values(column):
    return sorted(
        {v for lst in _meta[column].dropna() if isinstance(lst, list) for v in lst if v}
    )


GENRES = _list_values("genres")
THEMES = _list_values("themes")
MEDIA_TYPES = sorted(_meta["media_type"].dropna().unique().tolist())
MIN_YEAR = int(_meta["year"].dropna().min())
MAX_YEAR = int(_meta["year"].dropna().max())


INSTRUCTIONS = f"""You convert a user's natural-language anime request into a structured \
query for a semantic search engine.

Produce:
1. embedding_text: a rich, descriptive sentence capturing the plot, themes, and mood the \
user wants. This text is embedded and matched against anime descriptions, so make it \
specific and evocative. Do NOT put years, scores, or filter words in here.
2. filters: only set a filter when the user clearly asks for it; otherwise leave it \
null (for single values) or empty (for lists).

You MUST only use values from these allowed lists. Pick the closest match, or leave the \
field empty if nothing fits. Never invent genre or theme labels.
- media_type: {MEDIA_TYPES} (leave null to search anime by default)
- genres: {GENRES}
- themes: {THEMES}
- year_min / year_max: integers between {MIN_YEAR} and {MAX_YEAR}
- score_min / score_max: floats between 0 and 10
"""

query_agent = Agent(
    name="query_agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
)

async def create_query_agent(query, agent=query_agent):
    result = await Runner.run(agent, query)
    return result.final_output  # -> StructuredQuery

if __name__ == "__main__":
    print(create_query_agent("I want an anime that will make me feel something im turning 20 i want something that could resonate with my life", query_agent))