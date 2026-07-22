from typing import List

import pandas as pd
from pydantic import BaseModel
from agents import Agent, Runner

# How much of a synopsis to hand the agent per show. Long enough to explain the
# fit, short enough that 5 shows stay cheap.
SYNOPSIS_CHARS = 600


class ShowExplanation(BaseModel):
    title: str      # copied verbatim from the candidate list
    hook: str       # one line, the single reason this one is here
    why: str        # 2-3 sentences tying the show back to what the user asked for


class Explanations(BaseModel):
    explanations: List[ShowExplanation]


INSTRUCTIONS = """You explain why an anime recommendation engine returned the shows it \
returned.

You are given the user's original request and a numbered list of shows the search engine \
already picked, each with its year, score, genres, themes, and synopsis. Write one \
explanation per show, in the order given.

For each show produce:
1. title: copied exactly as it appears in the candidate list. Never rename, translate, or \
shorten it.
2. hook: one line -- the single strongest reason this show answers the request.
3. why: 2-3 sentences connecting this specific show to what the user actually asked for. \
Name the concrete thing in the show that does the work -- a character's arc, the way the \
story treats its subject, the pacing, the tone it leaves you with -- and tie it back to \
their words.

Ground every claim in the synopsis, genres, and themes you were given. If a show is a \
loose fit, say so plainly and name what it does share instead of overselling it -- a \
hedged honest explanation is more useful than a confident wrong one. Never invent plot \
points, characters, or endings that are not supported by the material you were given.

Do NOT recommend, mention, or compare against any show that is not in the candidate list, \
and never reorder or drop one. Explaining the list you were handed is the whole job.

Do not spoil endings or late twists, even when the synopsis gives them away.

Requests often involve depression, self-harm, addiction, abuse, or suicide. Describe how a \
story portrays those subjects in neutral, non-romanticizing language. Never diagnose, and \
never assume someone asking for a dark theme is describing their own life.
"""

explanation_agent = Agent(
    name="explanation_agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Explanations,
)


def _format_shows(results):
    """Render search() output as the numbered candidate list the prompt describes."""
    rows = results.to_dict("records") if isinstance(results, pd.DataFrame) else list(results)

    blocks = []
    for i, row in enumerate(rows, 1):
        # missing cells come through as NaN (a truthy float), not None or [],
        # so type-check rather than relying on falsiness
        synopsis = row.get("synopsis").strip() if isinstance(row.get("synopsis"), str) else ""
        if len(synopsis) > SYNOPSIS_CHARS:
            synopsis = synopsis[:SYNOPSIS_CHARS].rsplit(" ", 1)[0] + "..."
        genres = row.get("genres") if isinstance(row.get("genres"), list) else []
        themes = row.get("themes") if isinstance(row.get("themes"), list) else []
        blocks.append(
            f"{i}. {row.get('title')}\n"
            f"   year: {row.get('year')} | score: {row.get('score')}\n"
            f"   genres: {', '.join(genres) or 'none listed'}\n"
            f"   themes: {', '.join(themes) or 'none listed'}\n"
            f"   synopsis: {synopsis or 'none listed'}"
        )
    return "\n\n".join(blocks)


# request is what the user originally asked for; results is the frame search() returned
async def explain_results(request, results, agent=explanation_agent) -> Explanations:
    if (results.empty if isinstance(results, pd.DataFrame) else not len(results)):
        return Explanations(explanations=[])

    prompt = (
        f"The user asked for:\n{request}\n\n"
        f"The search engine returned these shows:\n\n{_format_shows(results)}"
    )
    result = await Runner.run(agent, prompt)
    return result.final_output  # -> Explanations


if __name__ == "__main__":
    import asyncio

    from shanime_agents.theme_agent import recommend_by_theme

    async def demo():
        request = "anime about grief where someone learns to move on"
        _theme, results = await recommend_by_theme(request)
        return await explain_results(request, results)

    for item in asyncio.run(demo()).explanations:
        print(f"{item.title} -- {item.hook}")
        print(item.why)
        print()
