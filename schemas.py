"""The structured shapes the agents hand each other.

Pydantic rather than dataclasses because the SDK turns these into the JSON schema it gives the
model as `output_type`. Every field a model fills is documented in its own description: that text
is the only instruction the model gets about the field, so it is written for the model, not a
reader of this file.

Ids are deliberately absent from `Finding`: Python mints them in `research.py` after a researcher
returns. A model asked to invent unique ids across parallel runs will collide, and a citation
system whose keys collide is worse than none.
"""

from pydantic import BaseModel, Field


class Clarification(BaseModel):
    """The gate's verdict on whether a question can be researched as asked."""

    ready: bool = Field(description="True if the question can be researched as written, even if you had to assume something reasonable. False only if you genuinely cannot tell what to screen.")
    simple: bool = Field(default=False, description="True if one analyst with the tools can answer this directly: a chart, a price history, news or filings for one symbol, a single named pair, or a lookup in the reference table. False for anything needing several lines of enquiry compared or combined.")
    out_of_scope: bool = Field(default=False, description="True only if the question asks for an asset class or data the desk has no tool for at all. Then `question` holds the sentence that says so.")
    question: str = Field(default="", description="If not ready: the ONE question to ask back, in a single sentence. If out_of_scope: the sentence stating what is unavailable and what is. Empty when ready.")
    restated: str = Field(default="", description="If ready: the task restated in one sentence, naming the universe to screen. Empty when not ready.")
    assumptions: list[str] = Field(default_factory=list, description="If ready: any defaults you applied, e.g. '3-year lookback'. At most three, each a short phrase.")


class Finding(BaseModel):
    """One checkable claim, tied to the tool result it came from."""

    claim: str = Field(description="One sentence, stating what was found. No hedging, no advice.")
    metric: str = Field(description="What was measured, e.g. 'deflated Sharpe' or 'half-life'.")
    value: str = Field(description="The number as the tool reported it, with units, e.g. '-2.04' or '20.8 sessions'.")
    pair: str = Field(default="", description="The pair or symbol this is about, e.g. 'DVN/SLB'. Empty if it is about the whole universe.")
    tool: str = Field(description="Which tool produced it: screen_pairs, analyze_pair, research_divergence or query_reference.")
    limitation: str = Field(default="", description="What stops this being conclusive, if anything. One clause.")


class ResearcherReport(BaseModel):
    """What one researcher hands back from one brief."""

    findings: list[Finding] = Field(description="Every checkable claim, each from a tool result you actually received. Empty if the tools returned nothing usable.")
    summary: str = Field(description="Two sentences at most: what you found, in plain words.")
    gap: str = Field(default="", description="What you could not establish and why, if anything. Empty if the brief was fully answered.")


class Report(BaseModel):
    """The writer's answer, assembled only from findings it was given."""

    answer: str = Field(description="The answer itself. Plain text, '-' bullets, **bold** numbers, no headings or tables. Cite every number as [F1], [F2] matching the evidence ids you were given.")
    caveats: str = Field(description="What the reader should not conclude from this: sample size, costs, what the data excludes.")
