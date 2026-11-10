"""The structured shapes the agents hand each other.

Pydantic rather than dataclasses because the SDK turns these into the JSON schema it gives the
model as `output_type`. **Every `description` below is written for the model, not for a reader of
this file** - it is the only instruction the model gets about that field.

Ids are deliberately absent from `Finding`: Python mints them in `research.py` after a researcher
returns. A model asked to invent unique ids across parallel runs will collide, and a citation
system whose keys collide is worse than none.
"""

from pydantic import BaseModel, Field


class Clarification(BaseModel):
    """The gate's verdict on whether a question can be researched as asked."""

    ready: bool = Field(description="True if the question can be researched as written, even if you had to assume something reasonable. False only if you genuinely cannot tell what to screen.")
    simple: bool = Field(default=False, description="True if one analyst with the tools can answer this directly; see SIMPLE vs RESEARCH in your instructions.")
    out_of_scope: bool = Field(default=False, description="True only if the question asks for an asset class or data the desk has no tool for at all. Then `question` holds the sentence that says so.")
    question: str = Field(default="", description="If not ready: the ONE question to ask back, in a single sentence. If out_of_scope: one sentence naming what is unavailable and what the desk can do instead. Empty when ready.")


class Finding(BaseModel):
    """One checkable claim, tied to the tool result it came from."""

    claim: str = Field(description="One sentence, stating what was found. No hedging, no advice.")
    metric: str = Field(description="What was measured, e.g. 'deflated Sharpe' or 'half-life'.")
    value: str = Field(description="The number as the tool reported it, with units, e.g. '-2.04' or '20.8 sessions'.")
    pair: str = Field(default="", description="The pair or symbol this is about, e.g. 'DVN/SLB'. Empty if it is about the whole universe.")
    tool: str = Field(description="The exact name of the tool that produced it, as you called it. Any of the tools you hold, not just the screening ones.")
    limitation: str = Field(default="", description="What stops this being conclusive, if anything. One clause.")


class ResearcherReport(BaseModel):
    """What one researcher hands back from one brief."""

    findings: list[Finding] = Field(description="Every checkable claim, each from a tool result you actually received. Empty if the tools returned nothing usable.")
    summary: str = Field(description="Two sentences at most: what you found, in plain words.")
    gap: str = Field(default="", description="What you could not establish and why, if anything. Empty if the brief was fully answered.")


class Report(BaseModel):
    """The writer's answer, assembled only from findings it was given."""

    answer: str = Field(description="The answer itself. Plain text, '-' bullets, **bold** numbers, no headings or tables. Cite every number with the exact id it came from, in square brackets, like [F1.2.3] - the ids have three parts and must be copied exactly as given. Repeat the id after every number, even when several numbers share one id.")
    caveats: str = Field(description="What the reader should not conclude from this: thin samples, costs, what the data excludes. Prose; these are citation-checked like the answer, so state no number here unless you cite its id.")
