"""Check, in Python, that the written answer only quotes numbers the researchers actually found.

Three layers stop fabrication, and this is the last one. The first is that a finding must name the
tool that produced it. The second is that the writer has no tools at all, so it cannot reach the
data and can only work from the findings it was handed. This layer catches what is left: a citation
pointing at an id that does not exist, and a sentence stating a number with no citation at all.

Nothing is silently deleted. An unsupported sentence keeps its text and gains a visible marker,
because a reader who can see the gap is better served than one shown a confident answer with the
doubtful parts quietly removed.
"""

import re

# Ids are minted by research.py as F{round}.{brief}.{k}, so the pattern is exact rather than guessy.
CITATION = re.compile(r"\[(F\d+\.\d+\.\d+)\]")

# Any number a sentence asserts. `\d+` rather than `\d{1,3}`: the three-digit version silently
# skipped every four-digit figure, so a fabricated "1095" or "2500" was never even looked at.
# The lookarounds keep it off the digits inside a citation id like [F1.2.3].
NUMBER = re.compile(
    r"""(?<![\w.])            # not mid-word and not the tail of a longer number
        [+-]?\d+              # the digits
        (?:,\d{3})*           # thousands separators
        (?:\.\d+)?            # decimals
        \s?%?                 # an optional percent sign
        (?![\w.])""",
    re.X,
)

# Numbers that are not claims about the data, removed before counting:
#   - calendar years and ISO dates
#   - hyphenated window parameters: "1095-day lookback", "3-year window". The hyphen is the tell
#     that it modifies a noun rather than measuring something. "26.2 days" keeps its space and so
#     stays a claim, which is right - that one is a measured half-life.
NOT_A_CLAIM = re.compile(
    r"\b(19|20)\d\d(-\d\d(-\d\d)?)?\b"
    r"|\b\d+(?:\.\d+)?-(?:day|week|month|year|session|sigma|leg)s?\b",
    re.I,
)

UNSUPPORTED = " _[unsupported: more numbers stated than cited]_"


def sentences(text: str) -> list[str]:
    """Split on sentence ends and line breaks, keeping bullets as their own units."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def claim_numbers(sentence: str) -> list[str]:
    """The numbers in `sentence` that assert something about the data.

    Calendar years and dates are removed first: "the divergence began in 2024" is not a measurement
    and demanding a citation for it would train the writer to cite noise.
    """
    return NUMBER.findall(NOT_A_CLAIM.sub(" ", sentence))


def audit(answer: str, findings: dict[str, dict]) -> dict:
    """What is wrong with `answer`, judged only against the findings that exist.

    Returns `invalid` (cited ids that were never minted), `uncited` (sentences stating more numbers
    than they cite), and `unused` (findings the writer ignored, a coverage signal rather than an
    error).

    Counting numbers against citations, rather than asking whether a sentence has any citation at
    all, is the difference between a check and a formality: a single marker would otherwise license
    every other number in the same sentence. The first version of this missed a fabricated deflated
    Sharpe sitting beside a correctly cited one.
    """
    cited = set(CITATION.findall(answer))
    invalid = sorted(cited - set(findings))

    uncited = []
    for sentence in sentences(answer):
        numbers = claim_numbers(sentence)
        if numbers and len(numbers) > len(CITATION.findall(sentence)):
            uncited.append(sentence)

    return {
        "cited": sorted(cited & set(findings)),
        "invalid": invalid,
        "uncited": uncited,
        "unused": sorted(set(findings) - cited),
        "clean": not invalid and not uncited,
    }


def feedback(report: dict) -> str:
    """The retry instruction for the writer, naming exactly what to fix."""
    lines = ["Your draft did not pass the citation check. Fix only these points and rewrite:"]
    if report["invalid"]:
        lines.append(
            f"- These ids do not exist: {', '.join(report['invalid'])}. Cite only ids from the "
            f"evidence list, exactly as given."
        )
    for sentence in report["uncited"][:6]:
        lines.append(
            f'- This states more numbers than it cites, so at least one is unsupported: '
            f'"{sentence[:140]}". Cite every number, or remove the ones you cannot cite.'
        )
    lines.append("- Do not add any number that is not already in the evidence list.")
    return "\n".join(lines)


def mark_unsupported(answer: str, report: dict) -> str:
    """Append a visible marker to each sentence that still states an uncited number.

    Used only after the writer has had its one retry. Marking beats deleting: the reader sees the
    claim and that it is unbacked, rather than a tidy answer with a hole in it.
    """
    out = answer
    for sentence in report["uncited"]:
        if sentence in out and UNSUPPORTED not in sentence:
            out = out.replace(sentence, sentence + UNSUPPORTED, 1)
    for bad in report["invalid"]:
        out = out.replace(f"[{bad}]", f"[{bad} - no such finding]")
    return out


def evidence_block(findings: dict[str, dict], cited: list[str]) -> str:
    """The evidence list printed under the answer, one row per finding the answer used."""
    if not cited:
        return ""
    rows = ["", "**Evidence**"]
    for fid in cited:
        f = findings[fid]
        where = f" {f['pair']}" if f.get("pair") else ""
        limit = f" - {f['limitation']}" if f.get("limitation") else ""
        rows.append(f"- [{fid}]{where} {f['metric']} = {f['value']} (via {f['tool']}){limit}")
    return "\n".join(rows)
