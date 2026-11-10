"""Check, in Python, that the written answer only quotes numbers the researchers actually found.

Three layers stop fabrication and this is the last. First, a finding must name the tool that
produced it. Second, the writer has no tools, so it can only work from the findings it was handed.
This layer catches what is left: a citation pointing at an id that does not exist, and a sentence
stating a number with no citation.

Nothing is silently deleted. An unsupported sentence keeps its text and gains a visible marker,
because a reader who can see the gap is better served than one shown a confident answer with the
doubtful parts quietly removed.
"""

import json
import re

# Ids are minted by research.py as F{round}.{brief}.{k}, so the pattern is exact rather than guessy.
CITATION = re.compile(r"\[(F\d+\.\d+\.\d+)\]")

# Any number a sentence asserts.
NUMBER = re.compile(
    r"""(?<![\w.])            # not mid-word, and not the tail of a longer number
        [+-]?\d+              # the digits
        (?:,\d{3})*           # thousands separators
        (?:\.\d+)?            # decimals
        \s?%?                 # an optional percent sign
        (?!\w)                # not mid-word...
        (?!\.\d)""",           # ...and not the head of a longer number, though a sentence-ending '.' is fine
    re.X,
)

# Not claims about the data, so removed before counting: calendar years, ISO and written-out dates,
# and hyphenated window parameters. The hyphen is the tell that a number modifies a noun rather than
# measuring something, so "26.2 days" keeps its space and stays a claim.
NOT_A_CLAIM = re.compile(
    r"\b(19|20)\d\d(-\d\d(-\d\d)?)?\b"
    r"|\b\d+(?:\.\d+)?-(?:day|week|month|year|session|sigma|leg)s?\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.I,
)

UNSUPPORTED = " _[unsupported: more numbers stated than cited]_"


def sentences(text: str) -> list[str]:
    """Split on sentence ends and line breaks, keeping bullets as their own units."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def claim_numbers(sentence: str) -> list[str]:
    """The numbers in `sentence` that assert something about the data."""
    return NUMBER.findall(NOT_A_CLAIM.sub(" ", sentence))


def audit(answer: str, findings: dict[str, dict]) -> dict:
    """What is wrong with `answer`, judged only against the findings that exist.

    Returns `invalid` (cited ids never minted) and `uncited` (sentences stating more numbers than
    they cite). Counting numbers against citations is what makes this a check rather than a
    formality: one marker would otherwise license every other number in the same sentence.
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
        # "via analyze_pair" says who computed it; the source says what it was computed from.
        via = f" (via {f['tool']}" + (f", from {f['source']}" if f.get("source") else "") + ")"
        limit = f" - {f['limitation']}" if f.get("limitation") else ""
        rows.append(f"- [{fid}]{where} {f['metric']} = {f['value']}{via}{limit}")
    return "\n".join(rows)


# --- The fast path -------------------------------------------------------------------------------
#
# One analyst answers a single-pair question in seconds, so that path has no findings store and no
# [F...] ids. The check available instead is stronger than a citation marker: every number must
# appear in a tool result from this run, which no plausible-looking id can satisfy.


def _as_float(raw: str) -> float | None:
    """A claim number as a float, or None if it will not parse."""
    try:
        return float(raw.strip().rstrip("%").replace(",", "").lstrip("+"))
    except ValueError:
        return None


def _decimals(raw: str) -> int:
    """How many decimal places the claim was written to, which is the precision to compare at."""
    body = raw.strip().rstrip("%")
    return len(body.split(".")[1]) if "." in body else 0


def verify_against_tools(answer: str, tool_calls: list[dict], question: str = "") -> dict:
    """Which numbers in `answer` are not present in any tool result from this run.

    Compared numerically at the precision the answer used, not as strings: a model writing 0.35 for
    a reported 0.348 has rounded real data, and a false "unverified" costs more trust than it
    protects. What it catches is a number with no counterpart at any rounding.
    """
    # The question's own numbers count as given: "chart AAPL over the last 6 months" supplies the 6,
    # and flagging it would report the user's own parameter as an unsourced claim.
    haystack = " ".join([*(str(c.get("result", "")) for c in tool_calls), question])
    data = {v for v in (_as_float(n) for n in claim_numbers(haystack)) if v is not None}
    missing = []
    for raw in claim_numbers(answer):
        value = _as_float(raw)
        if value is None:
            continue
        places = _decimals(raw)
        if not any(round(d, places) == round(value, places) for d in data):
            missing.append(raw.strip())
    return {"missing": sorted(set(missing))}


def sources_block(tool_calls: list[dict]) -> str:
    """The tools this answer rests on, one line each, for the fast path.

    Provenance rather than per-number citation: it names each call and its arguments, so a reader
    can re-run it. Weaker than the research path's `[F...]` ids, and all one analyst can offer.
    Emitted under the same **Evidence** heading so the page presents both paths identically.
    """
    rows, seen = [], set()
    for call in tool_calls:
        if call.get("name") in ("research", "write_report", "researcher"):
            continue
        args = ", ".join(f"{k}={v}" for k, v in (call.get("args") or {}).items() if v not in (None, "", []))
        line = f"{call.get('name')}({args})"
        if line in seen:
            continue
        seen.add(line)
        source = ""
        try:
            got = json.loads(str(call.get("result", "")))
            if isinstance(got, dict) and got.get("source"):
                source = f" - {got['source']}" + (f", {got['delay']}" if got.get("delay") else "")
        except (TypeError, ValueError):
            pass
        rows.append(f"- {line}{source}")
    # Same marker the research path uses, so the page collapses both into one control rather
    # than printing one as a list and the other as loose text.
    return "\n".join(["", "**Evidence**", *rows]) if rows else ""
