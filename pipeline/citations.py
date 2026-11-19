"""Check that the written answer only quotes numbers the researchers actually found.

Last of three anti-fabrication layers. Unsupported sentences are marked visibly, never deleted.
"""

import json
import re

# Ids are minted by research.py as F{round}.{brief}.{k}, so the pattern is exact rather than guessy.
CITATION = re.compile(r"\[(F\d+\.\d+\.\d+)\]")

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

# Not claims, so removed before counting: years, dates, window lengths ("26.2 days" still counts).
NOT_A_CLAIM = re.compile(
    r"\b(19|20)\d\d(-\d\d(-\d\d)?)?\b"
    r"|\b\d+(?:\.\d+)?-(?:day|week|month|year|session|sigma|leg)s?\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|S&P\s*\d+|\bNasdaq[-\s]?100\b|\bRussell\s*\d+\b|\bDow\s*30\b"  # index names, not data
    r"|\b(?:8|10)-[KQ]\b|\bForm\s+\d+\b",  # SEC form names
    re.I,
)

UNSUPPORTED = " [unsupported: more numbers stated than cited]"
MISMATCH = " [unsupported: does not match the finding it cites]"


def sentences(text: str) -> list[str]:
    """Split on sentence ends and line breaks, keeping bullets as their own units."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def claim_numbers(sentence: str) -> list[str]:
    """The numbers in `sentence` that assert something about the data."""
    # Models write "−2.04" with a Unicode minus; read as "2.04" it would match nothing.
    return NUMBER.findall(NOT_A_CLAIM.sub(" ", sentence.replace("\u2212", "-")))


def audit(answer: str, findings: dict[str, dict]) -> dict:
    """What is wrong with `answer`: `invalid` ids, `uncited` sentences (more numbers than
    citations) and `mismatched` numbers (in none of the findings they cite).
    """
    cited = set(CITATION.findall(answer))
    invalid = sorted(cited - set(findings))

    uncited, mismatched = [], []
    for sentence in sentences(answer):
        numbers = claim_numbers(sentence)
        ids = [i for i in CITATION.findall(sentence) if i in findings]
        if numbers and len(numbers) > len(CITATION.findall(sentence)):
            uncited.append(sentence)
        elif numbers and ids:
            pool = _values(" ".join(str(v) for i in ids for v in findings[i].values()))
            if not all(_matches(n, pool) for n in numbers):
                mismatched.append(sentence)

    return {
        "cited": sorted(cited & set(findings)),
        "invalid": invalid,
        "uncited": uncited,
        "mismatched": mismatched,
        "clean": not invalid and not uncited and not mismatched,
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
    for sentence in report.get("mismatched", [])[:6]:
        lines.append(
            f'- A number here is not in the finding it cites: "{sentence[:140]}". Use the value '
            f'exactly as the finding gives it, or cite the finding it came from.'
        )
    lines.append("- Do not add any number that is not already in the evidence list.")
    return "\n".join(lines)


def mark_unsupported(answer: str, report: dict) -> str:
    """Mark each sentence still stating an uncited number; used after the writer's one retry."""
    out = answer
    for found, marker in ((report["uncited"], UNSUPPORTED), (report.get("mismatched", []), MISMATCH)):
        for sentence in dict.fromkeys(found):
            # Never twice: a sentence already carrying the marker is skipped.
            out = re.sub(re.escape(sentence) + f"(?!{re.escape(marker)})",
                         lambda _, s=sentence, m=marker: s + m, out)
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


# --- The fast path: no findings store, so every number must appear in this run's tool results ---


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


def _values(text: str) -> set[float]:
    """Every claim number in `text`, as floats."""
    return {v for v in (_as_float(n) for n in claim_numbers(text)) if v is not None}


def _matches(raw: str, pool: set[float]) -> bool:
    """Whether `raw` equals some number in `pool` at the precision `raw` was written to.

    Compared by magnitude (prose drops signs); a percentage also matches its fraction.
    """
    value = _as_float(raw)
    if value is None:
        return True
    tolerance = 0.5 * 10 ** -_decimals(raw) + 1e-9
    targets = {abs(value)} | ({abs(value) / 100} if raw.strip().endswith("%") else set())
    return any(abs(abs(d) - t) <= tolerance for d in pool for t in targets)


def verify_against_tools(answer: str, tool_calls: list[dict], question: str = "") -> dict:
    """Which numbers in `answer` appear in no tool result from this run, at any rounding."""
    # The question's own numbers count as given ("last 6 months" supplies the 6).
    data = _values(" ".join([*(str(c.get("result", "")) for c in tool_calls), question]))
    missing = [raw.strip() for raw in claim_numbers(answer) if not _matches(raw, data)]
    return {"missing": sorted(set(missing))}


def sources_block(tool_calls: list[dict]) -> str:
    """The tools this answer rests on, one line each: the fast path's provenance."""
    rows, seen = [], set()
    for call in tool_calls:
        if call.get("name") in ("research", "write_report", "researcher", "clarify", "agent", "writer"):
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
    # Same heading as the research path, so the page renders both alike.
    return "\n".join(["", "**Evidence**", *rows]) if rows else ""
