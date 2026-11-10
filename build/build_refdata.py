"""Offline build of the reference-data database. Run once; its output is committed.

    uv run build/build_refdata.py [--limit N]

Writes `data/refdata.duckdb` (one `instruments` table) and `data/refdata_mappings.json` (the schema
card the Text2SQL agent reads). Nothing here runs at request time.

Why offline: symbols, names, sectors, ETF families and share classes change on the scale of months,
so resolving them live would spend a research agent's budget re-deriving constants. Everything that
*can* change day to day - prices, correlations, betas, half-lives, backtests - is computed live and
is never stored here. The output is committed because continuous deploy ships only committed files.

Universe (~640 names, approved 2026-11-09): the S&P 500 plus a curated list of major ETFs. That
covers every Dow 30 name and nearly every Nasdaq-100 name, and the ETF list carries the index funds
and their leveraged families explicitly, which is what "an ETF family" queries screen over.

Sector comes from GICS, published in the Wikipedia constituents table alongside the name and CIK.
An earlier version read the SEC's SIC code per company instead; SEC rate-limits per IP and hard, so
that crawl resolved 12 of 503 names and had to be retried for hours. GICS covers all of them in the
one page fetch, in the vocabulary traders use, so the build no longer touches a rate-limited feed
and finishes in seconds.
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import requests

# This script lives in build/ but reads the project's fetchers from the repo root, which Python does
# not put on sys.path when a script is run by path. Added before the local imports so that
# `uv run build/build_refdata.py` works from a clean checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data  # noqa: E402 - must follow the sys.path line above
import wikitables  # noqa: E402 - sibling module inside build/

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = OUT_DIR / "refdata.duckdb"
MAPPINGS_PATH = OUT_DIR / "refdata_mappings.json"

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Curated ETF universe, grouped into the families a pair screen actually wants to compare within.
# (symbol, leverage, underlying): leverage is the stated daily multiple and `underlying` the fund it
# tracks, which is what makes "leveraged vs underlying" screenable. 1.0/None means a plain fund.
ETF_FAMILIES: dict[str, list[tuple[str, float, str | None]]] = {
    "broad_us": [("SPY", 1, None), ("VOO", 1, None), ("IVV", 1, None), ("SPLG", 1, None),
                 ("VTI", 1, None), ("ITOT", 1, None), ("QQQ", 1, None), ("QQQM", 1, None),
                 ("DIA", 1, None), ("IWM", 1, None), ("IWB", 1, None), ("VT", 1, None)],
    "sp500_leveraged": [("SSO", 2, "SPY"), ("UPRO", 3, "SPY"), ("SPXL", 3, "SPY"), ("SPUU", 2, "SPY"),
                        ("SH", -1, "SPY"), ("SDS", -2, "SPY"), ("SPXU", -3, "SPY"), ("SPXS", -3, "SPY")],
    "nasdaq_leveraged": [("QLD", 2, "QQQ"), ("TQQQ", 3, "QQQ"), ("PSQ", -1, "QQQ"),
                         ("QID", -2, "QQQ"), ("SQQQ", -3, "QQQ")],
    "dow_leveraged": [("DDM", 2, "DIA"), ("UDOW", 3, "DIA"), ("DOG", -1, "DIA"), ("SDOW", -3, "DIA")],
    "russell_leveraged": [("UWM", 2, "IWM"), ("TNA", 3, "IWM"), ("RWM", -1, "IWM"), ("TZA", -3, "IWM")],
    "sector_spdr": [("XLE", 1, None), ("XLF", 1, None), ("XLK", 1, None), ("XLV", 1, None),
                    ("XLI", 1, None), ("XLY", 1, None), ("XLP", 1, None), ("XLU", 1, None),
                    ("XLB", 1, None), ("XLRE", 1, None), ("XLC", 1, None)],
    "vanguard_sector": [("VGT", 1, None), ("VHT", 1, None), ("VFH", 1, None), ("VDE", 1, None),
                        ("VIS", 1, None), ("VDC", 1, None), ("VPU", 1, None), ("VAW", 1, None),
                        ("VOX", 1, None), ("VNQ", 1, None), ("VCR", 1, None)],
    "equal_weight": [("RSP", 1, "SPY"), ("QQQE", 1, "QQQ"), ("EQAL", 1, None)],
    "style": [("IVW", 1, None), ("IVE", 1, None), ("VUG", 1, None), ("VTV", 1, None),
              ("MDY", 1, None), ("IJH", 1, None), ("IJR", 1, None), ("VB", 1, None), ("VO", 1, None)],
    "factor": [("MTUM", 1, None), ("QUAL", 1, None), ("USMV", 1, None), ("VLUE", 1, None),
               ("SPLV", 1, None), ("SPHB", 1, None), ("SPHD", 1, None)],
    "bond": [("TLT", 1, None), ("IEF", 1, None), ("SHY", 1, None), ("SHV", 1, None), ("BIL", 1, None),
             ("AGG", 1, None), ("BND", 1, None), ("LQD", 1, None), ("HYG", 1, None), ("JNK", 1, None),
             ("TIP", 1, None), ("MBB", 1, None)],
    "commodity": [("GLD", 1, None), ("IAU", 1, None), ("SLV", 1, None), ("USO", 1, None),
                  ("UNG", 1, None), ("DBC", 1, None), ("GDX", 1, None), ("GDXJ", 1, None)],
    "international": [("EFA", 1, None), ("VEA", 1, None), ("IEFA", 1, None), ("EEM", 1, None),
                      ("VWO", 1, None), ("IEMG", 1, None), ("FXI", 1, None), ("EWJ", 1, None),
                      ("EWZ", 1, None), ("INDA", 1, None), ("EWY", 1, None), ("EWT", 1, None)],
    "volatility": [("VXX", 1, None), ("VIXY", 1, None), ("UVXY", 1.5, "VXX"), ("SVXY", -0.5, "VXX")],
    "thematic": [("SMH", 1, None), ("SOXX", 1, None), ("XBI", 1, None), ("IBB", 1, None),
                 ("KRE", 1, None), ("XOP", 1, None), ("XME", 1, None), ("ITB", 1, None),
                 ("JETS", 1, None), ("TAN", 1, None), ("ICLN", 1, None), ("ARKK", 1, None)],
}

# One row per instrument. `symbol` is the Cboe spelling and `sec_symbol` the SEC one; they differ for
# multi-class names (BRK.B vs BRK-B) and every join needs both.
COLUMNS = [
    ("symbol", "VARCHAR", "Cboe ticker, the spelling every price tool takes. E.g. 'NVDA', 'BRK.B'."),
    ("sec_symbol", "VARCHAR", "SEC ticker spelling, which uses '-' for share classes. E.g. 'BRK-B'."),
    ("name", "VARCHAR", "Company or fund name from the listing directory. Rarely NULL when that directory omits a fund; the symbol still trades."),
    ("type", "VARCHAR", "'stock' or 'etf'."),
    ("is_etf", "BOOLEAN", "True for an exchange-traded fund, false for an operating company."),
    ("in_sp500", "BOOLEAN", "True if the name is an S&P 500 constituent."),
    ("cik", "BIGINT", "SEC Central Index Key. NULL for most ETFs, which are not SEC filers."),
    ("gics_sector", "VARCHAR", "GICS sector, the standard equity classification. Prefer this for resolving a sector to its symbols."),
    ("gics_sub_industry", "VARCHAR", "GICS sub-industry, a much finer grouping. Best for finding close substitutes to pair."),
    ("etf_family", "VARCHAR", "Which curated ETF family the fund belongs to. NULL for stocks."),
    ("leverage", "DOUBLE", "Stated daily multiple of the underlying: 3.0 for TQQQ, -1.0 for SH, 1.0 for a plain fund."),
    ("underlying", "VARCHAR", "The fund this one tracks a multiple of, when there is one. E.g. TQQQ -> QQQ."),
]


def sp500_records() -> list[dict[str, str]]:
    """One record per S&P 500 constituent, straight from the Wikipedia constituents table.

    That table also carries GICS sector, GICS sub-industry, the company name and the CIK, so it is
    the better source for all four: GICS is the classification equity traders actually use, and its
    CIK column covers recent spin-offs that the SEC ticker file has not caught up with.
    """
    html = requests.get(SP500_URL, headers={"User-Agent": data._UA}, timeout=30).text
    return wikitables.records(html, "symbol", 450)


def etf_rows() -> list[dict]:
    """One row per curated ETF, with its family, leverage and underlying."""
    rows = {}
    for family, members in ETF_FAMILIES.items():
        for symbol, leverage, underlying in members:
            # First family wins, so a fund listed twice keeps its most specific grouping.
            rows.setdefault(symbol, {
                "symbol": symbol, "sec_symbol": symbol.replace(".", "-"), "name": None,
                "type": "etf", "is_etf": True, "in_sp500": False, "cik": None,
                "gics_sector": "Exchange-Traded Fund",
                "gics_sub_industry": family, "etf_family": family,
                "leverage": float(leverage), "underlying": underlying,
            })
    return list(rows.values())


def stock_rows(records: list[dict[str, str]]) -> list[dict]:
    """One row per S&P 500 company: GICS and name from Wikipedia, CIK from the SEC ticker file."""
    ciks = data._sec_ciks()
    rows = []
    for rec in sorted(records, key=lambda r: r["symbol"]):
        symbol = rec["symbol"]
        sec_symbol = symbol.replace(".", "-")
        # Prefer the SEC's own mapping; fall back to Wikipedia's CIK column for very recent listings.
        cik = ciks.get(sec_symbol) or ciks.get(symbol)
        if not cik and rec.get("cik", "").isdigit():
            cik = int(rec["cik"])
        rows.append({
            "symbol": symbol, "sec_symbol": sec_symbol, "name": rec.get("security") or None,
            "type": "stock", "is_etf": False, "in_sp500": True, "cik": cik,
            "gics_sector": rec.get("gics_sector") or None,
            "gics_sub_industry": rec.get("gics_sub_industry") or None,
            "etf_family": None, "leverage": 1.0, "underlying": None,
        })
    return rows


def add_names(rows: list[dict]) -> None:
    """Fill any name still missing from the Nasdaq Trader listing directory, which covers ETFs."""
    directory = {sym: name for sym, name, _ in data._symbol_directory()}
    for row in rows:
        if not row["name"]:
            row["name"] = directory.get(row["symbol"]) or directory.get(row["sec_symbol"])
    print(f"  named {sum(1 for r in rows if r['name'])}/{len(rows)}")


def write_db(rows: list[dict]) -> None:
    """Replace `instruments` with `rows`. Rebuilt whole, never patched, so the file is reproducible."""
    OUT_DIR.mkdir(exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"CREATE TABLE instruments ({', '.join(f'{n} {t}' for n, t, _ in COLUMNS)})")
    names = [n for n, _, _ in COLUMNS]
    con.executemany(
        f"INSERT INTO instruments VALUES ({', '.join('?' * len(names))})",
        [[r[n] for n in names] for r in rows],
    )
    con.execute("CREATE INDEX idx_symbol ON instruments(symbol)")
    con.close()


def write_mappings(rows: list[dict]) -> None:
    """The schema card the Text2SQL agent reads: columns, descriptions, and the real enum values.

    Enums come from the built rows rather than being typed out, so they cannot drift from the data.
    """
    def values(column: str) -> list[str]:
        return sorted({r[column] for r in rows if r[column] is not None})

    MAPPINGS_PATH.write_text(json.dumps({
        "table": "instruments",
        "row_count": len(rows),
        "description": (
            "One row per instrument in the screening universe: S&P 500 companies and curated ETFs. "
            "Static reference data only - no prices, correlations or statistics live here."
        ),
        "columns": [
            {"name": n, "type": t, "description": d,
             **({"enum_values": values(n)} if n in ("type", "gics_sector", "etf_family") else {})}
            for n, t, d in COLUMNS
        ],
    }, indent=2) + "\n")


def build(limit: int | None = None) -> None:
    """Build the whole database, printing each step."""
    print("1/4 S&P 500 constituents, GICS sectors and CIKs from Wikipedia...", flush=True)
    records = sp500_records()
    if limit:
        records = sorted(records, key=lambda r: r["symbol"])[:limit]
    print(f"  {len(records)} constituents, {len({r['gics_sector'] for r in records})} GICS sectors")

    print("2/4 curated ETF families...", flush=True)
    rows = stock_rows(records) + etf_rows()
    print(f"  {len(rows)} instruments total ({sum(r['is_etf'] for r in rows)} ETFs)")

    print("3/4 names...", flush=True)
    add_names(rows)

    print("4/4 writing duckdb + mappings...", flush=True)
    write_db(rows)
    write_mappings(rows)
    print(f"  {DB_PATH} ({DB_PATH.stat().st_size // 1024} KB)")
    print(f"  {MAPPINGS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="only this many S&P names, for a quick smoke test")
    build(ap.parse_args().limit)
