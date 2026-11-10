"""Read-only SQL over the committed reference database.

This is the Text2SQL surface: an agent reads the schema, checks the real values of a column, then
writes its own SELECT. It is load-bearing rather than decorative, because resolving "the energy
sector" or "the leveraged Nasdaq family" to a list of symbols is the first step of every screen, and
doing it in SQL makes that step auditable - the query is shown alongside the answer.

The security boundary is the connection, not the prompt. `read_only` blocks writes,
`enable_external_access=False` stops a query reaching the filesystem or network through
`read_csv('http://...')`, and `lock_configuration=True` stops a query turning either back on. A
model may write any SELECT it likes; none of them can leave this file.

One connection for the process, a fresh cursor per call: DuckDB connections are not thread-safe but
cursors are cheap, and FastAPI serves requests on a thread pool.
"""

import functools
import json
import re
from pathlib import Path

import duckdb

import config

DB_PATH = Path(__file__).parent / "data" / "refdata.duckdb"
MAPPINGS_PATH = Path(__file__).parent / "data" / "refdata_mappings.json"

MAX_ROWS = 100  # a wide GROUP BY would otherwise flood an agent's own context

# Anything that is not a single read. Checked as whole words so a column named 'updated' is fine.
FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|install|load|export|import|create|insert|update|delete|drop|alter|"
    r"truncate|grant|revoke|pragma|set|call|read_csv|read_parquet|read_json|glob)\b",
    re.I,
)

# DuckDB's own catalog views. Harmless to read except that duckdb_databases() returns the database
# file's absolute path, which on this machine contains the owner's account name - so a model could
# put it in an answer. The schema card already describes every column, so nothing is lost.
INTERNAL_VIEWS = re.compile(r"\b(duckdb_databases|duckdb_settings|duckdb_extensions|sqlite_schema|"
                            r"pg_settings)\b", re.I)


class RefDataError(Exception):
    """A query or lookup an agent can fix by trying something else."""


@functools.cache
def _connection() -> duckdb.DuckDBPyConnection:
    """The one read-only, locked-down connection for this process."""
    if not DB_PATH.exists():
        raise RefDataError(f"{DB_PATH.name} is missing. Run `uv run build/build_refdata.py` to build it.")
    return duckdb.connect(
        str(DB_PATH),
        read_only=True,
        config={"enable_external_access": False, "lock_configuration": True},
    )


@functools.cache
def schema() -> dict:
    """The table's columns, descriptions and enum values, as the offline build recorded them."""
    if not MAPPINGS_PATH.exists():
        raise RefDataError(f"{MAPPINGS_PATH.name} is missing. Run `uv run build/build_refdata.py`.")
    return json.loads(MAPPINGS_PATH.read_text())


def enum_values(column: str) -> list[str]:
    """Every distinct value of `column`, so a filter can be written against real values.

    Guessing a sector name is the usual way a generated query returns nothing; this removes the
    guess. Reads the live table rather than the schema card, so it cannot go stale.
    """
    names = [c["name"] for c in schema()["columns"]]
    if column not in names:
        raise RefDataError(f"No column '{column}'. Columns are: {', '.join(names)}.")
    rows = _cursor().execute(
        f"SELECT DISTINCT {column} FROM instruments WHERE {column} IS NOT NULL ORDER BY 1"
    ).fetchall()
    return [row[0] for row in rows]


def _cursor() -> duckdb.DuckDBPyConnection:
    """A fresh cursor on the shared connection, which is how DuckDB wants concurrency handled."""
    return _connection().cursor()


def query(sql: str, limit: int = MAX_ROWS) -> dict:
    """Run one read-only SELECT and return {columns, rows, row_count, sql, truncated}.

    The returned `sql` is the statement actually executed, so a caller can cite it and a reader can
    re-run it. Refuses rather than sanitises: silently rewriting a model's query would make the
    citation a lie.
    """
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise RefDataError("The query is empty.")
    if ";" in text:
        raise RefDataError("Send one statement at a time, with no semicolons inside it.")
    if not re.match(r"^(select|with)\b", text, re.I):
        raise RefDataError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    if found := FORBIDDEN.search(text):
        raise RefDataError(f"'{found.group(0)}' is not allowed here; this connection is read-only.")
    if found := INTERNAL_VIEWS.search(text):
        raise RefDataError(
            f"'{found.group(0)}' is DuckDB's own catalog, not reference data. Call "
            f"get_reference_schema for the columns of 'instruments'."
        )

    capped = min(max(1, limit), MAX_ROWS)
    try:
        result = _cursor().execute(f"SELECT * FROM ({text}) LIMIT {capped + 1}")
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
    except duckdb.Error as e:
        # DuckDB's messages name the offending column and suggest near-matches, so they are worth
        # passing through verbatim rather than replacing with something vaguer.
        raise RefDataError(f"SQL error: {e}")
    truncated = len(rows) > capped
    return {
        "columns": columns,
        "rows": [list(r) for r in rows[:capped]],
        "row_count": len(rows[:capped]),
        "truncated": truncated,
        "sql": text,
        **({"note": f"Showing the first {capped} rows; add a tighter WHERE or an ORDER BY."} if truncated else {}),
    }


@functools.cache
def name_for(symbol: str) -> str | None:
    """The listed name of `symbol` from the reference table, or None if it is not in there.

    Table first, because the 625 instruments here are exactly the ones the agent screens, so a name
    shown in the tape matches the name used in an answer. Callers fall back to the listing directory
    for anything outside the table - indices, and any symbol a user asks about directly.
    """
    row = _cursor().execute("SELECT name FROM instruments WHERE symbol = ?", [symbol.strip().upper()]).fetchone()
    return row[0] if row else None


def resolve_universe(
    sector: str | None = None,
    etf_family: str | None = None,
    symbols: list[str] | None = None,
    limit: int = config.MAX_UNIVERSE,
) -> dict:
    """Turn a sector, an ETF family or an explicit list into the symbols to screen.

    Returns the symbols *and* the SQL that produced them, because the universe decides every pair
    that follows and a reader should be able to check it. Capped: screening every combination is
    quadratic, so 40 names is already 780 pairs.
    """
    capped = min(max(2, limit), config.MAX_UNIVERSE)
    if symbols:
        wanted = [s.strip().upper() for s in symbols if s and s.strip()]
        if len(wanted) < 2:
            raise RefDataError("Give at least two symbols to compare.")
        if len(wanted) > capped:
            # Capped here as well as in screen.screen: returning 200 symbols and failing downstream
            # tells the caller nothing about which limit it hit.
            raise RefDataError(
                f"{len(wanted)} symbols is over the {capped} cap "
                f"({len(wanted) * (len(wanted) - 1) // 2} pairs would be screened). Send fewer, or "
                f"pass a sector or etf_family and let the screen pick the universe."
            )
        quoted = ", ".join(f"'{_safe(s)}'" for s in wanted)
        sql = (f"SELECT symbol, name, gics_sector, etf_family FROM instruments "
               f"WHERE symbol IN ({quoted}) ORDER BY symbol")
        rows = _cursor().execute(sql).fetchall()
        found = {r[0] for r in rows}
        return _universe(rows, sql, [s for s in wanted if s not in found])

    if etf_family:
        sql = ("SELECT symbol, name, gics_sector, etf_family FROM instruments "
               f"WHERE etf_family = '{_safe(etf_family)}' ORDER BY leverage, symbol LIMIT {capped}")
    elif sector:
        # Match a GICS sector or a GICS sub-industry, so both the broad and the fine name work.
        value = _safe(sector)
        sql = ("SELECT symbol, name, gics_sector, etf_family FROM instruments "
               f"WHERE lower(gics_sector) = lower('{value}') OR lower(gics_sub_industry) = lower('{value}') "
               f"ORDER BY symbol LIMIT {capped}")
    else:
        raise RefDataError("Pass a sector, an etf_family, or an explicit list of symbols.")

    rows = _cursor().execute(sql).fetchall()
    if not rows:
        raise RefDataError(
            "Nothing matched. Call get_enum_values on 'gics_sector', 'gics_sub_industry' or "
            "'etf_family' to see the values that exist."
        )
    return _universe(rows, sql, [])


def _universe(rows: list[tuple], sql: str, missing: list[str]) -> dict:
    """Shape a resolved universe, noting any symbol that is not in the reference table."""
    out = {
        "symbols": [r[0] for r in rows],
        "count": len(rows),
        "instruments": [{"symbol": r[0], "name": r[1], "gics_sector": r[2], "etf_family": r[3]} for r in rows],
        "sql": sql,
        "pairs_to_screen": len(rows) * (len(rows) - 1) // 2,
    }
    if missing:
        out["not_found"] = missing
        out["note"] = (f"{', '.join(missing)} is not in the reference table, so it was dropped. "
                       "The universe is the S&P 500 plus curated ETFs.")
    return out


def _safe(value: str) -> str:
    """Escape a single-quoted SQL literal.

    These literals are interpolated so the returned `sql` is the complete, re-runnable statement a
    citation needs. The connection is read-only with external access locked off, so the blast radius
    of a bad literal is a syntax error, not a write.
    """
    return value.replace("'", "''")
