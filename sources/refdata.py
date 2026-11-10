"""Read-only SQL over the committed reference database.

The Text2SQL surface: an agent reads the schema, checks a column's real values, then writes its own
SELECT. Load-bearing rather than decorative, because resolving "the energy sector" to a list of
symbols is the first step of every screen, and doing it in SQL makes that step auditable - the
query is shown beside the answer.

**The security boundary is the connection, not the prompt.** `read_only` blocks writes,
`enable_external_access=False` stops a query reaching the filesystem or network through
`read_csv('http://...')`, and `lock_configuration=True` stops a query turning either back on. A
model may write any SELECT it likes; none can leave this file.

One connection per process, a fresh cursor per call: DuckDB connections are not thread-safe but
cursors are cheap, and FastAPI serves requests on a thread pool.
"""

import functools
import json
import re
from pathlib import Path

import duckdb

import config

# .parent.parent, not .parent: this module lives in sources/ and the database sits at the repo root.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "refdata.duckdb"
MAPPINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "refdata_mappings.json"

MAX_ROWS = 100  # a wide GROUP BY would otherwise flood an agent's own context

# Anything that is not a single read, matched as whole words so a column named 'updated' is fine.
# The reader families take a \w* suffix: \bread_csv\b does not match read_csv_auto, because the
# underscore is a word character. The connection refuses all of these anyway; this is so the model
# gets a message it can act on rather than a DuckDB permission error.
FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|install|load|export|import|create|insert|update|delete|drop|alter|"
    r"truncate|grant|revoke|set|call|glob|"
    r"read_csv\w*|read_parquet\w*|read_json\w*|read_text|read_blob|parquet_scan|pragma\w*)\b",
    re.I,
)

# DuckDB's own catalog and settings. This is a privacy boundary, not a safety one - the connection
# already blocks everything dangerous - and it is the only thing standing between a model-written
# SELECT and the database's absolute path, which on a developer machine contains the owner's
# account name and would then appear in an answer and in the tool transcript on the page.
#
# Matched by family rather than by name: duckdb_databases() was blocked while duckdb_tables() was
# not, and current_setting('temp_directory') returned the full path with no catalog view involved.
INTERNAL_VIEWS = re.compile(
    r"\b(duckdb_\w+|pg_\w+|sqlite_master|sqlite_schema|current_setting|current_database|"
    r"current_schemas?)\b",
    re.I,
)


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

    Guessing a sector name is the usual way a generated query returns nothing. Read from the live
    table rather than the schema card, so it cannot go stale.
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


def query(sql: str) -> dict:
    """Run one read-only SELECT and return {columns, rows, row_count, sql, truncated}.

    The returned `sql` is the statement actually executed, so it can be cited and re-run. Refuses
    rather than sanitises: rewriting a model's query would make that citation a lie.
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

    try:
        # One row past the cap, so `truncated` can report whether anything was left behind.
        result = _cursor().execute(f"SELECT * FROM ({text}) LIMIT {MAX_ROWS + 1}")
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
    except duckdb.Error as e:
        # DuckDB's messages name the offending column and suggest near-matches, so they are worth
        # passing through verbatim rather than replacing with something vaguer.
        raise RefDataError(f"SQL error: {e}")
    truncated = len(rows) > MAX_ROWS
    return {
        "columns": columns,
        "rows": [list(r) for r in rows[:MAX_ROWS]],
        "row_count": len(rows[:MAX_ROWS]),
        "truncated": truncated,
        "sql": text,
        # Named so the agent can attribute the figure without inventing a provenance string. There
        # is deliberately no `delay`: a committed table has none, and the prompt used to imply
        # every result had one, which produced answers citing a "delay: static" that never existed.
        "source": "committed reference database (data/refdata.duckdb)",
        **({"note": f"Showing the first {MAX_ROWS} rows; add a tighter WHERE or an ORDER BY."} if truncated else {}),
    }


@functools.cache
def name_for(symbol: str) -> str | None:
    """The listed name of `symbol` from the reference table, or None if it is not in there.

    Table first, so a name in the tape matches the name used in an answer. Callers fall back to the
    listing directory for anything outside the table, such as the indices.
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

    Returns the symbols *and* the SQL that produced them, since the universe decides every pair
    that follows. Capped because screening is quadratic, and five GICS sectors exceed the cap - for
    those the result must say so loudly, or the agent calls a quarter of a sector the sector.
    """
    capped = min(max(2, limit), config.MAX_UNIVERSE)
    if symbols:
        wanted = [s.strip().upper() for s in symbols if s and s.strip()]
        if len(wanted) < 2:
            raise RefDataError("Give at least two symbols to compare.")
        if len(wanted) > capped:
            # Capped here as well as in screen.screen, so the caller learns which limit it hit.
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
        where = f"etf_family = '{_safe(etf_family)}'"
        order = "leverage, symbol"
        label = f"the {etf_family} ETF family"
    elif sector:
        # Match a GICS sector or a GICS sub-industry, so both the broad and the fine name work.
        value = _safe(sector)
        where = (f"lower(gics_sector) = lower('{value}') "
                 f"OR lower(gics_sub_industry) = lower('{value}')")
        order = "symbol"
        label = sector
    else:
        raise RefDataError("Pass a sector, an etf_family, or an explicit list of symbols.")

    # Count before the cap, so truncation is reported rather than hidden.
    available = _cursor().execute(f"SELECT count(*) FROM instruments WHERE {where}").fetchone()[0]
    sql = (f"SELECT symbol, name, gics_sector, etf_family FROM instruments "
           f"WHERE {where} ORDER BY {order} LIMIT {capped}")

    rows = _cursor().execute(sql).fetchall()
    if not rows:
        raise RefDataError(
            "Nothing matched. Call get_reference_schema to see the values that exist for "
            "gics_sector and etf_family, or query_reference with "
            "SELECT DISTINCT gics_sub_industry FROM instruments for the finer groupings."
        )
    return _universe(rows, sql, [], available=available, label=label, where=where)


def _universe(rows: list[tuple], sql: str, missing: list[str],
              available: int | None = None, label: str = "", where: str = "") -> dict:
    """Shape a resolved universe, naming anything that was dropped - missing or over the cap."""
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
    if available is not None and available > len(rows):
        dropped = available - len(rows)
        out["truncated"] = True
        out["available"] = available
        out["note"] = (
            f"PARTIAL UNIVERSE: {label} has {available} names in the reference table but screening "
            f"is capped at {config.MAX_UNIVERSE}, so {dropped} were dropped and only the first "
            f"{len(rows)} alphabetically were screened. This is a subset: say so, and do not "
            f"describe the result as covering {label}. To cover it properly, screen its "
            f"gics_sub_industry groups one at a time: run query_reference with "
            f"\"SELECT gics_sub_industry, count(*) FROM instruments WHERE {where} "
            f"GROUP BY 1 ORDER BY 2 DESC\" and screen each in turn."
        )
    return out


def _safe(value: str) -> str:
    """Escape a single-quoted SQL literal.

    Interpolated rather than bound so the returned `sql` is complete and re-runnable. The
    connection is read-only with external access off, so a bad literal costs a syntax error.
    """
    return value.replace("'", "''")
