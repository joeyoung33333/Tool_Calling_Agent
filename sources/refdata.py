"""Read-only SQL over the committed reference database (the Text2SQL surface).

The boundary is the connection (read-only, no external access, config locked), not the prompt.
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

# Anything but one read; the connection refuses these anyway, but this tells the model why.
FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|install|load|export|import|create|insert|update|delete|drop|alter|"
    r"truncate|grant|revoke|set|call|glob|"
    r"read_csv\w*|read_parquet\w*|read_json\w*|read_text|read_blob|parquet_scan|pragma\w*)\b",
    re.I,
)

# Privacy, not safety: stops a SELECT revealing the database path, which has the owner's name.
INTERNAL_VIEWS = re.compile(
    r"\b(duckdb_\w+|pg_\w+|sqlite_master|sqlite_schema|current_setting|current_database|"
    r"current_schemas?|(?:query|query_table|which_secret)(?=\s*\())\b",  # query('...') runs any SQL
    re.I,
)


def _scrub(value):
    """Remove local filesystem paths from any value: the last defence past INTERNAL_VIEWS."""
    if isinstance(value, str):
        for path, label in ((str(DB_PATH.parent.parent), "<repo>"), (str(Path.home()), "~")):
            value = re.sub(re.escape(path), label, value, flags=re.I)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_scrub(v) for v in value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


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
    """Every distinct value of `column`, from the live table, so filters use real values."""
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

    Refuses rather than sanitises, so the returned `sql` is exactly what ran and can be cited.
    """
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise RefDataError("The query is empty.")
    if ";" in text:
        raise RefDataError("Send one statement at a time, with no semicolons inside it.")
    if not re.match(r"^(select|with)\b", text, re.I):
        raise RefDataError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    # Strip comments and double quotes first: `"query"(...)` is the same call as `query(...)`.
    bare = re.sub(r"/\*.*?\*/|--[^\n]*", " ", text, flags=re.S).replace('"', "")
    if found := FORBIDDEN.search(bare):
        raise RefDataError(f"'{found.group(0)}' is not allowed here; this connection is read-only.")
    if found := INTERNAL_VIEWS.search(bare):
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
        # DuckDB's messages name the column and suggest near-matches, so pass them through.
        raise RefDataError(_scrub(f"SQL error: {e}"))
    truncated = len(rows) > MAX_ROWS
    return {
        "columns": columns,
        "rows": _scrub([list(r) for r in rows[:MAX_ROWS]]),
        "row_count": len(rows[:MAX_ROWS]),
        "truncated": truncated,
        "sql": text,
        # No `delay`: a committed table has none.
        "source": "committed reference database (data/refdata.duckdb)",
        **({"note": f"Showing the first {MAX_ROWS} rows; add a tighter WHERE or an ORDER BY."} if truncated else {}),
    }


@functools.cache
def name_for(symbol: str) -> str | None:
    """The listed name of `symbol` from the reference table, or None if it is not in there."""
    row = _cursor().execute("SELECT name FROM instruments WHERE symbol = ?", [symbol.strip().upper()]).fetchone()
    return row[0] if row else None


def resolve_universe(
    sector: str | None = None,
    etf_family: str | None = None,
    symbols: list[str] | None = None,
    limit: int = config.MAX_UNIVERSE,
) -> dict:
    """Turn a sector, an ETF family or an explicit list into the symbols to screen.

    Returns the SQL too. Capped (screening is quadratic), and truncation must be reported loudly.
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
            f"describe the result as covering {label}. Full coverage takes one screen per "
            f"gics_sub_industry group; list them with query_reference: "
            f"\"SELECT gics_sub_industry, count(*) FROM instruments WHERE {where} "
            f"GROUP BY 1 ORDER BY 2 DESC\"."
        )
    return out


def _safe(value: str) -> str:
    """Escape a single-quoted SQL literal (interpolated so the returned `sql` is re-runnable)."""
    return value.replace("'", "''")
