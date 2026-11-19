"""Converge web surface: routes, the price tape and the page. No agent logic lives here."""

import functools
import gzip
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import uvicorn
from agents import SQLiteSession
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
import plotly
from plotly.offline import get_plotlyjs
from pydantic import BaseModel

import config
from pipeline import orchestrator
from sources import data, refdata

# --- Config ---

# Kept short: each symbol costs one rate-limited Cboe fetch at startup.
TAPE_SYMBOLS = [
    "SPY", "QQQ", "VIX", "NVDA", "MSFT", "AAPL", "GOOG", "AMZN",
    "META", "TSLA", "NFLX", "AMD", "PLTR", "JPM", "GS", "MS",
]
INDEX_NAMES = {"VIX": "Cboe Volatility Index"}  # indices are not in the symbol directory

# On Cloud Run this file is in memory, so history lasts only as long as the instance.
SESSIONS_DB = "conversations.db"

# Reused: each session holds a sqlite connection that only close() frees.
sessions: dict[str, SQLiteSession] = {}


def session_for(session_id: str) -> SQLiteSession:
    """The session for this id, created on first use."""
    if session_id not in sessions:
        sessions[session_id] = SQLiteSession(session_id, SESSIONS_DB)
    return sessions[session_id]


# --- Tape ---

tape = {"rows": []}


def tape_row(symbol: str) -> dict | None:
    """The last completed session's close and its change on the day before, plus tooltip detail."""
    try:
        bars = data.price_history(symbol, date.today() - timedelta(days=365))
    except Exception:  # one symbol failing must not blank the whole tape
        return None
    now = data.eastern_now()
    if bars and bars[-1]["date"] == now.strftime("%Y-%m-%d") and now.strftime("%H:%M") < "16:15":
        bars = bars[:-1]  # today's bar is a live price until the 15-minute-delayed close prints
    if len(bars) < 2:
        return None
    last, prev = bars[-1], bars[-2]
    try:
        # Reference table first: no network call.
        name = INDEX_NAMES.get(symbol) or refdata.name_for(symbol) or data.company_name(symbol)
    except Exception:  # the name is a nicety
        name = None
    return {
        "symbol": symbol,
        "name": name,
        "price": last["close"],
        "change": last["close"] - prev["close"],
        "change_pct": (last["close"] / prev["close"] - 1) * 100,
        "prev_close": prev["close"],
        "day_low": last["low"],
        "day_high": last["high"],
        "volume": last["volume"],
        "year_low": min(b["low"] for b in bars),
        "year_high": max(b["high"] for b in bars),
    }


def refresh_tape() -> None:
    """Quote every tape symbol once, at startup; no refresh, to spare the rate-limited feed."""
    with ThreadPoolExecutor(5) as pool:
        tape["rows"] = [row for row in pool.map(tape_row, TAPE_SYMBOLS) if row]


# --- FastAPI app ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh_tape()
    _plotly_js()  # gzip now so the first chart request does not wait
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str = config.DEFAULT_RESEARCHER_MODEL


class ChatResponse(BaseModel):
    response: str
    session_id: str
    # In completion order; the page re-sorts by start time to show the fan-out's overlap.
    tool_calls: list[dict]
    charts: list[dict] = []  # Plotly figure specs for the page, never sent to the model


@functools.cache
def _plotly_js() -> bytes:
    """Plotly's bundle, gzipped once and cached as bytes (4.8 MB -> ~1.1 MB)."""
    return gzip.compress(get_plotlyjs().encode(), 6)


@app.get("/plotly-{version}.js")
def plotly_js(version: str):
    """Serve Plotly from the installed package, not a CDN.

    The version is in the path so the year-long immutable cache never pins a stale bundle.
    """
    return Response(
        _plotly_js(),
        media_type="application/javascript",
        headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """The page, with everything it shows built in so it needs no second request."""
    page_config = {
        "tape": tape["rows"],
        "models": list(config.RESEARCHER_MODELS),
        "default_model": config.DEFAULT_RESEARCHER_MODEL,
    }
    page = (Path(__file__).parent / "index.html").read_text()
    # A company name containing "</script>" must not be able to end the script block.
    page = page.replace("__CONFIG__", json.dumps(page_config).replace("</", "<\\/"))
    return page.replace("__PLOTLY_SRC__", f"/plotly-{plotly.__version__}.js")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.model not in config.RESEARCHER_MODELS:
        raise HTTPException(422, f"Unknown model '{request.model}'. Choose one of: {list(config.RESEARCHER_MODELS)}")

    session_id = request.session_id or str(uuid.uuid4())
    try:
        result = await orchestrator.answer(request.message, session_for(session_id), request.model)
    except Exception as e:
        # Backstop (e.g. bad credentials): show the error in the chat, not as a 500.
        result = {"response": f"Model call failed: {type(e).__name__}: {str(e)[:300]}", "tool_calls": [], "charts": []}

    return ChatResponse(session_id=session_id, **result)


@app.post("/clear")
async def clear(session_id: str | None = None):
    """Forget a conversation: drop the cached session and delete its rows (both are needed)."""
    if session_id:
        session = sessions.pop(session_id, None) or SQLiteSession(session_id, SESSIONS_DB)
        try:
            await session.clear_session()
        finally:
            # Already out of the cache, so nothing else would close it.
            session.close()
    return {"status": "ok"}


if __name__ == "__main__":
    # Local runs only. Cloud Run runs `uvicorn app:app`: do not rename this module or `app`.
    uvicorn.run(app, host="127.0.0.1", port=8000)
