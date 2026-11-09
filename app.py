import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import litellm
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import data
from tools import TOOLS, run_tool

# --- Config ---

SYSTEM_PROMPT = (
    "You are Ticker, a markets research assistant for traders. Today is {today}. "
    "You research US equities with the tools provided.\n"
    "- Call a tool for every price, options, news, filings or chart question and use only what it returns. Never "
    "quote a price you did not just fetch, add details from memory, or invent data. A missing headline or filing "
    "proves nothing.\n"
    "- If the user gives a company or fund name instead of a ticker, always call search_symbol first, even if you "
    "think you know the ticker, and say which ticker you used.\n"
    "- Crypto, futures (VIX futures too), executive share transactions (Form 4), fundamentals, rates and FX are not "
    "available: when the user asks for one, say so in your first sentence and do not infer it from news or filings. "
    "Otherwise do not mention this list. Say 'executive share "
    "transactions', never 'insider'. For crypto, offer an ETF such as IBIT or ETHA, labelled as an ETF, and quote it "
    "only after calling get_price.\n"
    "- To explain a price move, always call all three of get_price_history, get_news and get_filings for the dates "
    "around it, together, and judge each item: earnings, guidance, analyst changes, M&A, regulation and products can "
    "move a stock; predictions, listicles and opinion rarely do. Check the news came before the move, and finish with "
    "a short verdict: the likeliest catalyst and date, what probably didn't matter, and your confidence (at most "
    "medium unless a headline directly attributes the move).\n"
    "- For options, give the expiry, ATM implied volatility, straddle price, put/call ratios and the implied_move "
    "text exactly as returned.\n"
    "- For risk, volatility, drawdown, beta or Sharpe questions, call get_risk_metrics, compare with its benchmark, "
    "and say returns exclude dividends.\n"
    "- Treat trade ideas as hypotheses, not buy/sell advice. State the source and delay: prices and options ~15 min "
    "delayed (the last close when the market is shut), news and filings same day.\n"
    "- If a tool errors, fix the arguments or say the data is unavailable.\n"
    "- Be concise: answer first, then the key numbers. Write every price or percent change or return with a "
    "sign (+4.65%, -$2.10), but not levels such as volatility, rates or ratios. Plain text, **bold** numbers and "
    "'-' bullets, no headings or tables."
)

# Small, light-weight Gemini models on Vertex AI that work with tool calling in this project.
# Name shown in the UI -> LiteLLM model id.
MODELS = {
    "Gemini 3.5 Flash-Lite": "vertex_ai/gemini-3.5-flash-lite",
    "Gemini 2.5 Flash-Lite": "vertex_ai/gemini-2.5-flash-lite",
    "Gemini 3.5 Flash": "vertex_ai/gemini-3.5-flash",
    "Gemini 2.5 Flash": "vertex_ai/gemini-2.5-flash",
}
DEFAULT_MODEL = "Gemini 3.5 Flash-Lite"
MAX_TOOL_CALLS, DEFAULT_TOOL_CALLS = 10, 5  # the user can pick a limit from 1 to MAX_TOOL_CALLS per question
# Most-watched US tickers for the scrolling tape: index funds, the VIX, then large caps. Quoted at start-up and kept.
TAPE_SYMBOLS = [
    "SPY", "QQQ", "VOO", "VIX", "NVDA", "MSFT", "AAPL", "GOOG", "AMZN", "META", "AVGO", "TSLA", "BRK.B",
    "JPM", "LLY", "V", "WMT", "XOM", "ORCL", "NFLX", "COST", "GS", "MS", "AMD", "PLTR",
]
INDEX_NAMES = {"VIX": "Cboe Volatility Index"}  # indices are not in the symbol directory
TAPE_MAX_AGE = 3600  # seconds; a long-running server asks for the quotes again once they are this old
TAPE_RETRY = 300  # seconds before trying again after every quote failed (Cboe rate limits bursts)
RATE_LIMITED = (
    "Vertex AI is rate limiting this model (HTTP 429: its per-minute quota is used up). Wait a minute and try again, "
    "or pick another model or a lower tool limit."
)
LIMIT_NOTE = {
    "role": "user",
    "content": "The tool-call limit is reached, so no more tools are available. Answer with what you have, and say if "
    "you could not make all the calls you needed.",
}
LIMIT_REACHED = json.dumps(
    {"error": "Tool-call limit reached, so this call was not run. Answer with what you have and say what is missing."}
)


def eastern_now() -> datetime:
    """Now in US Eastern time, where the markets are (UTC if the machine has no time-zone data)."""
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc)


# --- The Harness ---


def for_model(result: str) -> str:
    """Drop the chart image (bulky, only for the UI) so the model reads just the summary."""
    summary = json.loads(result)
    summary.pop("chart_svg", None)
    return json.dumps(summary)


def run_agent(messages: list[dict], model: str, call_limit: int) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool, or the tool-call limit is used up.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    while True:
        # Once the limit is used up, take the tools away so the model has to answer with what it has.
        out_of_calls = len(tool_calls) >= call_limit
        reply = litellm.completion(
            model=MODELS[model],
            vertex_location="global",
            messages=(messages + [LIMIT_NOTE]) if out_of_calls else messages,
            tools=None if out_of_calls else TOOLS,
        ).choices[0].message

        if out_of_calls or not reply.tool_calls:
            if not reply.content:  # models sometimes return nothing; keep that out of the history
                return "Sorry, the model returned an empty answer. Please try again.", tool_calls
            messages += [{"role": "assistant", "content": reply.content}]
            return reply.content, tool_calls

        # Append assistant's tool calls to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            if len(tool_calls) >= call_limit:  # one round can ask for more calls than are left
                messages += [{"role": "tool", "tool_call_id": call.id, "content": LIMIT_REACHED}]
                continue
            try:
                args = json.loads(call.function.arguments)
                result = run_tool(call.function.name, args)
            except json.JSONDecodeError:
                # Report it to the model like any tool error, so it can resend valid arguments.
                args, result = {}, json.dumps({"error": "Tool arguments were not valid JSON. Send a JSON object."})
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": for_model(result)}]


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# --- Tape ---

tape = {"at": 0.0, "rows": []}  # the saved quotes and when they were fetched


def tape_row(symbol: str) -> dict | None:
    """The last completed session's close and its change on the day before, with what the hover tooltip shows."""
    try:
        bars = data.price_history(symbol, date.today() - timedelta(days=365))
    except Exception:  # one symbol failing must not blank the whole tape
        return None
    now = eastern_now()
    if bars and bars[-1]["date"] == now.strftime("%Y-%m-%d") and now.strftime("%H:%M") < "16:15":
        bars = bars[:-1]  # today's bar is a live price until the 15-minute-delayed close prints
    if len(bars) < 2:
        return None
    last, prev = bars[-1], bars[-2]
    try:
        name = INDEX_NAMES.get(symbol) or data.company_name(symbol)
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


def refresh_tape():
    """Quote every tape symbol, a few at a time. Keeps the old quotes if every request fails."""
    with ThreadPoolExecutor(5) as pool:
        rows = [row for row in pool.map(tape_row, TAPE_SYMBOLS) if row]
    tape.update(at=time.time() if rows else time.time() - TAPE_MAX_AGE + TAPE_RETRY, rows=rows or tape["rows"])


# --- FastAPI App ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh_tape()  # quote once before the first visitor arrives
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str = DEFAULT_MODEL
    tool_call_limit: int = Field(DEFAULT_TOOL_CALLS, ge=1, le=MAX_TOOL_CALLS)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/", response_class=HTMLResponse)
def index():
    """The page, with what it shows (models, tool-call limit range, tape quotes) built in so it needs no second request."""
    if time.time() - tape["at"] > TAPE_MAX_AGE:
        refresh_tape()
    config = {
        "tape": tape["rows"],
        "models": list(MODELS),
        "default_model": DEFAULT_MODEL,
        "max_tool_calls": MAX_TOOL_CALLS,
        "default_tool_calls": DEFAULT_TOOL_CALLS,
    }
    page = (Path(__file__).parent / "index.html").read_text()
    return page.replace("__CONFIG__", json.dumps(config).replace("</", "<\\/"))  # a name with </script> must not end the script


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.model not in MODELS:
        raise HTTPException(422, f"Unknown model '{request.model}'. Choose one of: {list(MODELS)}")

    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT.format(today=eastern_now().strftime("%A %Y-%m-%d"))}]

    # Append user's message to the context
    sessions[session_id] += [{"role": "user", "content": request.message}]

    try:
        response, tool_calls = run_agent(sessions[session_id], request.model, request.tool_call_limit)
    except litellm.RateLimitError:
        response, tool_calls = RATE_LIMITED, []
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
