import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import litellm
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tools import TOOLS, run_tool

# --- Config ---

SYSTEM_PROMPT = (
    "You are Ticker, a markets research assistant for traders. Today is {today}. "
    "You research US equities with the tools provided.\n"
    "- Call a tool for every price, options, news, filings or chart question and use only what it returns. Never "
    "quote a price you did not just fetch, add details from memory, or invent data. A missing headline or filing "
    "proves nothing.\n"
    "- Crypto, futures (VIX futures too), executive share transactions (Form 4), fundamentals, rates and FX are not "
    "available: say so in your first sentence and do not infer them from news or filings. Say 'executive share "
    "transactions', never 'insider'. For crypto, offer an ETF such as IBIT or ETHA, labelled as an ETF, and quote it "
    "only after calling get_price.\n"
    "- To explain a price move, always call all three of get_price_history, get_news and get_filings for the dates "
    "around it, together, and judge each item: earnings, guidance, analyst changes, M&A, regulation and products can "
    "move a stock; predictions, listicles and opinion rarely do. Check the news came before the move, and finish with "
    "a short verdict: the likeliest catalyst and date, what probably didn't matter, and your confidence (at most "
    "medium unless a headline directly attributes the move).\n"
    "- For options, give the expiry, ATM implied volatility, straddle price, implied move % and put/call ratios.\n"
    "- Treat trade ideas as hypotheses, not buy/sell advice. State the source and delay: prices and options ~15 min "
    "delayed (the last close when the market is shut), news and filings same day.\n"
    "- If a tool errors, fix the arguments or say the data is unavailable.\n"
    "- Be concise: answer first, then the key numbers. Write every price or percent change with a sign "
    "(+4.65%, -$2.10). Plain text, **bold** numbers and '-' bullets, no headings or tables."
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
RATE_LIMITED = (
    "Vertex AI is rate limiting this model (HTTP 429: its per-minute quota is used up). Wait a minute and try again, "
    "or pick another model or a lower tool call limit."
)
LIMIT_NOTE = {
    "role": "user",
    "content": "The tool-call limit is reached, so no more tools are available. Answer with what you have, and say if "
    "you could not make all the calls you needed.",
}
LIMIT_REACHED = json.dumps(
    {"error": "Tool-call limit reached, so this call was not run. Answer with what you have and say what is missing."}
)


def today() -> str:
    """Today in US Eastern time, where the markets are (UTC if the machine has no time-zone data)."""
    try:
        now = datetime.now(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        now = datetime.now(timezone.utc)
    return now.strftime("%A %Y-%m-%d")


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

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str = DEFAULT_MODEL
    tool_call_limit: int = Field(DEFAULT_TOOL_CALLS, ge=1, le=MAX_TOOL_CALLS)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/config")
def config():
    """What the UI offers: the models and the tool-call limit range."""
    return {
        "models": list(MODELS),
        "default_model": DEFAULT_MODEL,
        "max_tool_calls": MAX_TOOL_CALLS,
        "default_tool_calls": DEFAULT_TOOL_CALLS,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.model not in MODELS:
        raise HTTPException(422, f"Unknown model '{request.model}'. Choose one of: {list(MODELS)}")

    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT.format(today=today())}]

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
