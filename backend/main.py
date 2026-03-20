import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drift")

app = FastAPI(title="Drift API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Provider Setup ---
# Default: OPENROUTER_API_KEY from env (free tier)
# Per-request: user can send X-Gemini-Key header to use Google Gemini directly
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
if openrouter_key:
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key, timeout=90.0)
    log.info("[init] OpenRouter configured (model=openrouter/free)")
else:
    openrouter_client = None
    log.warning("[init] No OPENROUTER_API_KEY — will need per-request Gemini keys or fallback")

supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")
supabase: Client | None = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


# --- Models ---

class ExistingItem(BaseModel):
    id: int
    text: str
    category: str | None = None
    is_goal: bool = False

class DumpRequest(BaseModel):
    text: str
    existing_items: list[ExistingItem] = []

class DumpNewItem(BaseModel):
    text: str
    category: str           # "task" or "goal"
    goal_horizon: str | None = None  # "week", "month", "quarter" — only for goals
    parent_goal: str | None = None   # text of the goal this task serves

class DumpUpdate(BaseModel):
    id: int                 # ID of the existing item
    action: str             # "touch", "done", "let_go"

class DumpResponse(BaseModel):
    new_items: list[DumpNewItem] = []
    updates: list[DumpUpdate] = []

class FocusItem(BaseModel):
    text: str
    created_at: str
    last_touched_at: str

class FocusGoal(BaseModel):
    text: str
    horizon: str

class FocusRequest(BaseModel):
    items: list[FocusItem]
    goals: list[FocusGoal]

class FocusAction(BaseModel):
    text: str           # What to do
    why: str            # Why this matters / which goal it serves
    goal: str | None    # Related goal, if any

class FocusResponse(BaseModel):
    greeting: str
    actions: list[FocusAction]
    parked: list[str]   # Things deliberately deprioritized today
    source: str = "ai"  # "ai" or "fallback" — tells the app whether to cache

class WaitlistRequest(BaseModel):
    email: str
    source: str = "quiz"


# --- Endpoints ---

@app.post("/parse-dump", response_model=DumpResponse)
async def parse_dump(req: DumpRequest, request: Request):
    device_id = request.headers.get("x-device-id")
    gemini_key = request.headers.get("x-gemini-key")
    log.info(f"[parse-dump] REQUEST: {req.text[:200]} (device={device_id}, gemini={'yes' if gemini_key else 'no'})")
    log.info(f"[parse-dump] EXISTING: {[(e.id, e.text) for e in req.existing_items]}")

    if not req.text.strip():
        raise HTTPException(400, "Empty dump")

    if not gemini_key and not openrouter_client:
        log.info("[parse-dump] No AI available — using fallback")
        result = _fallback_parse(req.text)
        log.info(f"[parse-dump] RESPONSE (fallback): {result.model_dump_json()}")
        _log_api_call("parse-dump", req.model_dump(), result.model_dump(), source="fallback", device_id=device_id)
        return result

    try:
        user_content = _build_parse_user_content(req.text, req.existing_items)

        raw_text, provider = _call_ai(PARSE_SYSTEM_PROMPT, user_content, temperature=0.3, max_tokens=2048, gemini_key=gemini_key)
        log.info(f"[parse-dump] AI RAW RESPONSE ({provider}): {raw_text}")

        data = json.loads(raw_text)
        result = DumpResponse(
            new_items=[DumpNewItem(**i) for i in data.get("new_items", [])],
            updates=[DumpUpdate(**u) for u in data.get("updates", [])],
        )
        log.info(f"[parse-dump] RESPONSE: {result.model_dump_json()}")
        _log_api_call("parse-dump", req.model_dump(), result.model_dump(), source=provider, device_id=device_id)
        return result
    except Exception as e:
        log.error(f"[parse-dump] AI ERROR: {e}")
        result = _fallback_parse(req.text)
        log.info(f"[parse-dump] RESPONSE (fallback): {result.model_dump_json()}")
        _log_api_call("parse-dump", req.model_dump(), result.model_dump(), source="fallback", device_id=device_id)
        return result


@app.post("/daily-focus", response_model=FocusResponse)
async def daily_focus(req: FocusRequest, request: Request):
    device_id = request.headers.get("x-device-id")
    gemini_key = request.headers.get("x-gemini-key")
    log.info(f"[daily-focus] REQUEST: items={len(req.items)}, goals={len(req.goals)} (device={device_id}, gemini={'yes' if gemini_key else 'no'})")
    log.info(f"[daily-focus] ITEMS: {[i.text for i in req.items]}")
    log.info(f"[daily-focus] GOALS: {[g.text for g in req.goals]}")

    if not gemini_key and not openrouter_client:
        log.info("[daily-focus] No AI available — using fallback")
        result = _fallback_focus(req.items, req.goals, source="fallback")
        log.info(f"[daily-focus] RESPONSE (fallback): {result.model_dump_json()}")
        _log_api_call("daily-focus", req.model_dump(), result.model_dump(), source="fallback", device_id=device_id)
        return result

    user_content = _build_focus_prompt(req.items, req.goals)

    try:
        raw_text, provider = _call_ai(FOCUS_SYSTEM_PROMPT, user_content, temperature=0.7, max_tokens=1024, gemini_key=gemini_key)
        log.info(f"[daily-focus] AI RAW RESPONSE ({provider}): {raw_text}")

        data = json.loads(raw_text)
        result = FocusResponse(
            greeting=data.get("greeting", "Hey! Here's your focus for today."),
            actions=[FocusAction(**a) for a in data.get("actions", [])],
            parked=data.get("parked", []),
            source="ai",
        )
        log.info(f"[daily-focus] RESPONSE: {result.model_dump_json()}")
        _log_api_call("daily-focus", req.model_dump(), result.model_dump(), source=provider, device_id=device_id)
        return result
    except Exception as e:
        log.error(f"[daily-focus] AI ERROR: {e}")
        result = _fallback_focus(req.items, req.goals, source="fallback")
        log.info(f"[daily-focus] RESPONSE (fallback): {result.model_dump_json()}")
        _log_api_call("daily-focus", req.model_dump(), result.model_dump(), source="fallback", device_id=device_id)
        return result


@app.get("/health")
@app.head("/health")
async def health():
    log.info(f"[health] openrouter={'yes' if openrouter_client else 'no'}")
    return {"status": "ok", "openrouter": openrouter_client is not None, "gemini": "per-request via X-Gemini-Key header"}


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drift — Privacy Policy</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }
  h1 { color: #7C3AED; }
  h2 { color: #555; margin-top: 2em; }
</style>
</head>
<body>
<h1>Drift — Privacy Policy</h1>
<p><strong>Last updated:</strong> March 2026</p>

<h2>What Drift collects</h2>
<p>Drift stores your brain dump text, tasks, and goals locally on your device using a local database. When you use AI-powered features (brain dump parsing, daily focus), your text is sent to our backend server for processing.</p>

<h2>How we use your data</h2>
<p>Your text is sent to an AI service (Google Gemini or OpenRouter) to generate structured responses (tasks, goals, daily focus suggestions). We do not store your text on our servers — it is processed in real time and discarded. Logs may temporarily contain request data for debugging and are not shared with third parties.</p>

<h2>Data stored on your device</h2>
<p>All items, goals, and focus history are stored locally on your Android device using Room (SQLite). This data never leaves your device except when you trigger an AI feature.</p>

<h2>Third-party services</h2>
<p>Drift uses AI services (Google Gemini or OpenRouter, depending on configuration) for AI features. The respective provider's privacy policy applies to data processed by their API. No other third-party analytics, ads, or tracking services are used.</p>

<h2>No account required</h2>
<p>Drift does not require sign-up, login, or any personal information. No email, name, or location data is collected.</p>

<h2>Data deletion</h2>
<p>Since all data is stored locally, you can delete everything by clearing the app's data in your Android settings or uninstalling the app.</p>

<h2>Contact</h2>
<p>Questions? Reach out at <strong>captwist81@gmail.com</strong></p>
</body>
</html>"""
@app.post("/waitlist")
async def waitlist(req: WaitlistRequest):
    log.info(f"[waitlist] NEW SIGNUP: {req.email} (source={req.source})")
    if not supabase:
        log.error("[waitlist] Supabase not configured")
        raise HTTPException(500, "Waitlist unavailable")
    try:
        supabase.table("waitlist").insert({
            "email": req.email,
            "source": req.source,
        }).execute()
        return {"status": "ok"}
    except Exception as e:
        log.error(f"[waitlist] ERROR: {e}")
        raise HTTPException(500, "Failed to save signup")


# --- Prompts ---

PARSE_SYSTEM_PROMPT = """You are Drift's brain dump parser. You understand natural language about tasks, goals, and progress updates.

You receive the user's brain dump AND their existing items/goals. Your job is to figure out:
1. What's NEW (tasks, goals, ideas) that should be added
2. What's an UPDATE on something that already exists (progress, completion, abandonment)

Return JSON with this exact structure:
{
  "new_items": [
    {
      "text": "Short actionable phrase",
      "category": "task" or "goal",
      "goal_horizon": "week" or "month" or "quarter" (only if category is "goal", else null),
      "parent_goal": "Text of the existing goal this task serves (or null)"
    }
  ],
  "updates": [
    {
      "id": 123,
      "action": "touch" or "done" or "let_go"
    }
  ]
}

Rules:
- If the user mentions completing or finishing something that matches an existing item, that's an update with action "done", NOT a new item.
- If the user reports progress on an existing item (e.g. "I gathered the documents"), mark it as "touch" and create new next-step tasks if mentioned.
- If something has a deadline or is outcome-shaped (e.g. "File taxes by March 31"), it's a "goal" not a "task".
- Tasks that clearly serve a goal should have "parent_goal" set to that goal's text.
- Be smart about splitting. "Buy groceries like milk and eggs" is ONE item. "File taxes and finish the book" is TWO.
- Don't create duplicate items that already exist.
- Ignore conversational fluff like "thanks drift" or "hey".
- Keep the user's intent intact. Don't invent things they didn't say."""

FOCUS_SYSTEM_PROMPT = """You are Drift, a calm and thoughtful AI companion.
Look at everything on someone's plate and their goals, then decide what they should focus on TODAY.

Return JSON with this exact structure:
{
  "greeting": "A short, warm one-liner (1 sentence max, like a friend texting)",
  "actions": [
    {
      "text": "What to do (specific, actionable, short)",
      "why": "Why this matters today (1 sentence)",
      "goal": "Which goal this serves, or null if it's standalone"
    }
  ],
  "parked": ["Things you're deliberately NOT suggesting today and why, as short strings"]
}

Rules:
- Pick 1-2 actions max. Less is more.
- If something has been idle for a while, prioritize it.
- Link actions to goals when possible.
- Put everything else in "parked" so the user knows you haven't forgotten.
- Keep all text short and warm. No corporate speak."""


# --- Unified AI caller ---

def _call_ai(system_prompt: str, user_content: str, temperature: float = 0.5, max_tokens: int = 1024, gemini_key: str | None = None) -> tuple[str, str]:
    """Call AI provider. Returns (raw_json_text, provider_name).
    Uses user's Gemini key if provided, otherwise falls back to OpenRouter."""
    if gemini_key:
        from google import genai
        gc = genai.Client(api_key=gemini_key)
        prompt = f"{system_prompt}\n\n{user_content}"
        resp = gc.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
        )
        return resp.text, "gemini"
    elif openrouter_client:
        resp = openrouter_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenRouter returned empty response (free tier may be at capacity)")
        return content, "openrouter"
    else:
        raise RuntimeError("No AI provider available")


# --- Helpers ---

def _log_api_call(endpoint: str, request_data: dict, response_data: dict, source: str = "ai", device_id: str | None = None):
    """Log API request/response to Supabase for observability."""
    if not supabase:
        return
    try:
        supabase.table("api_logs").insert({
            "endpoint": endpoint,
            "request": json.dumps(request_data),
            "response": json.dumps(response_data),
            "source": source,
            "device_id": device_id,
        }).execute()
    except Exception as e:
        log.error(f"[api_logs] Failed to log: {e}")


# --- Fallbacks ---

def _fallback_parse(text: str) -> DumpResponse:
    entries = [e.strip() for e in text.replace("\n", ",").split(",") if len(e.strip()) > 2]
    return DumpResponse(
        new_items=[DumpNewItem(text=e, category="task") for e in entries],
        updates=[],
    )


def _build_parse_prompt(text: str, existing: list[ExistingItem]) -> str:
    """Build a combined prompt string for logging purposes."""
    parts = [PARSE_SYSTEM_PROMPT]
    parts.append(_build_parse_user_content(text, existing))
    return "\n".join(parts)


def _build_parse_user_content(text: str, existing: list[ExistingItem]) -> str:
    """Build just the user content portion for the chat message."""
    parts = []
    if existing:
        items_text = "\n".join(
            f"- [id={e.id}] {e.text} ({'goal' if e.is_goal else e.category or 'task'})"
            for e in existing
        )
        parts.append(f"EXISTING ITEMS & GOALS:\n{items_text}")
    else:
        parts.append("EXISTING ITEMS & GOALS:\nNone yet.")
    parts.append(f"\nUSER'S BRAIN DUMP:\n{text}")
    return "\n".join(parts)


def _fallback_focus(items: list[FocusItem], goals: list[FocusGoal], source: str = "fallback") -> FocusResponse:
    if items:
        stale = sorted(items, key=lambda i: i.last_touched_at)[0]
        return FocusResponse(
            greeting="Hey! Here's what I'd focus on today.",
            actions=[FocusAction(text=stale.text, why="This has been sitting untouched the longest.", goal=None)],
            parked=[i.text for i in items if i.text != stale.text][:3],
            source=source,
        )
    if goals:
        return FocusResponse(
            greeting="You've set some goals — nice!",
            actions=[FocusAction(text="Add a few tasks to your plate", why="Goals need small steps to get started.", goal=goals[0].text)],
            parked=[],
            source=source,
        )
    return FocusResponse(
        greeting="Nothing on your plate yet.",
        actions=[FocusAction(text="Dump whatever's on your mind", why="That's how we figure out what matters.", goal=None)],
        parked=[],
        source=source,
    )


def _build_focus_prompt(items: list[FocusItem], goals: list[FocusGoal]) -> str:
    goals_text = "\n".join(f"- {g.text} ({g.horizon})" for g in goals) or "No goals set."
    items_text = "\n".join(
        f"- {i.text} (added: {i.created_at[:10]}, last touched: {i.last_touched_at[:10]})"
        for i in items
    ) or "Nothing on their plate."
    return f"GOALS:\n{goals_text}\n\nTASKS & ITEMS:\n{items_text}\n\nWhat should they focus on today? Pick 1-2 max."
