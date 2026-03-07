import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

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

api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None
MODEL = "gemini-2.5-flash"


# --- Models ---

class DumpRequest(BaseModel):
    text: str

class DumpItem(BaseModel):
    text: str
    category: str

class DumpResponse(BaseModel):
    items: list[DumpItem]

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


# --- Endpoints ---

@app.post("/parse-dump", response_model=DumpResponse)
async def parse_dump(req: DumpRequest):
    log.info(f"[parse-dump] REQUEST: {req.text[:200]}")

    if not req.text.strip():
        raise HTTPException(400, "Empty dump")

    if not client:
        log.info("[parse-dump] No AI client — using fallback")
        result = DumpResponse(items=_fallback_parse(req.text))
        log.info(f"[parse-dump] RESPONSE (fallback): {result.model_dump_json()}")
        return result

    try:
        prompt = f"{PARSE_SYSTEM_PROMPT}\n\nUser's brain dump:\n{req.text}"
        log.info(f"[parse-dump] AI PROMPT: {prompt}")

        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            },
        )
        log.info(f"[parse-dump] AI RAW RESPONSE: {resp.text}")

        data = json.loads(resp.text)
        items = [DumpItem(**i) for i in data.get("items", [])]
        result = DumpResponse(items=items if items else _fallback_parse(req.text))
        log.info(f"[parse-dump] RESPONSE: {result.model_dump_json()}")
        return result
    except Exception as e:
        log.error(f"[parse-dump] AI ERROR: {e}")
        result = DumpResponse(items=_fallback_parse(req.text))
        log.info(f"[parse-dump] RESPONSE (fallback): {result.model_dump_json()}")
        return result


@app.post("/daily-focus", response_model=FocusResponse)
async def daily_focus(req: FocusRequest):
    log.info(f"[daily-focus] REQUEST: items={len(req.items)}, goals={len(req.goals)}")
    log.info(f"[daily-focus] ITEMS: {[i.text for i in req.items]}")
    log.info(f"[daily-focus] GOALS: {[g.text for g in req.goals]}")

    if not client:
        log.info("[daily-focus] No AI client — using fallback")
        result = _fallback_focus(req.items, req.goals)
        log.info(f"[daily-focus] RESPONSE (fallback): {result.model_dump_json()}")
        return result

    prompt = f"{FOCUS_SYSTEM_PROMPT}\n\n{_build_focus_prompt(req.items, req.goals)}"
    log.info(f"[daily-focus] AI PROMPT: {prompt}")

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            },
        )
        log.info(f"[daily-focus] AI RAW RESPONSE: {resp.text}")

        data = json.loads(resp.text)
        result = FocusResponse(
            greeting=data.get("greeting", "Hey! Here's your focus for today."),
            actions=[FocusAction(**a) for a in data.get("actions", [])],
            parked=data.get("parked", []),
        )
        log.info(f"[daily-focus] RESPONSE: {result.model_dump_json()}")
        return result
    except Exception as e:
        log.error(f"[daily-focus] AI ERROR: {e}")
        result = _fallback_focus(req.items, req.goals)
        log.info(f"[daily-focus] RESPONSE (fallback): {result.model_dump_json()}")
        return result


@app.get("/health")
async def health():
    log.info(f"[health] ai={client is not None}, model={MODEL}")
    return {"status": "ok", "ai": client is not None, "model": MODEL}


# --- Prompts ---

PARSE_SYSTEM_PROMPT = """You extract structured items from a brain dump.
Return JSON with an "items" array. Each item has:
- "text": the task/idea cleaned up into a short actionable phrase
- "category": one of "task", "goal", "idea", "obligation"

Be smart about splitting. "Buy groceries like milk and eggs" is ONE item, not two.
"File taxes and finish the book" is TWO items.
Keep the user's intent intact. Don't invent things they didn't say."""

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


# --- Fallbacks ---

def _fallback_parse(text: str) -> list[DumpItem]:
    entries = [e.strip() for e in text.replace("\n", ",").split(",") if len(e.strip()) > 2]
    return [DumpItem(text=e, category="task") for e in entries]


def _fallback_focus(items: list[FocusItem], goals: list[FocusGoal]) -> FocusResponse:
    if items:
        stale = sorted(items, key=lambda i: i.last_touched_at)[0]
        return FocusResponse(
            greeting="Hey! Here's what I'd focus on today.",
            actions=[FocusAction(text=stale.text, why="This has been sitting untouched the longest.", goal=None)],
            parked=[i.text for i in items if i.text != stale.text][:3],
        )
    if goals:
        return FocusResponse(
            greeting="You've set some goals — nice!",
            actions=[FocusAction(text="Add a few tasks to your plate", why="Goals need small steps to get started.", goal=goals[0].text)],
            parked=[],
        )
    return FocusResponse(
        greeting="Nothing on your plate yet.",
        actions=[FocusAction(text="Dump whatever's on your mind", why="That's how we figure out what matters.", goal=None)],
        parked=[],
    )


def _build_focus_prompt(items: list[FocusItem], goals: list[FocusGoal]) -> str:
    goals_text = "\n".join(f"- {g.text} ({g.horizon})" for g in goals) or "No goals set."
    items_text = "\n".join(
        f"- {i.text} (added: {i.created_at[:10]}, last touched: {i.last_touched_at[:10]})"
        for i in items
    ) or "Nothing on their plate."
    return f"GOALS:\n{goals_text}\n\nTASKS & ITEMS:\n{items_text}\n\nWhat should they focus on today? Pick 1-2 max."
