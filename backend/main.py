import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json

app = FastAPI(title="Drift API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# --- Models ---

class DumpRequest(BaseModel):
    text: str

class DumpItem(BaseModel):
    text: str
    category: str  # "task", "goal", "idea", "obligation"

class DumpResponse(BaseModel):
    items: list[DumpItem]

class FocusItem(BaseModel):
    text: str
    created_at: str
    last_touched_at: str

class FocusGoal(BaseModel):
    text: str
    horizon: str  # "week", "month", "quarter"

class FocusRequest(BaseModel):
    items: list[FocusItem]
    goals: list[FocusGoal]

class FocusResponse(BaseModel):
    focus: str


# --- Endpoints ---

@app.post("/parse-dump", response_model=DumpResponse)
async def parse_dump(req: DumpRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty dump")

    if not client.api_key:
        return DumpResponse(items=_fallback_parse(req.text))

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": req.text},
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.3,
        )
        data = json.loads(resp.choices[0].message.content)
        items = [DumpItem(**i) for i in data.get("items", [])]
        return DumpResponse(items=items if items else _fallback_parse(req.text))
    except Exception:
        return DumpResponse(items=_fallback_parse(req.text))


@app.post("/daily-focus", response_model=FocusResponse)
async def daily_focus(req: FocusRequest):
    if not client.api_key:
        return FocusResponse(focus=_fallback_focus(req.items, req.goals))

    prompt = _build_focus_prompt(req.items, req.goals)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FOCUS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return FocusResponse(focus=resp.choices[0].message.content.strip())
    except Exception:
        return FocusResponse(focus=_fallback_focus(req.items, req.goals))


@app.get("/health")
async def health():
    return {"status": "ok", "ai": bool(client.api_key)}


# --- Prompts ---

PARSE_SYSTEM_PROMPT = """You extract structured items from a brain dump.
Return JSON with an "items" array. Each item has:
- "text": the task/idea cleaned up into a short actionable phrase
- "category": one of "task", "goal", "idea", "obligation"

Be smart about splitting. "Buy groceries like milk and eggs" is ONE item, not two.
"File taxes and finish the book" is TWO items.
Keep the user's intent intact. Don't invent things they didn't say."""

FOCUS_SYSTEM_PROMPT = """You are Drift, a calm and thoughtful AI companion.
Your job is to look at everything on someone's plate and their goals,
then pick the 1-2 things they should focus on TODAY.
Be specific, warm, and brief. No bullet lists — write like a friend texting.
If something has been idle for a while, mention it gently."""


# --- Fallbacks ---

def _fallback_parse(text: str) -> list[DumpItem]:
    entries = [e.strip() for e in text.replace("\n", ",").split(",") if len(e.strip()) > 2]
    return [DumpItem(text=e, category="task") for e in entries]


def _fallback_focus(items: list[FocusItem], goals: list[FocusGoal]) -> str:
    if items:
        stale = sorted(items, key=lambda i: i.last_touched_at)[0]
        return f'Hey — you haven\'t touched "{stale.text}" in a while. Maybe start there today?'
    if goals:
        return "You've set some goals but haven't added any tasks yet. What's one small step today?"
    return "Nothing on your plate yet. Dump whatever's on your mind."


def _build_focus_prompt(items: list[FocusItem], goals: list[FocusGoal]) -> str:
    goals_text = "\n".join(f"- {g.text} ({g.horizon})" for g in goals) or "No goals set."
    items_text = "\n".join(
        f"- {i.text} (added: {i.created_at[:10]}, last touched: {i.last_touched_at[:10]})"
        for i in items
    ) or "Nothing on their plate."

    return f"GOALS:\n{goals_text}\n\nTASKS & ITEMS:\n{items_text}\n\nWhat should they focus on today? Pick 1-2 max."
