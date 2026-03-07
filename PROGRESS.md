# Drift Website — Progress

## What's Built

### Landing Page (`frontend/index.html`)
- Single-file static HTML page, no dependencies
- Hero section with core pitch: "Stop letting things silently slip away"
- Problem section with relatable examples (abandoned books, late taxes, stale projects)
- "How it works" — 3-step explanation (dump, set goals, get focus)
- Comparison grid: to-do apps vs Drift
- Interactive 4-question quiz that qualifies visitors:
  - Scores responses and shows one of 3 results (high/medium/low match)
  - Email signup only appears after quiz completion
  - Emails go to Google Forms → Google Sheet
- Bottom CTA drives to quiz
- Fully responsive (mobile-friendly)

### Backend (`backend/main.py`)
- FastAPI server with 3 endpoints:
  - `POST /parse-dump` — Takes raw brain dump text, uses Gemini 2.5 Flash to extract structured items (text + category: task/goal/idea/obligation)
  - `POST /daily-focus` — Takes items + goals, returns structured focus: greeting, 1-2 action cards (what/why/goal), and parked items
  - `GET /health` — Health check showing AI status and model
- Gemini 2.5 Flash (free tier: 20 req/day, no credit card)
- Fallback logic for all endpoints when AI is unavailable
- Full request/response logging for debugging
- CORS enabled for cross-origin access
- Environment config via `.env` file (python-dotenv)

### Deployment
- Backend deployed on Render (free tier): `https://drift-api-evce.onrender.com`
- Render Blueprint (`render.yaml`) for automated deploys from GitHub
- UptimeRobot ping configured to prevent cold starts
- Gemini API key set as environment variable on Render

### Docs
- `docs/one-pager.md` — Full product one-pager (problem, solution, differentiation, market context)

## What's Not Done Yet
- Google Form setup for email collection
- Frontend hosting (Netlify/Vercel)
- Rate limiting on API endpoints
