# Drift — Website & Backend

Landing page, product docs, and API backend for Drift.

## Structure

```
Drift-Website/
├── frontend/         # Landing page (static HTML)
│   └── index.html
├── backend/          # FastAPI backend (Python)
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
└── docs/             # Product docs
    └── one-pager.md
```

## Frontend

Static HTML — open `frontend/index.html` or host anywhere.

## Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI key
uvicorn main:app --reload
```

API runs at `http://localhost:8000`. Endpoints:

- `POST /parse-dump` — AI-powered brain dump parsing
- `POST /daily-focus` — Daily focus recommendation
- `GET /health` — Health check

Works without an OpenAI key (uses simple fallback logic).
