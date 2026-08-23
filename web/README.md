# web/

React + Vite frontend for the meal-risk API.

```bash
# terminal 1 — backend (fixtures now, real model later)
uv run --with fastapi --with uvicorn python contract/stub_server.py

# terminal 2 — this app
cd web && npm install && npm run dev      # http://localhost:5173
```

**Read `AGENTS.md` before writing code.** The short version:

- `src/api/schema.d.ts` is **generated** from `../contract/openapi.json`. Never
  edit it, never hand-write duplicate types. `npm run gen:api` regenerates;
  `npm run check:api` fails if it's stale; `npm run build` regenerates first so a
  stale schema cannot ship.
- Render numbers from `getMeta()` — never hardcode the cohort size, the AUC or
  the disclaimer text.
- Never the word "dangerous". Use `describeRisk()`.
- Alternatives are predictions, not dietary advice.
- Nothing personal leaves the browser; `src/lib/storage.ts` is the only place it
  lives.

`npm run typecheck` must pass before committing.
