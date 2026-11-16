# Margin

Margin is a monorepo with a React + TypeScript web app and a FastAPI backend.

## Requirements

- Bun 1.2.16 or newer
- uv
- Python 3.13
- Docker Desktop or Docker Engine with Compose

## Start locally

```sh
cp .env.example .env
bun install
uv sync --directory api
bun run db:up
bun run dev
```

The web app runs at <http://localhost:5173>. The API runs at <http://127.0.0.1:8000>; its health endpoint is `/health` and interactive API docs are at `/docs`.

Stop the local database with `bun run db:down`.

## Quality checks

```sh
bun run lint
bun run format
bun run typecheck
bun run check
```

The web workspace uses Oxlint, Prettier, and TypeScript. The API uses Ruff for linting and formatting, and ty for type checking. Bun manages the JavaScript workspace; uv manages Python dependencies and the `api/uv.lock` lockfile.
