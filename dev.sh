#!/usr/bin/env bash
# Start both servers with one command.
# Usage: bash dev.sh
# Open: http://localhost:5173

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate virtualenv if present
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
fi

echo "▶ Starting FastAPI backend on http://localhost:8000 ..."
cd "$PROJECT_ROOT"
uvicorn backend.main:app \
  --reload \
  --reload-dir backend \
  --reload-dir core \
  --reload-delay 1 \
  --port 8000 &
BACKEND_PID=$!

echo "▶ Starting React dev server on http://localhost:5173 ..."
cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

cleanup() {
  echo ""
  echo "Stopping servers..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "✓ Both servers running."
echo "  Frontend → http://localhost:5173"
echo "  API docs → http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop."

wait
