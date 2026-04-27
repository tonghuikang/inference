#!/bin/bash

PORT=1433

# Kill any existing process on the port
PID=$(lsof -t -i :$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Killing existing process on port $PORT (PID: $PID)"
    kill $PID 2>/dev/null
    sleep 1
fi

export HF_HOME="${HF_HOME:-/srv/vllm/hf}"
export PYTHONPATH=src
# No auth by default — set VLLM_API_KEY in the environment to require a Bearer token.
unset VLLM_API_KEY

echo "Starting server on http://localhost:$PORT/ (observer at /observer/)"
nohup uv run python -m inference.server \
    --model Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port $PORT \
    --num-kv-blocks 1024 \
    --block-size 64 \
    --kv-log ./kv_observer.log >./serve.log 2>&1 &
echo "Server running in background (PID: $!)"
