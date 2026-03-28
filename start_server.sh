#!/bin/bash

echo "Starting server..."

# Activate main environment
source venv/bin/activate

echo "Using Python: $(which python)"

# Start FastAPI app
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
