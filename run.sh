#!/bin/bash
echo "Starting Bangla NLP Demo Website..."
echo ""

# Activate virtual environment and run
source .venv/bin/activate
.venv/bin/python backend/app.py &

sleep 3

echo "Opening browser..."
open http://localhost:8000

echo ""
echo "Website running at http://localhost:8000"
echo "Press Ctrl+C to stop."

wait
