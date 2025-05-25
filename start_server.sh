#!/bin/bash

# Kill any existing uvicorn processes
echo "Stopping any existing uvicorn processes..."
pkill -f uvicorn

# Wait a moment
sleep 2

# Check if port 8000 is still in use
if lsof -i :8000 > /dev/null; then
    echo "Port 8000 is still in use. Trying to free it..."
    sudo fuser -k 8000/tcp
    sleep 2
fi

# Start the server in daemon mode
echo "Starting server in daemon mode..."
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > cv_scoring_api.log 2>&1 &

# Get the process ID
PID=$!
echo "Server started with PID: $PID"
echo "You can check the logs in cv_scoring_api.log"
echo "To stop the server, run: kill $PID" 