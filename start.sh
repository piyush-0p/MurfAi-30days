#!/bin/bash
# Render.com startup script
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
