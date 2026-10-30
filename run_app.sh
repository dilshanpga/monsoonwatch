#!/usr/bin/env bash
# MonsoonWatch - start the Streamlit app on macOS / Linux.
# Run:  bash run_app.sh     (needs Python 3.12+; first run installs packages)
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
echo "Opening MonsoonWatch at http://localhost:8501 (Ctrl+C to stop)"
python -m streamlit run app/app.py
