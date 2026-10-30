#!/usr/bin/env bash
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() {
  printf '\n============================================================\n'
  printf '  SETUP OR EXECUTION FAILED\n  Review the error messages above.\n'
  printf '============================================================\n' >&2
  exit 1
}
trap fail ERR

printf '============================================================\n'
printf '  MULTIMODAL LUNG DISEASE CLASSIFIER\n'
printf '  Environment Setup and GUI Launcher\n'
printf '============================================================\n\n'

printf '[1/5] Checking Python installation...\n'
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  printf '[ERROR] Python was not found.\n' >&2
  exit 1
fi
"$PYTHON" --version
printf '[OK] Python available.\n\n'

printf '[2/5] Checking virtual environment...\n'
if [[ ! -f .venv/bin/python ]]; then
  printf '[INFO] Creating virtual environment...\n'
  "$PYTHON" -m venv .venv
else
  printf '[OK] Existing virtual environment found.\n'
fi
printf '\n'

printf '[3/5] Activating virtual environment...\n'
# shellcheck disable=SC1091
source .venv/bin/activate
printf '[OK] Virtual environment activated.\n'
python -c "import sys; print('Python:', sys.executable)"
printf '\n'

printf '[4/5] Installing required dependencies...\n'
if [[ ! -f requirements.txt ]]; then
  printf '[ERROR] requirements.txt not found.\n' >&2
  exit 1
fi
python -m pip install -r requirements.txt
printf '[OK] Dependencies installed.\n\n'

printf '[5/5] Starting Gradio application...\n'
if [[ ! -f GUI/GUI_demo.py ]]; then
  printf '[ERROR] GUI/GUI_demo.py not found.\n' >&2
  exit 1
fi
printf '[INFO] Initializing application...\n'
printf '[INFO] Default local URL: http://127.0.0.1:7860 (actual URL may differ)\n'
printf '[INFO] Press CTRL+C to stop the server.\n\n'
python GUI/GUI_demo.py

printf '\n============================================================\n'
printf '  APPLICATION STOPPED\n'
printf '============================================================\n'
