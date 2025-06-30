@echo off
cd /d %~dp0
 .\.venv\Scripts\activate.bat
uv run pyinstaller --onefile --additional-hooks-dir=./app/hooks ./app/very_simple_rag.py
