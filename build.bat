@echo off
cd /d %~dp0
pyinstaller --onefile --additional-hooks-dir=./app/hooks ./app/very_simple_rag.py
