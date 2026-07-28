# Convenience wrapper for Unix/CI. `make` is NOT required to work on this repo --
# scripts/check.py is the real entry point and is what the Windows dev flow uses:
#
#     .venv\Scripts\python.exe scripts\check.py
#
# Everything here just forwards to that script so there is one implementation.

PYTHON ?= .venv/bin/python

.PHONY: help check fix test lint format

help:
	@echo "make check   - ruff check + ruff format --check + pytest (the CI task)"
	@echo "make fix     - ruff check --fix + ruff format + pytest"
	@echo "make lint    - ruff check only"
	@echo "make format  - ruff format only"
	@echo "make test    - pytest only"
	@echo ""
	@echo "Windows/PowerShell (no make): .venv\\\\Scripts\\\\python.exe scripts\\\\check.py"

check:
	$(PYTHON) scripts/check.py

fix:
	$(PYTHON) scripts/check.py --fix

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest
