"""Load environment variables from a local .env file.

Where to put secrets:
  mem01/.env          ← preferred (gitignored)
  open-source/.env    ← also checked (parent)
  process environment ← always wins if already set

Copy the template:
  cp .env.example .env
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(*, override: bool = False) -> list[Path]:
    """Load .env files if python-dotenv is installed. Returns paths loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return []

    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / ".env",  # mem01/.env (src/mem01/env.py → mem01/)
        here.parents[3] / ".env",  # open-source/.env
        Path.cwd() / ".env",
    ]
    loaded: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        load_dotenv(path, override=override)
        loaded.append(path)
    return loaded


def require_openai_key() -> str:
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Create mem01/.env from .env.example:\n"
            "  cp .env.example .env\n"
            "  # edit .env and paste your key\n"
        )
    return key
