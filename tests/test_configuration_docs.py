from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _active_env_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / ".env.example").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_extraction_model_is_documented_across_configuration_surfaces() -> None:
    env_values = _active_env_values()
    readme = (ROOT / "README.md").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()

    assert env_values["MEM01_LLM_MODEL"] == "gpt-5.6-sol"
    assert env_values["MEM01_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert "`MEM01_LLM_MODEL` | No (default `gpt-5.6-sol`)" in readme
    assert "MEM01_LLM_MODEL=gpt-5.6-sol" in readme
    assert "MEM01_LLM_MODEL: ${MEM01_LLM_MODEL:-gpt-5.6-sol}" in compose
    assert (
        "MEM01_EMBEDDING_MODEL: ${MEM01_EMBEDDING_MODEL:-text-embedding-3-small}"
        in compose
    )
    assert "OPENAI_BASE_URL" not in env_values
    assert "OPENAI_BASE_URL" not in compose
    assert "`OPENAI_BASE_URL`" not in readme
