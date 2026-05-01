from __future__ import annotations

from pathlib import Path
import os


def load_env(path: str | None = None, override: bool = False) -> Path | None:
    """Load simple KEY=VALUE pairs from .env without exposing secrets.

    The project intentionally avoids a dependency on python-dotenv. This loader
    supports comments, blank lines, optional `export`, and quoted values.
    """

    env_path = Path(path) if path else find_env_file()
    if env_path is None or not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_inline_comment(value.strip())
        value = _unquote(value)
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def find_env_file(start: str | None = None) -> Path | None:
    current = Path(start or os.getcwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    package_root_candidate = Path(__file__).resolve().parents[1] / ".env"
    return package_root_candidate if package_root_candidate.exists() else None


def has_openai_key() -> bool:
    load_env()
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_2"))


def openai_key_source() -> str | None:
    load_env()
    if os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY"
    if os.environ.get("OPENAI_API_KEY_2"):
        return "OPENAI_API_KEY_2"
    return None


def _strip_inline_comment(value: str) -> str:
    if not value or value[0] in {"'", '"'}:
        return value
    marker = value.find(" #")
    return value[:marker].rstrip() if marker >= 0 else value


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
