import os
import tempfile
import unittest
from pathlib import Path

from rlraft.env import load_env


class EnvTests(unittest.TestCase):
    def test_load_env_reads_key_without_overriding_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=from_file\nOPENAI_MODEL=\"test-model\"\n",
                encoding="utf-8",
            )
            old_key = os.environ.get("OPENAI_API_KEY")
            old_model = os.environ.get("OPENAI_MODEL")
            try:
                os.environ["OPENAI_API_KEY"] = "already_set"
                load_env(str(env_path))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "already_set")
                self.assertEqual(os.environ["OPENAI_MODEL"], "test-model")
            finally:
                _restore_env("OPENAI_API_KEY", old_key)
                _restore_env("OPENAI_MODEL", old_model)


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
