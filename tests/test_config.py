"""Офлайн-тесты безопасной конфигурации проекта."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import ConfigurationError, Settings


class ConfigTests(unittest.TestCase):
    def _empty_env_file(self, directory: str) -> Path:
        path = Path(directory) / "empty.env"
        path.touch()
        return path

    def test_missing_secrets_are_reported_by_names_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ConfigurationError) as context:
                    Settings.from_env(
                        require_secrets=True,
                        env_path=self._empty_env_file(directory),
                    )
        message = str(context.exception)
        self.assertIn("TELEGRAM_BOT_TOKEN", message)
        self.assertIn("PROXYAPI_KEY", message)
        self.assertIn("PINECONE_API_KEY", message)

    def test_defaults_are_valid_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(
                    require_secrets=False,
                    env_path=self._empty_env_file(directory),
                )
        self.assertEqual(settings.openai_embedding_dimensions, 1536)
        self.assertEqual(settings.retrieval_top_k, 12)
        self.assertEqual(settings.max_document_bytes, 10_000_000)
        self.assertTrue(settings.openai_base_url.startswith("https://"))

    def test_positive_integer_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"RETRIEVAL_TOP_K": "0"},
                clear=True,
            ):
                with self.assertRaises(ConfigurationError):
                    Settings.from_env(
                        require_secrets=False,
                        env_path=self._empty_env_file(directory),
                    )


if __name__ == "__main__":
    unittest.main()
