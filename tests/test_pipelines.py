"""Структурный тест трёх Haystack-пайплайнов без API-вызовов."""

from __future__ import annotations

import unittest
from pathlib import Path

from config import Settings
from pipeline_manager import PipelineManager


class PipelineStructureTests(unittest.TestCase):
    def test_required_and_document_pipelines_are_created_offline(self) -> None:
        settings = Settings(
            telegram_bot_token="fake-telegram",
            proxyapi_key="fake-proxyapi",
            openai_base_url="https://example.invalid/v1",
            openai_chat_model="openai/gpt-5-mini",
            openai_embedding_model="openai/text-embedding-3-small",
            openai_embedding_dimensions=1536,
            pinecone_api_key="fake-pinecone",
            pinecone_index_name="offline-test",
            pinecone_namespace="offline-test",
            pinecone_cloud="aws",
            pinecone_region="us-east-1",
            database_path=Path("data/offline-test.db"),
            retrieval_top_k=12,
            max_summary_messages=100,
            max_document_bytes=10_000_000,
            log_level="INFO",
        )

        manager = PipelineManager(settings)

        self.assertIsNotNone(manager.indexing_pipeline)
        self.assertIsNotNone(manager.query_pipeline)
        self.assertIsNotNone(manager.summarization_pipeline)
        self.assertIsNotNone(manager.document_summary_pipeline)


if __name__ == "__main__":
    unittest.main()
