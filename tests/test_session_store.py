"""Офлайн-тесты изоляции и жизненного цикла сессий SQLite."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from session_store import SessionStore, StoredMessage


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database = Path(self.temp_directory.name) / "test.db"
        self.store = SessionStore(database)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def _message(
        self,
        *,
        chat_id: int,
        message_id: int,
        session_id: str,
        text: str,
    ) -> StoredMessage:
        return StoredMessage(
            chat_id=chat_id,
            message_id=message_id,
            session_id=session_id,
            user_id=100 + message_id,
            author=f"Участник {message_id}",
            text=text,
            created_at=f"2026-01-01T00:00:{message_id:02d}+00:00",
        )

    def test_one_active_session_per_chat(self) -> None:
        first_id, first_created = self.store.start_session(10)
        second_id, second_created = self.store.start_session(10)
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertEqual(first_id, second_id)

    def test_chats_are_isolated(self) -> None:
        session_a, _ = self.store.start_session(10)
        session_b, _ = self.store.start_session(20)
        self.store.add_message(
            self._message(
                chat_id=10,
                message_id=1,
                session_id=session_a,
                text="Чат A",
            )
        )
        self.store.add_message(
            self._message(
                chat_id=20,
                message_id=1,
                session_id=session_b,
                text="Чат B",
            )
        )
        self.assertEqual(self.store.list_messages(session_a, limit=10)[0].text, "Чат A")
        self.assertEqual(self.store.list_messages(session_b, limit=10)[0].text, "Чат B")

    def test_messages_keep_chronological_order(self) -> None:
        session_id, _ = self.store.start_session(10)
        self.store.add_message(
            self._message(
                chat_id=10,
                message_id=2,
                session_id=session_id,
                text="Второе",
            )
        )
        self.store.add_message(
            self._message(
                chat_id=10,
                message_id=1,
                session_id=session_id,
                text="Первое",
            )
        )
        messages = self.store.list_messages(session_id, limit=10)
        self.assertEqual([item.text for item in messages], ["Первое", "Второе"])

    def test_duplicate_message_is_ignored(self) -> None:
        session_id, _ = self.store.start_session(10)
        message = self._message(
            chat_id=10,
            message_id=1,
            session_id=session_id,
            text="Один раз",
        )
        self.assertTrue(self.store.add_message(message))
        self.assertFalse(self.store.add_message(message))

    def test_stop_and_reset(self) -> None:
        session_id, _ = self.store.start_session(10)
        self.store.add_message(
            self._message(
                chat_id=10,
                message_id=7,
                session_id=session_id,
                text="Удалить",
            )
        )
        self.assertEqual(self.store.stop_session(10), session_id)
        self.assertFalse(self.store.status(10).is_active)
        self.assertEqual(self.store.document_ids_for_chat(10), ["tg:10:7"])
        self.assertEqual(self.store.reset_chat(10), 1)
        self.assertIsNone(self.store.latest_session_id(10))

    def test_document_chunks_are_counted_and_removed_with_chat(self) -> None:
        session_id, _ = self.store.start_session(10)
        self.store.add_document_chunks(
            chat_id=10,
            session_id=session_id,
            document_id="document-hash",
            file_name="example.pdf",
            created_at="2026-01-01T00:00:00+00:00",
            chunk_ids=["chunk-1", "chunk-2"],
        )

        status = self.store.status(10)

        self.assertEqual(status.document_count, 1)
        self.assertEqual(
            self.store.document_ids_for_chat(10),
            ["chunk-1", "chunk-2"],
        )
        self.store.reset_chat(10)
        self.assertEqual(self.store.document_ids_for_chat(10), [])


if __name__ == "__main__":
    unittest.main()
