"""Тесты чистых функций обработки Telegram-сообщений."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from text_utils import (
    build_stored_content,
    clean_question,
    display_name,
    is_addressed_to_bot,
)


class TextUtilsTests(unittest.TestCase):
    def test_display_name_prefers_username(self) -> None:
        self.assertEqual(
            display_name(username="denis", first_name="Денис", user_id=1),
            "@denis",
        )

    def test_address_by_username_is_case_insensitive(self) -> None:
        self.assertTrue(is_addressed_to_bot("@TeamBot подскажи", "teambot"))

    def test_address_by_phrase(self) -> None:
        self.assertTrue(is_addressed_to_bot("Что ты думаешь?", "teambot"))

    def test_unrelated_message_is_not_addressed(self) -> None:
        self.assertFalse(is_addressed_to_bot("Согласен с планом", "teambot"))

    def test_clean_question_removes_mention(self) -> None:
        self.assertEqual(
            clean_question("@TeamBot: какой итог?", "teambot"),
            "какой итог?",
        )

    def test_stored_content_contains_author_and_text(self) -> None:
        content = build_stored_content(
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            author="@denis",
            text="  Берём второй вариант.  ",
        )
        self.assertIn("@denis: Берём второй вариант.", content)


if __name__ == "__main__":
    unittest.main()
