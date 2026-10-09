"""Чистые функции подготовки сообщений и распознавания обращений к боту."""

from __future__ import annotations

import re
from datetime import datetime


WHAT_DO_YOU_THINK_RE = re.compile(
    r"\bчто\s+(?:ты\s+)?думаешь\b",
    flags=re.IGNORECASE,
)


def display_name(
    *,
    username: str | None,
    first_name: str | None,
    user_id: int,
) -> str:
    """Получить понятное имя автора без выдуманных персональных данных."""

    if username:
        return f"@{username.lstrip('@')}"
    if first_name and first_name.strip():
        return first_name.strip()
    return f"Пользователь {user_id}"


def build_stored_content(
    *,
    created_at: datetime,
    author: str,
    text: str,
) -> str:
    """Сформировать текст документа для семантического поиска."""

    timestamp = created_at.astimezone().isoformat(timespec="seconds")
    return f"[{timestamp}] {author}: {text.strip()}"


def is_addressed_to_bot(text: str, bot_username: str) -> bool:
    """Проверить упоминание бота или фразу «Что думаешь?»."""

    normalized = text.strip()
    mention = f"@{bot_username.lstrip('@')}".casefold()
    return mention in normalized.casefold() or bool(
        WHAT_DO_YOU_THINK_RE.search(normalized)
    )


def clean_question(text: str, bot_username: str) -> str:
    """Убрать упоминание бота и вернуть вопрос для поискового пайплайна."""

    mention_re = re.compile(
        re.escape(f"@{bot_username.lstrip('@')}"),
        flags=re.IGNORECASE,
    )
    question = mention_re.sub("", text).strip(" ,:;-\n\t")
    return question or "Что думаешь об обсуждении?"

