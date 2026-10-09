"""Безопасная живая проверка Telegram, ProxyAPI и Pinecone без polling."""

from __future__ import annotations

import uuid
import time

import telebot

from config import ConfigurationError, Settings
from pipeline_manager import PipelineManager


def redact_error(error: Exception, secrets: list[str]) -> str:
    """Скрыть известные секреты в диагностическом сообщении."""

    text = str(error)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return f"{type(error).__name__}: {text}"


def main() -> int:
    """Выполнить реальные запросы и удалить тестовый документ."""

    try:
        settings = Settings.from_env(require_secrets=True)
    except ConfigurationError as exc:
        print(f"[ОШИБКА] Конфигурация: {exc}")
        return 1

    secrets = [
        settings.telegram_bot_token,
        settings.proxyapi_key,
        settings.pinecone_api_key,
    ]
    manager: PipelineManager | None = None
    marker = uuid.uuid4().hex
    chat_id = -9_000_000_001
    session_id = f"preflight-{marker}"
    document_id = f"preflight:{marker}"

    try:
        telebot.TeleBot(settings.telegram_bot_token).get_me()
        print("[OK] Telegram Bot API: токен принят, бот найден")

        manager = PipelineManager(settings)
        manager.index_message(
            document_id=document_id,
            content=(
                "[проверка] Участник: контрольная фраза проекта — "
                f"маяк {marker}."
            ),
            chat_id=chat_id,
            session_id=session_id,
            message_id=1,
            user_id=1,
            author="Участник проверки",
            created_at="2026-01-01T00:00:00+00:00",
            extra_meta={"preflight": True},
        )
        print("[OK] ProxyAPI Embeddings + Pinecone: документ записан")

        answer = ""
        for attempt in range(3):
            answer = manager.answer_question(
                chat_id=chat_id,
                session_id=session_id,
                question=(
                    "Верни только значение после слова «маяк» "
                    "в контрольном сообщении."
                ),
            )
            if marker in answer:
                break
            if attempt < 2:
                time.sleep(2)
        if marker not in answer:
            raise RuntimeError(
                "Ответ не содержит уникальный маркер из Pinecone-контекста."
            )
        print("[OK] Pinecone retrieval + ProxyAPI Chat: ответ получен")

        summary = manager.summarize(
            ["[проверка] Участник: тестовая сессия успешно создана."]
        )
        if not summary.strip():
            raise RuntimeError("Модель вернула пустой итог.")
        print("[OK] Пайплайн итогов: ответ получен")
        return 0
    except Exception as exc:
        print(f"[ОШИБКА] {redact_error(exc, secrets)}")
        return 1
    finally:
        if manager is not None:
            try:
                manager.delete_documents([document_id])
                print("[OK] Очистка: тестовый документ удалён")
            except Exception as exc:
                print(
                    "[ПРЕДУПРЕЖДЕНИЕ] Тестовый документ не удалён: "
                    f"{redact_error(exc, secrets)}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
