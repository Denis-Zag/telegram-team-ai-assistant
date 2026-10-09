"""Telegram-бот, который слушает командное обсуждение и отвечает по нему."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import telebot
from telebot import types

from config import ConfigurationError, Settings
from document_processor import (
    DocumentProcessor,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    UnsupportedDocumentError,
)
from pipeline_manager import PipelineManager
from session_store import SessionStore, StoredMessage
from text_utils import (
    build_stored_content,
    clean_question,
    display_name,
    is_addressed_to_bot,
)


LOGGER = logging.getLogger(__name__)
GROUP_TYPES = {"group", "supergroup"}
MAX_TELEGRAM_MESSAGE_LENGTH = 3900


def send_long_message(
    bot: telebot.TeleBot,
    chat_id: int,
    text: str,
    *,
    reply_to_message_id: int | None = None,
) -> None:
    """Отправить длинный ответ несколькими безопасными частями."""

    remaining = text.strip()
    first_part = True
    while remaining:
        if len(remaining) <= MAX_TELEGRAM_MESSAGE_LENGTH:
            part, remaining = remaining, ""
        else:
            boundary = remaining.rfind("\n", 0, MAX_TELEGRAM_MESSAGE_LENGTH)
            if boundary < MAX_TELEGRAM_MESSAGE_LENGTH // 2:
                boundary = remaining.rfind(" ", 0, MAX_TELEGRAM_MESSAGE_LENGTH)
            if boundary < MAX_TELEGRAM_MESSAGE_LENGTH // 2:
                boundary = MAX_TELEGRAM_MESSAGE_LENGTH
            part = remaining[:boundary].strip()
            remaining = remaining[boundary:].strip()

        bot.send_message(
            chat_id,
            part,
            reply_to_message_id=(
                reply_to_message_id if first_part else None
            ),
        )
        first_part = False


def is_group_message(message: types.Message) -> bool:
    """Проверить, что команда отправлена в группе или супергруппе."""

    return message.chat.type in GROUP_TYPES


def create_bot(
    settings: Settings,
) -> tuple[telebot.TeleBot, SessionStore, PipelineManager]:
    """Создать бота и зарегистрировать обработчики без запуска polling."""

    bot = telebot.TeleBot(settings.telegram_bot_token, threaded=True)
    store = SessionStore(settings.database_path)
    pipelines = PipelineManager(settings)
    document_processor = DocumentProcessor()
    bot_identity = bot.get_me()
    bot_username = bot_identity.username or ""
    if not bot_username:
        raise RuntimeError("BotFather не вернул username Telegram-бота.")

    def require_group(message: types.Message) -> bool:
        if is_group_message(message):
            return True
        bot.reply_to(
            message,
            "Эта команда предназначена для группового чата. "
            "Добавьте меня в учебную группу и вызовите команду там.",
        )
        return False

    def answer_from_session(message: types.Message, question: str) -> None:
        session_id = store.latest_session_id(message.chat.id)
        if not session_id:
            bot.reply_to(
                message,
                "У меня пока нет записанного обсуждения. "
                "Сначала выполните /start_listening.",
            )
            return
        try:
            answer = pipelines.answer_question(
                chat_id=message.chat.id,
                session_id=session_id,
                question=question,
            )
            send_long_message(
                bot,
                message.chat.id,
                answer,
                reply_to_message_id=message.message_id,
            )
        except Exception:
            LOGGER.exception(
                "Не удалось ответить по контексту: chat_id=%s",
                message.chat.id,
            )
            bot.reply_to(
                message,
                "Не удалось получить ответ от AI-сервиса. "
                "Проверьте настройки через preflight.py.",
            )

    @bot.message_handler(commands=["start", "help"])
    def show_help(message: types.Message) -> None:
        bot.reply_to(
            message,
            "Я запоминаю новые сообщения командного обсуждения и отвечаю "
            "только по записанному контексту.\n\n"
            "Команды:\n"
            "/start_listening — начать новую запись обсуждения\n"
            "/stop_listening — остановить запись и получить итог\n"
            "/ask вопрос — задать вопрос по последней сессии\n"
            "/status — показать состояние записи\n"
            "/reset — удалить данные этой группы\n\n"
            "Во время активной записи можно отправить PDF или DOCX размером "
            f"до {settings.max_document_bytes // 1_000_000} МБ.\n\n"
            f"Во время записи можно обратиться ко мне через "
            f"@{bot_username} или фразой «Что думаешь?»."
        )

    @bot.message_handler(commands=["start_listening"])
    def start_listening(message: types.Message) -> None:
        if not require_group(message):
            return
        session_id, created = store.start_session(message.chat.id)
        if created:
            bot.reply_to(
                message,
                "Запись обсуждения начата. Я буду молча сохранять новые "
                "текстовые сообщения этой группы.",
            )
            LOGGER.info(
                "Сессия начата: chat_id=%s session_id=%s",
                message.chat.id,
                session_id,
            )
        else:
            bot.reply_to(message, "Запись в этой группе уже идёт.")

    @bot.message_handler(commands=["stop_listening"])
    def stop_listening(message: types.Message) -> None:
        if not require_group(message):
            return
        session_id = store.stop_session(message.chat.id)
        if not session_id:
            bot.reply_to(message, "Сейчас активной записи нет.")
            return

        messages = store.list_messages(
            session_id,
            limit=settings.max_summary_messages,
        )
        if not messages:
            bot.reply_to(
                message,
                "Запись остановлена. В этой сессии не было сообщений.",
            )
            return

        bot.reply_to(message, "Запись остановлена. Готовлю итог обсуждения…")
        discussion = [
            f"[{item.created_at}] {item.author}: {item.text}"
            for item in messages
        ]
        try:
            summary = pipelines.summarize(discussion)
            send_long_message(bot, message.chat.id, summary)
        except Exception:
            LOGGER.exception(
                "Не удалось подготовить итог: chat_id=%s",
                message.chat.id,
            )
            bot.send_message(
                message.chat.id,
                "Запись остановлена, но AI-сервис не смог подготовить итог. "
                "Сообщения сохранены; проверьте настройки через preflight.py.",
            )

    @bot.message_handler(commands=["status"])
    def show_status(message: types.Message) -> None:
        if not require_group(message):
            return
        status = store.status(message.chat.id)
        if not status.session_id:
            bot.reply_to(message, "В этой группе ещё не было сессий.")
            return
        state = "идёт" if status.is_active else "остановлена"
        bot.reply_to(
            message,
            f"Запись: {state}. Сообщений в последней сессии: "
            f"{status.message_count}. Документов: {status.document_count}.",
        )

    @bot.message_handler(commands=["ask"])
    def ask(message: types.Message) -> None:
        if not require_group(message):
            return
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            bot.reply_to(message, "Напишите вопрос после команды /ask.")
            return
        answer_from_session(message, parts[1].strip())

    @bot.message_handler(commands=["reset"])
    def reset(message: types.Message) -> None:
        if not require_group(message):
            return
        status_before_reset = store.status(message.chat.id)
        document_ids = store.document_ids_for_chat(message.chat.id)
        try:
            if document_ids:
                pipelines.delete_documents(document_ids)
            deleted_count = store.reset_chat(message.chat.id)
        except Exception:
            LOGGER.exception(
                "Не удалось очистить данные: chat_id=%s",
                message.chat.id,
            )
            bot.reply_to(
                message,
                "Не удалось полностью очистить данные. "
                "Локальная история сохранена, чтобы не получить "
                "частичное удаление.",
            )
            return
        bot.reply_to(
            message,
            f"Данные этой группы удалены. Сообщений: {deleted_count}. "
            f"Документов: {status_before_reset.document_count}.",
        )

    @bot.message_handler(content_types=["document"])
    def handle_document(message: types.Message) -> None:
        if not require_group(message):
            return
        if message.from_user is None or message.from_user.is_bot:
            return

        session_id = store.active_session_id(message.chat.id)
        if not session_id:
            bot.reply_to(
                message,
                "Документы принимаются только во время активной записи. "
                "Сначала выполните /start_listening.",
            )
            return

        telegram_document = message.document
        if telegram_document is None:
            return
        file_name = Path(
            telegram_document.file_name or "document"
        ).name
        if Path(file_name).suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            bot.reply_to(message, "Поддерживаются только документы PDF и DOCX.")
            return
        if (
            telegram_document.file_size
            and telegram_document.file_size > settings.max_document_bytes
        ):
            bot.reply_to(
                message,
                "Документ слишком большой. Максимальный размер: "
                f"{settings.max_document_bytes // 1_000_000} МБ.",
            )
            return

        bot.reply_to(
            message,
            "Файл получен. Docling извлекает структуру и страницы; "
            "обработка может занять несколько минут…",
        )
        author = display_name(
            username=message.from_user.username,
            first_name=message.from_user.first_name,
            user_id=message.from_user.id,
        )
        created_at = datetime.fromtimestamp(message.date, UTC)
        indexed_chunk_ids: list[str] = []

        try:
            telegram_file = bot.get_file(telegram_document.file_id)
            content = bot.download_file(telegram_file.file_path)
            if len(content) > settings.max_document_bytes:
                raise ValueError("Документ превышает допустимый размер.")
            processed = document_processor.process_bytes(
                content=content,
                file_name=file_name,
                chat_id=message.chat.id,
                session_id=session_id,
                telegram_message_id=message.message_id,
                author=author,
                created_at=created_at.isoformat(timespec="seconds"),
            )
            indexed_chunk_ids = [str(chunk.id) for chunk in processed.chunks]
            written = pipelines.index_documents(processed.chunks)
            store.add_document_chunks(
                chat_id=message.chat.id,
                session_id=session_id,
                document_id=processed.document_id,
                file_name=processed.file_name,
                created_at=created_at.isoformat(timespec="seconds"),
                chunk_ids=indexed_chunk_ids,
            )
        except UnsupportedDocumentError as exc:
            bot.reply_to(message, str(exc))
            return
        except Exception:
            LOGGER.exception(
                "Не удалось обработать документ: chat_id=%s file_name=%s",
                message.chat.id,
                file_name,
            )
            if indexed_chunk_ids:
                try:
                    pipelines.delete_documents(indexed_chunk_ids)
                except Exception:
                    LOGGER.exception(
                        "Не удалось удалить чанки после ошибки документа: "
                        "chat_id=%s",
                        message.chat.id,
                    )
            bot.reply_to(
                message,
                "Не удалось обработать документ. Проверьте формат, размер "
                "и настройки сервисов.",
            )
            return

        try:
            document_summary = pipelines.summarize_document(
                file_name=processed.file_name,
                document_text=processed.summary_text,
            )
        except Exception:
            LOGGER.exception(
                "Документ сохранён, но резюме не создано: chat_id=%s",
                message.chat.id,
            )
            document_summary = (
                "Документ проиндексирован, но краткое резюме сейчас "
                "недоступно. Вопросы по содержанию уже можно задавать."
            )

        store.add_message(
            StoredMessage(
                chat_id=message.chat.id,
                message_id=message.message_id,
                session_id=session_id,
                user_id=message.from_user.id,
                author=author,
                text=(
                    f"[Загружен документ {processed.file_name}] "
                    f"{document_summary}"
                ),
                created_at=created_at.isoformat(timespec="seconds"),
            )
        )
        send_long_message(
            bot,
            message.chat.id,
            "Документ обработан и добавлен в контекст этой сессии.\n"
            f"Файл: {processed.file_name}\n"
            f"Фрагментов: {written}\n\n"
            f"Краткое резюме: {document_summary}",
            reply_to_message_id=message.message_id,
        )

    @bot.message_handler(content_types=["text"])
    def handle_text(message: types.Message) -> None:
        text = (message.text or "").strip()
        if not text or text.startswith("/"):
            return
        if message.from_user is None or message.from_user.is_bot:
            return
        if not is_group_message(message):
            return

        addressed = is_addressed_to_bot(text, bot_username)
        session_id = store.active_session_id(message.chat.id)

        if session_id:
            author = display_name(
                username=message.from_user.username,
                first_name=message.from_user.first_name,
                user_id=message.from_user.id,
            )
            created_at = datetime.fromtimestamp(message.date, UTC)
            stored_message = StoredMessage(
                chat_id=message.chat.id,
                message_id=message.message_id,
                session_id=session_id,
                user_id=message.from_user.id,
                author=author,
                text=text,
                created_at=created_at.isoformat(timespec="seconds"),
            )
            inserted = store.add_message(stored_message)
            if inserted:
                try:
                    pipelines.index_message(
                        document_id=(
                            f"tg:{message.chat.id}:{message.message_id}"
                        ),
                        content=build_stored_content(
                            created_at=created_at,
                            author=author,
                            text=text,
                        ),
                        chat_id=message.chat.id,
                        session_id=session_id,
                        message_id=message.message_id,
                        user_id=message.from_user.id,
                        author=author,
                        created_at=stored_message.created_at,
                    )
                except Exception:
                    LOGGER.exception(
                        "Не удалось проиндексировать сообщение: "
                        "chat_id=%s message_id=%s",
                        message.chat.id,
                        message.message_id,
                    )
                    bot.reply_to(
                        message,
                        "Сообщение сохранено локально, но не попало в "
                        "семантический поиск. Проверьте preflight.py.",
                    )
                    return

        if addressed:
            answer_from_session(
                message,
                clean_question(text, bot_username),
            )

    return bot, store, pipelines


def main() -> int:
    """Проверить настройки и запустить длительный Telegram polling."""

    try:
        settings = Settings.from_env(require_secrets=True)
    except ConfigurationError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 1

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        bot, _, _ = create_bot(settings)
        LOGGER.info("Бот готов. Запускаю Telegram polling.")
        bot.infinity_polling(
            skip_pending=True,
            timeout=30,
            long_polling_timeout=30,
            allowed_updates=["message"],
        )
    except KeyboardInterrupt:
        LOGGER.info("Бот остановлен пользователем.")
    except Exception:
        LOGGER.exception("Бот завершён из-за ошибки.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
