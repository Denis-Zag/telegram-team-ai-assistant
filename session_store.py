"""SQLite-хранилище сессий прослушивания и порядка сообщений."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SessionStatus:
    """Текущее состояние прослушивания одного Telegram-чата."""

    chat_id: int
    session_id: str | None
    is_active: bool
    started_at: str | None
    message_count: int
    document_count: int


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """Сообщение, сохранённое в хронологическом порядке."""

    chat_id: int
    message_id: int
    session_id: str
    user_id: int
    author: str
    text: str
    created_at: str


class SessionStore:
    """Управляет сессиями без общего SQLite-соединения между потоками."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    stopped_at TEXT,
                    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1))
                );

                CREATE UNIQUE INDEX IF NOT EXISTS one_active_session_per_chat
                ON sessions(chat_id)
                WHERE is_active = 1;

                CREATE INDEX IF NOT EXISTS sessions_chat_started_idx
                ON sessions(chat_id, started_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    author TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (chat_id, message_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS messages_session_created_idx
                ON messages(session_id, created_at, message_id);

                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS document_chunks_session_idx
                ON document_chunks(session_id, document_id);
                """
            )

    def start_session(self, chat_id: int) -> tuple[str, bool]:
        """Создать сессию или вернуть уже активную."""

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE chat_id = ? AND is_active = 1
                """,
                (chat_id,),
            ).fetchone()
            if row:
                return str(row["session_id"]), False

            session_id = uuid.uuid4().hex
            started_at = datetime.now(UTC).isoformat(timespec="seconds")
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, chat_id, started_at, is_active
                ) VALUES (?, ?, ?, 1)
                """,
                (session_id, chat_id, started_at),
            )
            return session_id, True

    def stop_session(self, chat_id: int) -> str | None:
        """Остановить активную сессию и вернуть её идентификатор."""

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE chat_id = ? AND is_active = 1
                """,
                (chat_id,),
            ).fetchone()
            if not row:
                return None

            session_id = str(row["session_id"])
            connection.execute(
                """
                UPDATE sessions
                SET is_active = 0, stopped_at = ?
                WHERE session_id = ?
                """,
                (datetime.now(UTC).isoformat(timespec="seconds"), session_id),
            )
            return session_id

    def active_session_id(self, chat_id: int) -> str | None:
        """Вернуть активную сессию чата."""

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE chat_id = ? AND is_active = 1
                """,
                (chat_id,),
            ).fetchone()
        return str(row["session_id"]) if row else None

    def latest_session_id(self, chat_id: int) -> str | None:
        """Вернуть активную или последнюю завершённую сессию чата."""

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE chat_id = ?
                ORDER BY is_active DESC, started_at DESC
                LIMIT 1
                """,
                (chat_id,),
            ).fetchone()
        return str(row["session_id"]) if row else None

    def add_message(self, message: StoredMessage) -> bool:
        """Сохранить сообщение один раз и сообщить, была ли вставка."""

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO messages (
                    chat_id,
                    message_id,
                    session_id,
                    user_id,
                    author,
                    text,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.chat_id,
                    message.message_id,
                    message.session_id,
                    message.user_id,
                    message.author,
                    message.text,
                    message.created_at,
                ),
            )
            return cursor.rowcount == 1

    def add_document_chunks(
        self,
        *,
        chat_id: int,
        session_id: str,
        document_id: str,
        file_name: str,
        created_at: str,
        chunk_ids: list[str],
    ) -> int:
        """Запомнить Pinecone-идентификаторы чанков для безопасной очистки."""

        if not chunk_ids:
            return 0
        rows = [
            (
                chunk_id,
                document_id,
                chat_id,
                session_id,
                file_name,
                created_at,
            )
            for chunk_id in chunk_ids
        ]
        with closing(self._connect()) as connection, connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO document_chunks (
                    chunk_id,
                    document_id,
                    chat_id,
                    session_id,
                    file_name,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[StoredMessage]:
        """Вернуть последние сообщения сессии в исходном порядке."""

        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT *
                FROM (
                    SELECT
                        chat_id,
                        message_id,
                        session_id,
                        user_id,
                        author,
                        text,
                        created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC, message_id DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, message_id ASC
                """,
                (session_id, limit),
            ).fetchall()

        return [StoredMessage(**dict(row)) for row in rows]

    def status(self, chat_id: int) -> SessionStatus:
        """Получить состояние последней сессии и число её сообщений."""

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT
                    s.chat_id,
                    s.session_id,
                    s.started_at,
                    s.is_active,
                    (
                        SELECT COUNT(*)
                        FROM messages AS m
                        WHERE m.session_id = s.session_id
                    ) AS message_count,
                    (
                        SELECT COUNT(DISTINCT d.document_id)
                        FROM document_chunks AS d
                        WHERE d.session_id = s.session_id
                    ) AS document_count
                FROM sessions AS s
                WHERE s.chat_id = ?
                ORDER BY s.is_active DESC, s.started_at DESC
                LIMIT 1
                """,
                (chat_id,),
            ).fetchone()

        if not row:
            return SessionStatus(chat_id, None, False, None, 0, 0)
        return SessionStatus(
            chat_id=int(row["chat_id"]),
            session_id=str(row["session_id"]),
            is_active=bool(row["is_active"]),
            started_at=str(row["started_at"]),
            message_count=int(row["message_count"]),
            document_count=int(row["document_count"]),
        )

    def document_ids_for_chat(self, chat_id: int) -> list[str]:
        """Вернуть идентификаторы Pinecone-документов выбранного чата."""

        with closing(self._connect()) as connection, connection:
            message_rows = connection.execute(
                """
                SELECT message_id
                FROM messages
                WHERE chat_id = ?
                ORDER BY message_id
                """,
                (chat_id,),
            ).fetchall()
            document_rows = connection.execute(
                """
                SELECT chunk_id
                FROM document_chunks
                WHERE chat_id = ?
                ORDER BY chunk_id
                """,
                (chat_id,),
            ).fetchall()
        message_ids = [
            f"tg:{chat_id}:{int(row['message_id'])}"
            for row in message_rows
        ]
        chunk_ids = [str(row["chunk_id"]) for row in document_rows]
        return message_ids + chunk_ids

    def reset_chat(self, chat_id: int) -> int:
        """Удалить только сессии и сообщения выбранного чата."""

        with closing(self._connect()) as connection, connection:
            count_row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages
                WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
            connection.execute(
                "DELETE FROM sessions WHERE chat_id = ?",
                (chat_id,),
            )
        return int(count_row["count"])
