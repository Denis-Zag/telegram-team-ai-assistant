"""Загрузка и проверка настроек приложения из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigurationError(ValueError):
    """Ошибка отсутствующей или некорректной конфигурации."""


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"Переменная {name} должна содержать целое число."
        ) from exc

    if value <= 0:
        raise ConfigurationError(
            f"Переменная {name} должна быть больше нуля."
        )
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Проверенные настройки Telegram, ProxyAPI, Pinecone и SQLite."""

    telegram_bot_token: str
    proxyapi_key: str
    openai_base_url: str
    openai_chat_model: str
    openai_embedding_model: str
    openai_embedding_dimensions: int
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    database_path: Path
    retrieval_top_k: int
    max_summary_messages: int
    max_document_bytes: int
    log_level: str

    @classmethod
    def from_env(
        cls,
        *,
        require_secrets: bool = True,
        env_path: Path | None = None,
    ) -> "Settings":
        """Загрузить настройки, не раскрывая секретные значения в ошибках."""

        load_dotenv(env_path or PROJECT_ROOT / ".env", override=False)

        database_value = os.getenv("DATABASE_PATH", "data/team_ai.db").strip()
        database_path = Path(database_value)
        if not database_path.is_absolute():
            database_path = PROJECT_ROOT / database_path

        settings = cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            proxyapi_key=os.getenv("PROXYAPI_KEY", "").strip(),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.proxyapi.ru/v1"
            ).strip(),
            openai_chat_model=os.getenv(
                "OPENAI_CHAT_MODEL", "openai/gpt-5-mini"
            ).strip(),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "openai/text-embedding-3-small",
            ).strip(),
            openai_embedding_dimensions=_positive_int(
                "OPENAI_EMBEDDING_DIMENSIONS", 1536
            ),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
            pinecone_index_name=os.getenv(
                "PINECONE_INDEX_NAME", "telegram-team-ai-assistant"
            ).strip(),
            pinecone_namespace=os.getenv(
                "PINECONE_NAMESPACE", "team-chat"
            ).strip(),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
            pinecone_region=os.getenv(
                "PINECONE_REGION", "us-east-1"
            ).strip(),
            database_path=database_path,
            retrieval_top_k=_positive_int("RETRIEVAL_TOP_K", 12),
            max_summary_messages=_positive_int(
                "MAX_SUMMARY_MESSAGES", 100
            ),
            max_document_bytes=_positive_int(
                "MAX_DOCUMENT_BYTES", 10_000_000
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
        settings.validate(require_secrets=require_secrets)
        return settings

    def validate(self, *, require_secrets: bool = True) -> None:
        """Проверить обязательные параметры и безопасные ограничения."""

        missing: list[str] = []
        if require_secrets:
            secret_fields = {
                "TELEGRAM_BOT_TOKEN": self.telegram_bot_token,
                "PROXYAPI_KEY": self.proxyapi_key,
                "PINECONE_API_KEY": self.pinecone_api_key,
            }
            missing.extend(
                name for name, value in secret_fields.items() if not value
            )

        required_values = {
            "OPENAI_BASE_URL": self.openai_base_url,
            "OPENAI_CHAT_MODEL": self.openai_chat_model,
            "OPENAI_EMBEDDING_MODEL": self.openai_embedding_model,
            "PINECONE_INDEX_NAME": self.pinecone_index_name,
            "PINECONE_NAMESPACE": self.pinecone_namespace,
        }
        missing.extend(
            name for name, value in required_values.items() if not value
        )

        if missing:
            unique_names = ", ".join(sorted(set(missing)))
            raise ConfigurationError(
                f"Не заполнены обязательные переменные: {unique_names}."
            )

        if not self.openai_base_url.startswith("https://"):
            raise ConfigurationError(
                "OPENAI_BASE_URL должен начинаться с https://."
            )
