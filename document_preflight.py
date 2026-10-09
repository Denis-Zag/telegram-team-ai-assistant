"""Живая проверка Docling, ProxyAPI и Pinecone на локальном документе."""

from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

from config import ConfigurationError, Settings
from document_processor import DocumentProcessor
from pipeline_manager import PipelineManager
from preflight import redact_error


DEFAULT_SAMPLE = (
    Path(__file__).resolve().parent
    / "samples"
    / "Auto_Service_AI_Triage_Overview.pdf"
)


def parse_args() -> argparse.Namespace:
    """Прочитать путь к PDF/DOCX без запуска Telegram polling."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "document",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="Локальный PDF или DOCX",
    )
    return parser.parse_args()


def main() -> int:
    """Проверить полный документный контур и удалить тестовые чанки."""

    args = parse_args()
    document_path = args.document.resolve()
    if not document_path.is_file():
        print(f"[ОШИБКА] Документ не найден: {document_path}")
        return 1

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
    marker = uuid.uuid4().hex
    chat_id = -9_000_000_002
    session_id = f"document-preflight-{marker}"
    chunk_ids: list[str] = []
    manager: PipelineManager | None = None

    try:
        processed = DocumentProcessor().process_bytes(
            content=document_path.read_bytes(),
            file_name=document_path.name,
            chat_id=chat_id,
            session_id=session_id,
            telegram_message_id=1,
            author="Участник проверки",
            created_at="2026-10-09T12:00:00+03:00",
        )
        pages = sorted(
            {
                str(chunk.meta.get("page_numbers", ""))
                for chunk in processed.chunks
            }
        )
        print(
            "[OK] Docling: документ разобран, "
            f"фрагментов={len(processed.chunks)}, страниц={','.join(pages)}"
        )

        manager = PipelineManager(settings)
        chunk_ids = [str(chunk.id) for chunk in processed.chunks]
        written = manager.index_documents(processed.chunks)
        print(
            "[OK] ProxyAPI Embeddings + Pinecone: "
            f"записано фрагментов={written}"
        )

        summary = manager.summarize_document(
            file_name=processed.file_name,
            document_text=processed.summary_text,
        )
        if not summary.strip():
            raise RuntimeError("Модель вернула пустое резюме документа.")
        print("[OK] ProxyAPI: краткое резюме документа получено")

        answer = manager.answer_question(
            chat_id=chat_id,
            session_id=session_id,
            question=(
                "Как формируется checkpoint ID для изоляции памяти? "
                "Назови оба исходных идентификатора."
            ),
        )
        answer_normalized = re.sub(
            r"[^a-zа-яё0-9_]+",
            "",
            answer.casefold(),
        )
        required_terms = ("client_id", "thread_id", "sha256")
        if not all(term in answer_normalized for term in required_terms):
            raise RuntimeError(
                "RAG-ответ не содержит контрольные факты документа."
            )
        print("[OK] Pinecone retrieval + ProxyAPI Chat: факты подтверждены")
        return 0
    except Exception as exc:
        print(f"[ОШИБКА] {redact_error(exc, secrets)}")
        return 1
    finally:
        if manager is not None and chunk_ids:
            try:
                manager.delete_documents(chunk_ids)
                print("[OK] Очистка: тестовые чанки документа удалены")
            except Exception as exc:
                print(
                    "[ПРЕДУПРЕЖДЕНИЕ] Чанки не удалены: "
                    f"{redact_error(exc, secrets)}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
