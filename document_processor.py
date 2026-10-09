"""Преобразование PDF/DOCX через Docling в изолированные чанки Haystack."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from config import PROJECT_ROOT


os.environ.setdefault(
    "DOCLING_CACHE_DIR",
    str(PROJECT_ROOT / ".cache" / "docling"),
)

from docling.chunking import HierarchicalChunker  # noqa: E402
from docling.datamodel.base_models import InputFormat  # noqa: E402
from docling.document_converter import (  # noqa: E402
    DocumentConverter,
    NativePdfFormatOption,
)
from haystack import Document  # noqa: E402


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx"}
MAX_DOCUMENT_SUMMARY_CHARS = 18_000
MAX_PDF_CHUNK_CHARS = 2_500


class UnsupportedDocumentError(ValueError):
    """Формат документа не входит в согласованный учебный сценарий."""


class DocumentConversionError(RuntimeError):
    """Docling не смог получить пригодный для поиска текст."""


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """Результат временной обработки загруженного документа."""

    file_name: str
    document_id: str
    chunks: list[Document]
    summary_text: str


class DocumentProcessor:
    """Безопасно преобразует документы и удаляет временный оригинал."""

    def __init__(self) -> None:
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: NativePdfFormatOption(),
            }
        )
        self._chunker = HierarchicalChunker()
        self._conversion_lock = threading.Lock()

    def process_bytes(
        self,
        *,
        content: bytes,
        file_name: str,
        chat_id: int,
        session_id: str,
        telegram_message_id: int,
        author: str,
        created_at: str,
    ) -> ProcessedDocument:
        """Создать чанки с метаданными чата и не сохранять исходный файл."""

        safe_name = Path(file_name).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise UnsupportedDocumentError(
                "Поддерживаются только документы PDF и DOCX."
            )
        if not content:
            raise DocumentConversionError("Получен пустой документ.")

        document_id = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory(prefix="team-ai-docling-") as temp_dir:
            temp_path = Path(temp_dir) / f"source{suffix}"
            temp_path.write_bytes(content)
            with self._conversion_lock:
                converted = self._converter.convert(temp_path).document
                chunks = self._build_chunks(
                    converted=converted,
                    file_name=safe_name,
                    document_id=document_id,
                    chat_id=chat_id,
                    session_id=session_id,
                    telegram_message_id=telegram_message_id,
                    author=author,
                    created_at=created_at,
                )
                if suffix == ".pdf":
                    chunks = self._merge_pdf_chunks(
                        chunks=chunks,
                        document_id=document_id,
                        chat_id=chat_id,
                        session_id=session_id,
                    )

        if not chunks:
            raise DocumentConversionError(
                "Docling не извлёк текст из документа."
            )

        summary_text = "\n\n".join(
            chunk.content or "" for chunk in chunks
        )[:MAX_DOCUMENT_SUMMARY_CHARS]
        return ProcessedDocument(
            file_name=safe_name,
            document_id=document_id,
            chunks=chunks,
            summary_text=summary_text,
        )

    def _build_chunks(
        self,
        *,
        converted: object,
        file_name: str,
        document_id: str,
        chat_id: int,
        session_id: str,
        telegram_message_id: int,
        author: str,
        created_at: str,
    ) -> list[Document]:
        """Перенести структуру Docling в документы Haystack."""

        documents: list[Document] = []
        for chunk_index, chunk in enumerate(
            self._chunker.chunk(dl_doc=converted)
        ):
            content = self._chunker.contextualize(chunk).strip()
            if not content:
                continue

            page_numbers = sorted(
                {
                    int(provenance.page_no)
                    for item in (chunk.meta.doc_items or [])
                    for provenance in (item.prov or [])
                    if getattr(provenance, "page_no", None) is not None
                }
            )
            headings = [
                heading.strip()
                for heading in (chunk.meta.headings or [])
                if heading.strip()
            ]
            raw_chunk_id = (
                f"tgdoc:{chat_id}:{session_id}:"
                f"{document_id}:{chunk_index}"
            )
            documents.append(
                Document(
                    id=hashlib.sha256(
                        raw_chunk_id.encode("utf-8")
                    ).hexdigest(),
                    content=content,
                    meta={
                        "type": "group_document",
                        "chat_id": str(chat_id),
                        "session_id": session_id,
                        "telegram_message_id": telegram_message_id,
                        "author": author,
                        "created_at": created_at,
                        "document_id": document_id,
                        "file_name": file_name,
                        "chunk_index": chunk_index,
                        "page_numbers": ",".join(
                            str(number) for number in page_numbers
                        ),
                        "heading": " > ".join(headings),
                    },
                )
            )
        return documents

    @staticmethod
    def _merge_pdf_chunks(
        *,
        chunks: list[Document],
        document_id: str,
        chat_id: int,
        session_id: str,
    ) -> list[Document]:
        """Объединить мелкие элементы native PDF в цельные чанки страниц."""

        merged_groups: list[list[Document]] = []
        current_group: list[Document] = []
        current_length = 0
        current_pages = ""

        for chunk in chunks:
            content = (chunk.content or "").strip()
            pages = str(chunk.meta.get("page_numbers", ""))
            should_flush = bool(current_group) and (
                pages != current_pages
                or current_length + len(content) + 1 > MAX_PDF_CHUNK_CHARS
            )
            if should_flush:
                merged_groups.append(current_group)
                current_group = []
                current_length = 0

            if not current_group:
                current_pages = pages
            current_group.append(chunk)
            current_length += len(content) + 1

        if current_group:
            merged_groups.append(current_group)

        merged_documents: list[Document] = []
        for chunk_index, group in enumerate(merged_groups):
            meta = dict(group[0].meta)
            meta["chunk_index"] = chunk_index
            raw_chunk_id = (
                f"tgdoc:{chat_id}:{session_id}:"
                f"{document_id}:pdf:{chunk_index}"
            )
            merged_documents.append(
                Document(
                    id=hashlib.sha256(
                        raw_chunk_id.encode("utf-8")
                    ).hexdigest(),
                    content="\n".join(
                        (item.content or "").strip() for item in group
                    ),
                    meta=meta,
                )
            )
        return merged_documents
