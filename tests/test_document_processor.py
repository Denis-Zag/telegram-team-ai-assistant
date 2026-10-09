"""Офлайн-тесты преобразования документов без запуска моделей Docling."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from document_processor import DocumentProcessor, UnsupportedDocumentError


class FakeConverter:
    def convert(self, _path: object) -> SimpleNamespace:
        return SimpleNamespace(document="docling-document")


class FakeChunker:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks

    def chunk(self, *, dl_doc: object) -> list[SimpleNamespace]:
        if dl_doc != "docling-document":
            raise AssertionError("Передан неожиданный документ")
        return self._chunks

    @staticmethod
    def contextualize(chunk: SimpleNamespace) -> str:
        return str(chunk.text)


def make_processor(chunks: list[SimpleNamespace]) -> DocumentProcessor:
    processor = object.__new__(DocumentProcessor)
    processor._converter = FakeConverter()  # type: ignore[attr-defined]
    processor._chunker = FakeChunker(chunks)  # type: ignore[attr-defined]
    processor._conversion_lock = threading.Lock()  # type: ignore[attr-defined]
    return processor


class DocumentProcessorTests(unittest.TestCase):
    def test_pdf_chunk_contains_group_session_and_page_metadata(self) -> None:
        chunk = SimpleNamespace(
            text="Checkpoint ID строится из client_id и thread_id.",
            meta=SimpleNamespace(
                doc_items=[
                    SimpleNamespace(
                        prov=[SimpleNamespace(page_no=2)]
                    )
                ],
                headings=["Память и изоляция диалогов"],
            ),
        )
        processor = make_processor([chunk])

        result = processor.process_bytes(
            content=b"pdf-content",
            file_name="../Auto_Service.pdf",
            chat_id=-1001,
            session_id="session-1",
            telegram_message_id=50,
            author="@tester",
            created_at="2026-10-09T12:00:00+00:00",
        )

        self.assertEqual(result.file_name, "Auto_Service.pdf")
        self.assertEqual(len(result.chunks), 1)
        metadata = result.chunks[0].meta
        self.assertEqual(metadata["chat_id"], "-1001")
        self.assertEqual(metadata["session_id"], "session-1")
        self.assertEqual(metadata["page_numbers"], "2")
        self.assertEqual(
            metadata["heading"],
            "Память и изоляция диалогов",
        )

    def test_unsupported_extension_is_rejected_before_conversion(self) -> None:
        processor = make_processor([])

        with self.assertRaises(UnsupportedDocumentError):
            processor.process_bytes(
                content=b"text",
                file_name="notes.txt",
                chat_id=1,
                session_id="session-1",
                telegram_message_id=1,
                author="Участник",
                created_at="2026-10-09T12:00:00+00:00",
            )

    def test_native_pdf_fragments_are_merged_by_page(self) -> None:
        def chunk(text: str, page: int) -> SimpleNamespace:
            return SimpleNamespace(
                text=text,
                meta=SimpleNamespace(
                    doc_items=[
                        SimpleNamespace(
                            prov=[SimpleNamespace(page_no=page)]
                        )
                    ],
                    headings=[],
                ),
            )

        processor = make_processor(
            [chunk("Первый блок", 1), chunk("Второй блок", 1), chunk("Третий", 2)]
        )

        result = processor.process_bytes(
            content=b"pdf-content",
            file_name="example.pdf",
            chat_id=1,
            session_id="session-1",
            telegram_message_id=1,
            author="Участник",
            created_at="2026-10-09T12:00:00+00:00",
        )

        self.assertEqual(len(result.chunks), 2)
        self.assertIn("Первый блок\nВторой блок", result.chunks[0].content)
        self.assertEqual(result.chunks[1].meta["page_numbers"], "2")


if __name__ == "__main__":
    unittest.main()
