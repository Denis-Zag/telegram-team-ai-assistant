"""Обязательные и документные Haystack-пайплайны командного бота."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack_integrations.document_stores.pinecone import (
    PineconeDocumentStore,
)

from config import Settings


LOGGER = logging.getLogger(__name__)


QUESTION_TEMPLATE = [
    ChatMessage.from_system(
        """
Ты — помощник команды. Отвечай только на основе найденных сообщений и
документов текущего Telegram-чата и текущей сессии. Учитывай авторов, время,
названия файлов и страницы.
Если контекста недостаточно, прямо скажи об этом. Не придумывай факты.
Для фактов из документа укажи название файла и страницу, если она известна.
Отвечай по-русски, кратко и по существу.
        """.strip()
    ),
    ChatMessage.from_user(
        """
Контекст обсуждения:
{% for document in documents %}
{% if document.meta.type == "group_document" %}
- [Документ: {{ document.meta.file_name }}{% if document.meta.page_numbers %}, стр. {{ document.meta.page_numbers }}{% endif %}] {{ document.content }}
{% else %}
- {{ document.content }}
{% endif %}
{% endfor %}

Вопрос: {{ question }}
        """.strip()
    ),
]


SUMMARY_TEMPLATE = [
    ChatMessage.from_system(
        """
Ты подводишь итог командного обсуждения. Используй только переданную
переписку. Сохрани смысл реплик и не приписывай участникам чужие слова.
Выдели тему, основные позиции, принятые решения, нерешённые вопросы и
следующие действия. Если решения нет, так и напиши.
Отвечай по-русски и не добавляй сведения вне переписки.
        """.strip()
    ),
    ChatMessage.from_user(
        """
Переписка в хронологическом порядке:
{% for message in discussion %}
- {{ message }}
{% endfor %}
        """.strip()
    ),
]


DOCUMENT_SUMMARY_TEMPLATE = [
    ChatMessage.from_system(
        """
Ты готовишь краткое первичное резюме загруженного командой документа.
Используй только переданный текст. Укажи назначение документа, ключевые факты
и важные ограничения. Не выдумывай отсутствующие сведения. Ответь по-русски
в 3-5 коротких предложениях.
        """.strip()
    ),
    ChatMessage.from_user(
        """
Файл: {{ file_name }}

Текст документа:
{{ document_text }}
        """.strip()
    ),
]


def _extract_reply_text(result: dict, component_name: str = "llm") -> str:
    """Достать текст ответа из совместимых форматов Haystack 3.x."""

    component_result = result.get(component_name, result)
    replies = component_result.get("replies") or []
    if replies:
        text = getattr(replies[0], "text", None)
        if text:
            return str(text).strip()

    last_message = component_result.get("last_message")
    if last_message is not None:
        text = getattr(last_message, "text", None)
        if text:
            return str(text).strip()

    messages = component_result.get("messages") or []
    if messages:
        text = getattr(messages[-1], "text", None)
        if text:
            return str(text).strip()

    raise RuntimeError("Haystack не вернул текстовый ответ модели.")


class PipelineManager:
    """Создаёт компоненты один раз и безопасно выполняет пайплайны."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._run_lock = threading.Lock()
        self.document_store = PineconeDocumentStore(
            index=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            dimension=settings.openai_embedding_dimensions,
            metric="cosine",
            api_key=Secret.from_token(settings.pinecone_api_key),
            spec={
                "serverless": {
                    "cloud": settings.pinecone_cloud,
                    "region": settings.pinecone_region,
                }
            },
            show_progress=False,
        )
        self.indexing_pipeline = self._create_indexing_pipeline()
        self.query_pipeline = self._create_query_pipeline()
        self.summarization_pipeline = self._create_summarization_pipeline()
        self.document_summary_pipeline = self._create_document_summary_pipeline()

    def _openai_secret(self) -> Secret:
        """Передать ключ ProxyAPI в OpenAI-совместимый компонент Haystack."""

        return Secret.from_token(self.settings.proxyapi_key)

    def _create_indexing_pipeline(self) -> Pipeline:
        """Создать пайплайн: документ → embedding → Pinecone."""

        document_embedder = OpenAIDocumentEmbedder(
            api_key=self._openai_secret(),
            api_base_url=self.settings.openai_base_url,
            model=self.settings.openai_embedding_model,
            dimensions=self.settings.openai_embedding_dimensions,
            timeout=60,
            max_retries=2,
            progress_bar=False,
        )
        writer = DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.OVERWRITE,
        )

        pipeline = Pipeline()
        pipeline.add_component("document_embedder", document_embedder)
        pipeline.add_component("writer", writer)
        pipeline.connect(
            "document_embedder.documents",
            "writer.documents",
        )
        return pipeline

    def _create_query_pipeline(self) -> Pipeline:
        """Создать RAG-пайплайн поиска и ответа по переписке."""

        text_embedder = OpenAITextEmbedder(
            api_key=self._openai_secret(),
            api_base_url=self.settings.openai_base_url,
            model=self.settings.openai_embedding_model,
            dimensions=self.settings.openai_embedding_dimensions,
            timeout=60,
            max_retries=2,
        )
        retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store
        )
        prompt_builder = ChatPromptBuilder(
            template=QUESTION_TEMPLATE,
            required_variables="*",
        )
        llm = OpenAIChatGenerator(
            api_key=self._openai_secret(),
            api_base_url=self.settings.openai_base_url,
            model=self.settings.openai_chat_model,
            generation_kwargs={
                "max_completion_tokens": 3_000,
            },
            timeout=60,
            max_retries=2,
        )

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", llm)
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        return pipeline

    def _create_summarization_pipeline(self) -> Pipeline:
        """Создать пайплайн полного резюме упорядоченной переписки."""

        prompt_builder = ChatPromptBuilder(
            template=SUMMARY_TEMPLATE,
            required_variables="*",
        )
        llm = OpenAIChatGenerator(
            api_key=self._openai_secret(),
            api_base_url=self.settings.openai_base_url,
            model=self.settings.openai_chat_model,
            generation_kwargs={
                "max_completion_tokens": 3_000,
            },
            timeout=60,
            max_retries=2,
        )

        pipeline = Pipeline()
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", llm)
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        return pipeline

    def _create_document_summary_pipeline(self) -> Pipeline:
        """Создать дополнительный пайплайн краткого резюме документа."""

        prompt_builder = ChatPromptBuilder(
            template=DOCUMENT_SUMMARY_TEMPLATE,
            required_variables="*",
        )
        llm = OpenAIChatGenerator(
            api_key=self._openai_secret(),
            api_base_url=self.settings.openai_base_url,
            model=self.settings.openai_chat_model,
            generation_kwargs={"max_completion_tokens": 1_500},
            timeout=60,
            max_retries=2,
        )

        pipeline = Pipeline()
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", llm)
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        return pipeline

    @staticmethod
    def session_filter(chat_id: int, session_id: str) -> dict:
        """Ограничить поиск одним чатом и одной сессией."""

        return {
            "operator": "AND",
            "conditions": [
                {
                    "field": "meta.chat_id",
                    "operator": "==",
                    "value": str(chat_id),
                },
                {
                    "field": "meta.session_id",
                    "operator": "==",
                    "value": session_id,
                },
            ],
        }

    def index_message(
        self,
        *,
        document_id: str,
        content: str,
        chat_id: int,
        session_id: str,
        message_id: int,
        user_id: int,
        author: str,
        created_at: str,
        extra_meta: dict | None = None,
    ) -> str:
        """Создать embedding и записать одно сообщение в Pinecone."""

        meta = {
            "type": "chat_message",
            "chat_id": str(chat_id),
            "session_id": session_id,
            "message_id": message_id,
            "user_id": str(user_id),
            "author": author,
            "created_at": created_at,
        }
        if extra_meta:
            meta.update(extra_meta)

        document = Document(id=document_id, content=content, meta=meta)
        with self._run_lock:
            result = self.indexing_pipeline.run(
                {"document_embedder": {"documents": [document]}}
            )
        if int(result["writer"]["documents_written"]) != 1:
            raise RuntimeError("Сообщение не было записано в Pinecone.")
        LOGGER.info(
            "Сообщение проиндексировано: chat_id=%s message_id=%s",
            chat_id,
            message_id,
        )
        return document.id

    def index_documents(self, documents: Sequence[Document]) -> int:
        """Создать embeddings и записать чанки загруженного документа."""

        prepared = list(documents)
        if not prepared:
            raise ValueError("Нельзя индексировать пустой список чанков.")
        with self._run_lock:
            result = self.indexing_pipeline.run(
                {"document_embedder": {"documents": prepared}}
            )
        written = int(result["writer"]["documents_written"])
        if written != len(prepared):
            raise RuntimeError(
                "В Pinecone записаны не все фрагменты документа."
            )
        return written

    def answer_question(
        self,
        *,
        chat_id: int,
        session_id: str,
        question: str,
    ) -> str:
        """Найти контекст одной сессии и ответить на вопрос."""

        with self._run_lock:
            result = self.query_pipeline.run(
                {
                    "text_embedder": {"text": question},
                    "retriever": {
                        "filters": self.session_filter(chat_id, session_id),
                        "top_k": self.settings.retrieval_top_k,
                    },
                    "prompt_builder": {"question": question},
                }
            )
        return _extract_reply_text(result)

    def summarize(self, discussion: Sequence[str]) -> str:
        """Подвести итог полной последовательности сообщений."""

        if not discussion:
            raise ValueError("Нельзя резюмировать пустую переписку.")

        with self._run_lock:
            result = self.summarization_pipeline.run(
                {"prompt_builder": {"discussion": list(discussion)}}
            )
        return _extract_reply_text(result)

    def summarize_document(self, *, file_name: str, document_text: str) -> str:
        """Подготовить краткое резюме загруженного файла."""

        if not document_text.strip():
            raise ValueError("Нельзя резюмировать документ без текста.")
        with self._run_lock:
            result = self.document_summary_pipeline.run(
                {
                    "prompt_builder": {
                        "file_name": file_name,
                        "document_text": document_text,
                    }
                }
            )
        return _extract_reply_text(result)

    def delete_documents(self, document_ids: list[str]) -> None:
        """Удалить сообщения, чанки файлов или служебные документы."""

        if document_ids:
            self.document_store.delete_documents(document_ids)
