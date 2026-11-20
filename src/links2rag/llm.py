"""Ingest-time LLM tasks through the OpenAI-compatible gateway.

Every result is cached in Postgres by task, model and a hash of the input,
so re-runs and re-indexes don't repeat the calls. With no ``LLM_MODEL`` set
the pipeline builds no LLM and these tasks are skipped.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

MANUAL_SECTIONS = (
    "safety",
    "setup",
    "operation",
    "maintenance",
    "troubleshooting",
    "specifications",
    "parts",
)


class Cache(Protocol):
    # Any: cached outputs are JSON.
    def llm_cache_get(self, task: str, model: str, input_hash: str) -> Any: ...
    def llm_cache_put(self, task: str, model: str, input_hash: str, output: Any) -> None: ...


def input_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetAnalysis:
    description: str
    topics: list[str]


class LLMError(RuntimeError):
    """The gateway failed or returned something unusable."""


class LLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: int,
        cache: Cache,
        vision_model: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.vision_model = vision_model
        self._cache = cache
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    # Any: message content is a string or a list of content parts.
    def _complete(self, model: str, messages: list[dict[str, Any]]) -> str:
        """One chat completion (``POST {base_url}/chat/completions``)."""
        try:
            response = self._client.post(
                "/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError(f"chat completion failed: {e}") from e
        if not isinstance(content, str):
            raise LLMError("chat completion returned no text")
        return content.strip()

    def chat(self, system: str, user: str) -> str:
        return self._complete(
            self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
        )

    # Any: the parsed JSON object the model returned.
    def _json(self, system: str, user: str) -> dict[str, Any]:
        text = self.chat(system, user)
        match = _JSON_OBJECT.search(text)
        try:
            value = json.loads(match.group() if match else text)
        except json.JSONDecodeError as e:
            raise LLMError(f"response is not JSON: {text[:200]!r}") from e
        if not isinstance(value, dict):
            raise LLMError("response JSON is not an object")
        return value

    # Any: whatever JSON the task stores.
    def _cached(self, task: str, key: str, model: str | None = None) -> Any:
        return self._cache.llm_cache_get(task, model or self.model, key)

    def analyze_asset(self, *, facts: str, taxonomy: list[str]) -> AssetAnalysis:
        """A plain description of a piece of equipment and its taxonomy topics."""
        key = input_hash(facts, "\n".join(taxonomy))
        cached = self._cached("asset_analysis", key)
        if cached is None:
            system = (
                "You describe equipment and other assets of a hackerspace for its members' "
                'search index. Reply with one JSON object: {"description": "...", '
                '"topics": [...]}. description: one short paragraph, plain and factual, saying '
                "what the thing is, what it is for, where it is and anything a member must know "
                "before using it. Use only the facts given; never invent specifications, "
                "status or rules. topics: 0 to 4 slugs, chosen only from this list: "
                + ", ".join(taxonomy)
            )
            cached = self._json(system, facts)
            self._cache.llm_cache_put("asset_analysis", self.model, key, cached)
        allowed = set(taxonomy)
        raw_topics = cached.get("topics") or []
        topics = sorted({t.strip().lower() for t in raw_topics if isinstance(t, str)} & allowed)
        return AssetAnalysis(
            description=str(cached.get("description") or "").strip(), topics=topics
        )

    def chunk_context(self, *, document_title: str, document: str, chunk: str) -> str:
        """One or two sentences placing a chunk within its document."""
        key = input_hash(document_title, document[:8000], chunk)
        cached = self._cached("chunk_context", key)
        if cached is None:
            text = self.chat(
                "Write one or two sentences that place the excerpt within the document, "
                "for search indexing. Reply with the sentences only.",
                f"Document: {document_title}\n\n{document[:8000]}\n\nExcerpt:\n{chunk}",
            )
            cached = {"context": text}
            self._cache.llm_cache_put("chunk_context", self.model, key, cached)
        return str(cached.get("context") or "")

    def manual_section(self, *, heading: str, text: str) -> str | None:
        """Which kind of manual section this is, or None if it fits none."""
        key = input_hash(heading, text[:3000])
        cached = self._cached("manual_section", key)
        if cached is None:
            cached = self._json(
                'Reply with one JSON object: {"section": "..."}. section is one of '
                + ", ".join(MANUAL_SECTIONS)
                + ', or "other" if none fits.',
                f"Equipment manual section.\nHeading: {heading or '(none)'}\n\n{text[:3000]}",
            )
            self._cache.llm_cache_put("manual_section", self.model, key, cached)
        value = str(cached.get("section") or "").strip().lower()
        return value if value in MANUAL_SECTIONS else None

    def propose_nicknames(self, *, name: str, facts: str, mentions: list[str]) -> list[str]:
        """Nicknames members use for an asset, read from wiki and Slack excerpts."""
        if not mentions:
            return []
        excerpts = "\n---\n".join(m[:1500] for m in mentions[:8])
        key = input_hash(name, facts, excerpts)
        cached = self._cached("asset_nicknames", key)
        if cached is None:
            cached = self._json(
                'Reply with one JSON object: {"nicknames": [...]}. List the informal names '
                "the excerpts use for the asset described (for example 'the big laser'), "
                "only when an excerpt clearly uses that name for this asset. Never include "
                "people's names. An empty list is fine.",
                f"Asset: {name}\n{facts}\n\nExcerpts:\n{excerpts}",
            )
            self._cache.llm_cache_put("asset_nicknames", self.model, key, cached)
        raw = cached.get("nicknames") or []
        names = {n.strip() for n in raw if isinstance(n, str) and 3 <= len(n.strip()) <= 60}
        return sorted(names - {name})

    def confirm_link(self, *, text: str, alias: str, entity_type: str, entity_id: str) -> bool:
        """Does ``alias`` in ``text`` refer to this entity? For ambiguous linker hits."""
        key = input_hash(text[:4000], alias, entity_type, entity_id)
        cached = self._cached("confirm_link", key)
        if cached is None:
            cached = self._json(
                'Reply with one JSON object: {"match": true|false}.',
                f"Does the phrase {alias!r} in this text refer to the {entity_type} "
                f"{entity_id!r}?\n\n{text[:4000]}",
            )
            self._cache.llm_cache_put("confirm_link", self.model, key, cached)
        return bool(cached.get("match"))

    def describe_image(self, *, png: bytes, context: str) -> str | None:
        """A short description of a diagram page; None without a vision model."""
        if not self.vision_model:
            return None
        key = hashlib.sha256(png).hexdigest()
        cached = self._cached("diagram_description", key, self.vision_model)
        if cached is None:
            image = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            text = self._complete(
                self.vision_model,
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "This is a page of an equipment manual that is mostly "
                                f"a diagram ({context}). Describe in 2 to 5 plain sentences "
                                "what it shows and the labels it names, for a search index.",
                            },
                            {"type": "image_url", "image_url": {"url": image}},
                        ],
                    }
                ],
            )
            cached = {"description": text}
            self._cache.llm_cache_put("diagram_description", self.vision_model, key, cached)
        return str(cached.get("description") or "").strip() or None
