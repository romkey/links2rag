import json
from typing import Any

import httpx
import pytest

from links2rag.llm import LLM, LLMError, input_hash


class DictCache:
    def __init__(self) -> None:
        # Any: cached JSON.
        self.data: dict[tuple[str, str, str], Any] = {}

    # Any: cached JSON.
    def llm_cache_get(self, task: str, model: str, key: str) -> Any:
        return self.data.get((task, model, key))

    # Any: cached JSON.
    def llm_cache_put(self, task: str, model: str, key: str, output: Any) -> None:
        self.data[(task, model, key)] = output


def _llm(
    reply: str | httpx.Response, cache: DictCache | None = None, vision: str | None = None
) -> tuple[LLM, list[Any]]:
    # Any: the request bodies sent.
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer key"
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    llm = LLM(
        model="m",
        base_url="http://gw/v1/",
        api_key="key",
        timeout_seconds=5,
        cache=cache or DictCache(),
        vision_model=vision,
        transport=httpx.MockTransport(handler),
    )
    return llm, sent


def test_analyze_asset_filters_topics_and_caches() -> None:
    reply = 'Here: {"description": " A laser. ", "topics": ["Laser-Cutting", "nope", 3]}'
    cache = DictCache()
    llm, sent = _llm(reply, cache)
    first = llm.analyze_asset(facts="Big Laser", taxonomy=["laser-cutting", "sewing"])
    second = llm.analyze_asset(facts="Big Laser", taxonomy=["laser-cutting", "sewing"])
    assert first == second
    assert first.description == "A laser."
    assert first.topics == ["laser-cutting"]
    assert len(sent) == 1
    assert sent[0]["model"] == "m"
    assert sent[0]["temperature"] == 0
    llm.close()


def test_chunk_context_and_section() -> None:
    llm, _ = _llm("It is the safety section.")
    assert llm.chunk_context(document_title="M", document="d", chunk="c") == (
        "It is the safety section."
    )
    llm, _ = _llm('{"section": "Troubleshooting"}')
    assert llm.manual_section(heading="Problems", text="If it fails...") == "troubleshooting"
    llm, _ = _llm('{"section": "other"}')
    assert llm.manual_section(heading="", text="x") is None


def test_propose_nicknames() -> None:
    llm, sent = _llm('{"nicknames": ["the beast", "Big Laser", "x", 5, "the beast "]}')
    assert llm.propose_nicknames(name="Big Laser", facts="f", mentions=[]) == []
    assert sent == []
    assert llm.propose_nicknames(name="Big Laser", facts="f", mentions=["the beast is down"]) == [
        "the beast"
    ]


def test_confirm_link() -> None:
    llm, _ = _llm('{"match": true}')
    assert llm.confirm_link(text="t", alias="a", entity_type="location", entity_id="l") is True


def test_describe_image() -> None:
    llm, sent = _llm("A wiring diagram.")
    assert llm.describe_image(png=b"x", context="page 3") is None
    assert sent == []
    cache = DictCache()
    llm, sent = _llm("A wiring diagram.", cache, vision="vlm")
    assert llm.describe_image(png=b"x", context="page 3") == "A wiring diagram."
    assert llm.describe_image(png=b"x", context="page 3") == "A wiring diagram."
    assert len(sent) == 1
    assert sent[0]["model"] == "vlm"
    part = sent[0]["messages"][0]["content"][1]
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        "not json at all",
        "[1, 2]",
    ],
)
def test_errors(reply: str | httpx.Response) -> None:
    llm, _ = _llm(reply)
    with pytest.raises(LLMError):
        llm.confirm_link(text="t", alias="a", entity_type="x", entity_id="y")


def test_input_hash() -> None:
    assert input_hash("a", "b") != input_hash("ab")
