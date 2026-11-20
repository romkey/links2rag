from collections.abc import Callable

import httpx
import pytest

from links2rag.links_client import LinksClient, LinksError

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> LinksClient:
    return LinksClient(
        base_url="http://links.local/",
        api_key="k",
        timeout_seconds=5,
        page_size=2,
        transport=httpx.MockTransport(handler),
        retry_delay=0,
    )


def test_list_things_follows_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k"
        assert request.url.params["per_page"] == "2"
        page = int(request.url.params["page"])
        things = [{"id": page * 10 + n} for n in range(2)] if page < 3 else [{"id": 99}, "junk"]
        return httpx.Response(
            200, json={"things": things, "pagination": {"next": page + 1 if page < 3 else None}}
        )

    client = _client(handler)
    assert [t["id"] for t in client.list_things()] == [10, 11, 20, 21, 99]
    client.close()


def test_get_thing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/abc"):
            return httpx.Response(200, json={"thing": {"id": 1}})
        return httpx.Response(200, json={"nope": 1})

    client = _client(handler)
    assert client.get_thing("abc") == {"id": 1}
    with pytest.raises(LinksError, match="no 'thing'"):
        client.get_thing("other")


def test_retries_then_succeeds() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("down")
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"things": [], "pagination": {}})

    assert _client(handler).list_things() == []
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401), "LINKS_API_KEY"),
        (httpx.Response(404), "returned 404"),
        (httpx.Response(503), "returned 503"),
        (httpx.Response(200, content=b"not json"), "invalid JSON"),
        (httpx.Response(200, json=[1]), "not an object"),
        (httpx.Response(200, json={"things": 1}), "no 'things' list"),
    ],
)
def test_errors(response: httpx.Response, message: str) -> None:
    with pytest.raises(LinksError, match=message):
        _client(lambda request: response).list_things()


def test_transport_error_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(LinksError, match="failed"):
        _client(handler).list_things()
