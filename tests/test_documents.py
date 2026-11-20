from links2rag.documents import base_payload, json_hash, payload_hash, point_id, text_hash


def test_point_id_is_stable() -> None:
    assert point_id("asset", "a", 0) == point_id("asset", "a", 0)
    assert point_id("asset", "a", 0) != point_id("manual", "a", 0)


def test_hashes() -> None:
    assert text_hash("a", "b") != text_hash("ab")
    assert json_hash({"b": 1, "a": 2}) == json_hash({"a": 2, "b": 1})


def test_base_payload() -> None:
    payload = base_payload(
        source="manual",
        doc_id="d",
        chunk_index=1,
        title="T",
        text="body",
        context="ctx",
        url=None,
        run_id="r1",
        created_at=None,
        updated_at=None,
        page=3,
        links={"asset_ids": ["a1"]},
    )
    assert payload["authority"] == 2
    assert payload["tier"] == "member"
    assert payload["asset_ids"] == ["a1"]
    assert payload["location_ids"] == []
    assert payload["page"] == 3
    other = {**payload, "indexed_at": "x", "sync_run_id": "r2"}
    assert payload_hash(payload) == payload_hash(other)
    assert payload_hash(payload) != payload_hash({**payload, "title": "U"})
