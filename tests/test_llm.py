import pytest

from clustergrep.cluster import BackendError, Cluster
from clustergrep.llm import BATCH_SIZE, LLMBackend, _schema


def expand(backend, word="escape", threshold=0.4):
    return Cluster.build(word, "llm", backend.expand(word, threshold), threshold)


def test_llm_variants_use_the_common_distance_scale(monkeypatch):
    backend = LLMBackend("test-model", max_variants=5)
    monkeypatch.setattr(backend, "_request", lambda payload: {
        "variants": [
            {"term": "jailbreak", "relation": "equivalent", "reason": "direct alternative"},
            {"term": "detention", "relation": "contextual", "reason": "related setting"},
            {"term": "escape", "relation": "equivalent", "reason": "query"},
        ],
        "complete": True,
    })

    cluster = expand(backend)
    assert cluster.distances() == {"escape": 0.0, "jailbreak": 0.15}
    assert cluster.terms[1].via == "LLM equivalent: direct alternative"


def test_llm_rejects_invalid_or_overbroad_terms(monkeypatch):
    backend = LLMBackend("test-model")
    monkeypatch.setattr(backend, "_request", lambda payload: {
        "variants": [
            {"term": "", "relation": "equivalent", "reason": "bad"},
            {"term": "movement", "relation": "unknown", "reason": "invalid"},
            {"term": "breakout", "relation": "narrower", "reason": "good"},
        ],
        "complete": True,
    })

    assert expand(backend, threshold=0.5).distances() == {
        "escape": 0.0,
        "breakout": 0.25,
    }


def test_llm_requires_a_model_name():
    with pytest.raises(BackendError, match="model name"):
        LLMBackend("")


def test_llm_expands_even_when_threshold_filters_every_generated_term(monkeypatch):
    backend = LLMBackend("test-model")
    calls = []
    monkeypatch.setattr(backend, "_request", lambda payload: calls.append(payload) or {
        "variants": [{"term": "jailbreak", "relation": "equivalent", "reason": "same"}],
        "complete": True,
    })

    # Threshold filtering belongs to Cluster.build, after the opted-in LLM
    # expansion has been fetched. This matches WordNet's backend contract.
    assert expand(backend, threshold=0).distances() == {"escape": 0.0}
    assert len(calls) == 1
    assert "contextual" in calls[0]["prompt"]


def test_schema_caps_the_generated_list():
    assert _schema(7)["properties"]["variants"]["maxItems"] == 7


def test_llm_stops_when_a_follow_up_has_no_new_terms(monkeypatch):
    backend = LLMBackend("test-model", max_variants=20)
    responses = iter([
        {"variants": [{"term": "jailbreak", "relation": "equivalent", "reason": "same"}], "complete": False},
        {"variants": [{"term": "jailbreak", "relation": "equivalent", "reason": "same"}], "complete": False},
    ])
    calls = []
    monkeypatch.setattr(backend, "_request", lambda payload: calls.append(payload) or next(responses))

    assert expand(backend).distances() == {"escape": 0.0, "jailbreak": 0.15}
    assert len(calls) == 2


def test_llm_stops_at_the_user_cap_even_if_every_batch_is_incomplete(monkeypatch):
    backend = LLMBackend("test-model", max_variants=BATCH_SIZE + 1)
    calls = []

    def response(payload):
        calls.append(payload)
        start = len(calls) * BATCH_SIZE
        return {
            "variants": [
                {"term": f"term {i}", "relation": "equivalent", "reason": "new"}
                for i in range(start, start + BATCH_SIZE)
            ],
            "complete": False,
        }

    monkeypatch.setattr(backend, "_request", response)
    cluster = expand(backend, threshold=0.2)
    assert len(cluster.terms) == BATCH_SIZE + 2  # Query plus the user cap.
    assert len(calls) == 2
