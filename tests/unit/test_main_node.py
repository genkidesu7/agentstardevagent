# CMN-C1-130 — Unit Tests: Main Node

import json

from framework.schemas.agent_status import AgentStatus
from src.nodes.main_node import MainNode


def _base_state(**overrides):
    state = {
        "query_id": "q-1",
        "normalized_query": "Which L1 base should I inherit for a fixed pipeline?",
        "detected_topic": "level2_selection",
        "intent_type": "architecture_decision",
        "retrieval_strategy": "focused",
        "top_k": 5,
        "score_threshold": 0.6,
    }
    state.update(overrides)
    return state


class _StubRetriever:
    def __init__(self, docs=None, error=None):
        self.docs = docs or []
        self.error = error
        self.calls = []

    def retrieve(self, query, filters, top_k, score_threshold):
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        if self.error:
            raise self.error
        return self.docs


class TestMainNodeRetrieval:
    """BL-03 / BL-04 / BL-05."""

    def setup_method(self):
        self.node = MainNode()

    def test_retrieval_success_returns_json_docs(self):
        retriever = _StubRetriever(docs=[{"content": "Inherit AgentBaseGraph", "source": "api_ref", "score": 0.9}])
        result = self.node.execute(_base_state(config={"configurable": {"retriever": retriever}}))

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieval_count"] == 1
        assert json.loads(result["retrieved_docs"])[0]["source"] == "api_ref"

    def test_focused_strategy_applies_the_topic_filter(self):
        retriever = _StubRetriever()
        self.node.execute(_base_state(config={"configurable": {"retriever": retriever}}))
        assert retriever.calls[0]["filters"]["topic"] == "level2_selection"

    def test_broad_strategy_applies_no_topic_filter(self):
        """A general concept question is answered better by the whole corpus."""
        retriever = _StubRetriever()
        self.node.execute(
            _base_state(
                retrieval_strategy="broad",
                detected_topic="general",
                config={"configurable": {"retriever": retriever}},
            )
        )
        assert retriever.calls[0]["filters"] == {}

    def test_security_only_strategy_pins_the_rules_source(self):
        retriever = _StubRetriever()
        self.node.execute(
            _base_state(
                detected_topic="security",
                retrieval_strategy="security_only",
                config={"configurable": {"retriever": retriever}},
            )
        )
        filters = retriever.calls[0]["filters"]
        assert filters["source"] == "security_rules"

    def test_no_retriever_returns_no_evidence(self):
        result = self.node.execute(_base_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert json.loads(result["retrieved_docs"]) == []

    def test_retrieval_failure_is_recoverable_and_hides_backend_detail(self):
        retriever = _StubRetriever(error=ConnectionError("private backend detail"))
        result = self.node.execute(_base_state(config={"configurable": {"retriever": retriever}}))

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["recoverable_error_code"] == "retrieval_unavailable"
        assert "private backend detail" not in result["error_message"]

    def test_upstream_input_error_short_circuits(self):
        assert self.node.execute({"input_error_message": "too short"})["status"] == AgentStatus.SUCCESS.value

    def test_upstream_error_short_circuits(self):
        assert self.node.execute({"error_message": "boom"})["status"] == AgentStatus.ERROR.value


class TestMainNodeS2SecurityGate:
    """TC-09: PII is masked out of the retrieval query."""

    def test_pii_is_masked_before_retrieval(self, monkeypatch):
        import src.nodes.main_node as module

        target = "dev@example.com"
        monkeypatch.setattr(
            module,
            "detect_pii",
            lambda text: [{"type": "email", "start": text.index(target), "end": text.index(target) + len(target)}],
        )
        gated = MainNode()._extra_security_gate_input({"normalized_query": f"Ask {target} about the schema"})

        assert target not in gated["normalized_query"]
        assert "[MASKED]" in gated["normalized_query"]

    def test_clean_query_is_untouched(self, monkeypatch):
        import src.nodes.main_node as module

        monkeypatch.setattr(module, "detect_pii", lambda text: [])
        original = "How do I wire the graph edges?"
        assert MainNode()._extra_security_gate_input({"normalized_query": original})["normalized_query"] == original


class TestMainNodeAuditLogging:
    """TC-11 / PB-1."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.main_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        MainNode().execute(_base_state())
        assert "main_completed" in events

    def test_retrieval_error_emits_domain_event(self, monkeypatch):
        import src.nodes.main_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        retriever = _StubRetriever(error=TimeoutError("slow"))
        MainNode().execute(_base_state(config={"configurable": {"retriever": retriever}}))
        assert "main_retrieval_error" in events
