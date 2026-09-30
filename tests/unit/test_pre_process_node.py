# CMN-C1-130 — Unit Tests: PreProcess Node

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.pre_process_node import PreProcessNode


class TestPreProcessNodeClassification:
    """BL-01: topic, intent, and retrieval-strategy classification."""

    def setup_method(self):
        self.node = PreProcessNode()

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Which L1 base agent should I inherit for a fixed pipeline?", "level2_selection"),
            ("How do I declare the State TypedDict so msgpack works?", "state_design"),
            ("How does the HITL interrupt and resume flow work?", "hitl"),
            ("What trust level should a node performing writes declare?", "security"),
            ("How do I wire the graph edges with register_nodes?", "graph_wiring"),
            ("Tell me generally about building things on this platform", "general"),
        ],
    )
    def test_topic_detection(self, query, expected):
        assert self.node.execute({"raw_query": query})["detected_topic"] == expected

    def test_security_topic_wins_over_wiring_words(self):
        """A security question routes to the rules corpus even when it names a node."""
        result = self.node.execute({"raw_query": "What trust level should this node declare for PII?"})
        assert result["detected_topic"] == "security"
        assert result["intent_type"] == "security_rule"
        assert result["retrieval_strategy"] == "security_only"

    @pytest.mark.parametrize(
        "query,intent,strategy",
        [
            ("Show me an example of a FunctionNode execute method", "code_example", "focused"),
            ("Should I choose AgentBaseGraph or AutonomousBaseGraph?", "architecture_decision", "focused"),
            ("What is the difference in the framework pipeline stages?", "concept", "broad"),
        ],
    )
    def test_intent_and_strategy(self, query, intent, strategy):
        result = self.node.execute({"raw_query": query})
        assert result["intent_type"] == intent
        assert result["retrieval_strategy"] == strategy

    def test_code_context_is_detected(self):
        result = self.node.execute({"raw_query": "Why does my def execute(self, state) return the wrong dict?"})
        assert result["has_code_context"] is True

    def test_plain_question_has_no_code_context(self):
        result = self.node.execute({"raw_query": "What is the recommended reporting pipeline shape?"})
        assert result["has_code_context"] is False

    def test_query_id_is_assigned(self):
        assert self.node.execute({"raw_query": "How do I wire the graph edges properly?"})["query_id"]


class TestPreProcessNodeSanitization:
    """BL-02: sanitization preserves code, bounds return guidance."""

    def setup_method(self):
        self.node = PreProcessNode()

    def test_script_tag_is_stripped(self):
        result = self.node.execute({"raw_query": "<script>alert(1)</script>How do I wire graph edges?"})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "alert" not in result["normalized_query"]

    def test_type_annotations_survive_sanitization(self):
        """Generic HTML stripping would corrupt the code the question is about."""
        result = self.node.execute({"raw_query": "Should execute return list[dict[str, Any]] or a plain dict?"})
        assert "list[dict[str, Any]]" in result["normalized_query"]

    def test_over_length_query_returns_input_error(self):
        result = self.node.execute({"raw_query": "a" * 1200, "max_input_length": 1000})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "too long" in result["input_error_message"]

    def test_too_short_query_returns_input_error(self):
        result = self.node.execute({"raw_query": "state?"})
        assert "too short" in result["input_error_message"]
        assert result["input_error_guidance"]

    def test_markup_only_query_is_rejected(self):
        result = self.node.execute({"raw_query": "<script>alert(1)</script>"})
        assert result["status"] == AgentStatus.ERROR.value
        assert "S-1 violation" in result["error_message"]

    def test_upstream_error_short_circuits(self):
        assert self.node.execute({"error_message": "boom"})["status"] == AgentStatus.ERROR.value


class TestPreProcessNodeS2SecurityGate:
    """TC-09: a credential pasted into the question is redacted, not rejected."""

    def setup_method(self):
        self.node = PreProcessNode()

    def test_pasted_credential_is_redacted_in_place(self, monkeypatch):
        import src.nodes.pre_process_node as module

        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        monkeypatch.setattr(
            module,
            "detect_credentials",
            lambda text: [{"start": text.index(secret), "end": text.index(secret) + len(secret)}]
            if secret in text
            else [],
        )
        gated = self.node._extra_security_gate_input({"raw_query": f'Why does api_key="{secret}" fail?'})

        assert secret not in gated["raw_query"]
        assert "[REDACTED]" in gated["raw_query"]

    def test_question_remains_answerable_after_redaction(self):
        """The rest of the config is what makes the question answerable."""
        gated = self.node._extra_security_gate_input({"raw_query": "Why does my node config fail on startup?"})
        assert "node config" in gated["raw_query"]


class TestPreProcessNodeAuditLogging:
    """TC-11 / PB-1."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.pre_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PreProcessNode().execute({"raw_query": "How do I wire the graph edges properly?"})
        assert "pre_process_completed" in events

    def test_rejection_path_emits_domain_event(self, monkeypatch):
        import src.nodes.pre_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PreProcessNode().execute({"raw_query": "state?"})
        assert "pre_process_rejected" in events
