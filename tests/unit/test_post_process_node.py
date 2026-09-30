# CMN-C1-130 — Unit Tests: PostProcess Node

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import PostProcessNode, _extract_cat_recommendation


class _StubLLM:
    def __init__(self, answer="## Choosing a base\n\nInherit AgentBaseGraph for a fixed pipeline."):
        self.answer = answer
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return {"content": self.answer}


class _FailingLLM:
    def complete(self, messages):
        raise RuntimeError("provider unavailable")


def _docs(count=2):
    return [
        {"content": f"Documentation chunk {index}", "source": f"api_ref-{index}", "score": 0.9}
        for index in range(count)
    ]


def _state(**overrides):
    state = {
        "query_id": "q-1",
        "normalized_query": "Which L1 base should I inherit?",
        "detected_topic": "level2_selection",
        "intent_type": "architecture_decision",
        "retrieved_docs": json.dumps(_docs()),
        "response_budget": 1800,
    }
    state.update(overrides)
    return state


class TestPostProcessSynthesis:
    """BL-06 / BL-07."""

    def test_answer_is_synthesized_from_docs(self):
        result = PostProcessNode(llm=_StubLLM()).execute(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["synthesis_mode"] == "azure_openai"
        assert "Choosing a base" in result["answer"]

    def test_provider_failure_degrades_to_evidence_only(self):
        result = PostProcessNode(llm=_FailingLLM()).execute(_state())
        assert result["synthesis_mode"] == "evidence_only"
        assert "api_ref-0" in result["answer"]

    def test_no_docs_reports_no_results(self):
        result = PostProcessNode(llm=_FailingLLM()).execute(_state(retrieved_docs="[]"))
        assert result["confidence"] == "low"
        assert "No matching documentation found" in result["answer"]

    def test_input_error_is_surfaced_as_guidance(self):
        result = PostProcessNode(llm=_StubLLM()).execute(
            {"input_error_message": "too short", "input_error_guidance": ["Name the area"]}
        )
        payload = json.loads(result["formatted_output"])
        assert payload["status"] == "input_error"

    def test_retrieval_outage_returns_actionable_answer(self):
        result = PostProcessNode(llm=_StubLLM()).execute(
            {"recoverable_error_code": "retrieval_unavailable", "trace_id": "tr-9"}
        )
        assert "temporarily unavailable" in result["answer"].lower()
        assert "tr-9" in result["answer"]

    def test_upstream_error_short_circuits(self):
        assert PostProcessNode(llm=_StubLLM()).execute({"error_message": "boom"})["status"] == AgentStatus.ERROR.value


class TestCatRecommendationExtraction:
    """BL-08: the Cat recommendation is read back out of the prose answer."""

    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("This capability is a Cat 1 template.", "Cat1"),
            ("Use Cat2 for a multi-step workflow.", "Cat2"),
            ("An autonomous loop makes this Cat 3.", "Cat3"),
            ("No category guidance in this answer.", None),
        ],
    )
    def test_single_cat_is_extracted(self, answer, expected):
        assert _extract_cat_recommendation(answer) == expected

    def test_comparison_of_several_cats_yields_no_recommendation(self):
        """Several Cats named means the answer compares them, not recommends one."""
        answer = "Cat 1 is a single capability, Cat 2 orchestrates steps, Cat 3 is autonomous."
        assert _extract_cat_recommendation(answer) is None

    def test_recommendation_reaches_the_result(self):
        result = PostProcessNode(llm=_StubLLM(answer="Use a Cat 1 template here.")).execute(_state())
        assert result["cat_recommendation"] == "Cat1"
        assert json.loads(result["formatted_output"])["cat_recommendation"] == "Cat1"


class TestPostProcessOutputShaping:
    """BL-09."""

    def test_code_snippets_are_extracted_with_language(self):
        answer = "Try:\n\n```python\nclass G(AgentBaseGraph):\n    pass\n```"
        result = PostProcessNode(llm=_StubLLM(answer=answer)).execute(_state())
        assert json.loads(result["code_snippets"])[0]["language"] == "python"

    def test_citations_are_deduplicated(self):
        duplicated = _docs(1) * 3
        result = PostProcessNode(llm=_StubLLM()).execute(_state(retrieved_docs=json.dumps(duplicated)))
        assert len(json.loads(result["citations"])) == 1

    @pytest.mark.parametrize(
        "count,score,expected",
        [(6, 0.9, "high"), (3, 0.5, "medium"), (1, 0.9, "low"), (0, 0.0, "low")],
    )
    def test_confidence_tiers(self, count, score, expected):
        docs = [dict(doc, score=score) for doc in _docs(count)] if count else []
        assert PostProcessNode._compute_confidence(docs) == expected


class TestPostProcessS3SecurityGate:
    """TC-10 / BL-10."""

    def setup_method(self):
        self.node = PostProcessNode(llm=_StubLLM())

    def _gate(self, answer, snippets=None):
        snippets = snippets if snippets is not None else []
        return self.node._extra_security_gate_output(
            {
                "answer": answer,
                "code_snippets": json.dumps(snippets),
                "formatted_output": json.dumps({"answer": answer, "code_snippets": snippets}),
            }
        )

    def test_credential_in_the_answer_is_redacted(self):
        gated = self._gate('Set api_key = "abcdef1234567890abcdef" in your config.')
        assert "abcdef1234567890abcdef" not in gated["answer"]
        assert "[REDACTED]" in gated["answer"]

    def test_json_config_form_is_redacted(self):
        gated = self._gate('Use this:\n{"api_key": "abcdef1234567890abcdef"}')
        assert "[REDACTED]" in gated["answer"]

    @pytest.mark.parametrize("placeholder", ["YOUR_API_KEY", "<API_TOKEN>", "${AZURE_OPENAI_API_KEY}"])
    def test_documented_placeholders_survive(self, placeholder):
        gated = self._gate(f"Reference the secret as {placeholder} in agent.yaml.")
        assert placeholder in gated["answer"]

    def test_framework_vocabulary_is_not_masked_as_a_person_name(self):
        answer = "Extend Function Node and declare the Trust Level in the class body."
        assert self._gate(answer)["answer"] == answer

    def test_real_person_name_is_still_masked(self):
        gated = self._gate("Extend Function Node. Ask Alice Smith for the review.")
        assert "Alice Smith" not in gated["answer"]

    def test_credentials_inside_code_snippets_are_redacted(self):
        snippets = [{"snippet_id": "snippet-1", "language": "python", "code": 'api_key = "abcdef1234567890abcdef"'}]
        gated = self._gate("See below.", snippets)
        assert "abcdef1234567890abcdef" not in gated["code_snippets"]

    def test_surviving_credential_blocks_the_output(self, monkeypatch):
        import src.nodes.post_process_node as module

        monkeypatch.setattr(module, "detect_credentials", lambda text: [{"type": "custom"}])
        with pytest.raises(RuntimeError, match="S-3 blocked"):
            self._gate("some answer")


class TestPostProcessAuditLogging:
    """TC-11 / PB-1."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.post_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PostProcessNode(llm=_StubLLM()).execute(_state())
        assert "post_process_completed" in events

    def test_fallback_path_emits_domain_event(self, monkeypatch):
        import src.nodes.post_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PostProcessNode(llm=_FailingLLM()).execute(_state())
        assert "post_process_llm_fallback" in events
