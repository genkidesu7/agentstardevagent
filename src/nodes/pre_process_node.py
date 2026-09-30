"""Pre-process node for AgentStarDevAgent.

Sanitizes the developer question, detects the architecture topic, classifies
intent, and selects the retrieval strategy.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event

_INJECTION_PATTERNS = [
    re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w{1,20}\s{0,4}=", re.IGNORECASE),
]

# Topic order matters: security is checked first because a security question
# routes to a different corpus, and a question can mention both a node and a
# trust level while only the security answer is correct.
_TOPIC_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "security",
        [
            "security",
            "trust level",
            "pii",
            "credential",
            "s-1",
            "s-2",
            "s-3",
            "s-4",
            "s-5",
            "jwt",
            "guardrail",
            "secret",
            "redact",
        ],
    ),
    (
        "level2_selection",
        ["base agent", "level 2", "l1 base", "which agent", "inherit", "agentbasegraph", "autonomousbasegraph"],
    ),
    ("state_design", ["state", "typeddict", "agentstate", "msgpack", "pydantic", "checkpoint", "flat"]),
    ("hitl", ["hitl", "human in the loop", "interrupt", "approve", "resume", "human review"]),
    ("graph_wiring", ["graph", "edge", "node", "register_nodes", "add_edges", "route", "pipeline", "wiring"]),
]

_CODE_INTENT_TERMS = ["how to", "example", "code", "implement", "write", "show me", "snippet", "sample"]
_DECISION_INTENT_TERMS = ["should i", "which", "recommend", "choose", "decide", "when to", "better"]

_CODE_CONTEXT_RE = re.compile(r"```|`[^`]{1,200}`|\.py\b|\bdef \w|\bclass \w|\bimport \w")

_MAX_QUERY_TOKENS = 4000


class PreProcessNode(FunctionNode):
    """Sanitize the question and classify topic, intent, and retrieval strategy."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("pre_process_skipped", {"reason": "upstream_error"}, result)
            return result

        raw_query = str(state.get("raw_query") or state.get("user_input") or "")
        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        max_input_length = int(state.get("max_input_length") or agent_config.get("max_input_length", 1000))

        if len(raw_query) > max_input_length:
            result = self._input_error(
                raw_query,
                f"The question is too long ({len(raw_query)} characters). The maximum is {max_input_length} characters.",
            )
            emit_trace_event("pre_process_rejected", {"reason": "input_too_long"}, result)
            return result

        normalized_query = self._sanitize(raw_query)
        if not normalized_query.strip():
            if raw_query.strip():
                result = {
                    "raw_query": raw_query,
                    "error_message": "S-1 violation: question is empty after sanitization",
                    "status": AgentStatus.ERROR.value,
                }
            else:
                result = self._input_error(raw_query, "No usable developer question was provided.")
            emit_trace_event("pre_process_rejected", {"reason": "empty_after_sanitization"}, result)
            return result

        if len(normalized_query) < 10:
            result = self._input_error(
                normalized_query,
                "The question is too short to identify an architecture topic.",
            )
            emit_trace_event("pre_process_rejected", {"reason": "input_too_short"}, result)
            return result

        query_id = str(uuid.uuid4())
        has_code_context = bool(_CODE_CONTEXT_RE.search(normalized_query))
        detected_topic = self._detect_topic(normalized_query)
        intent_type = self._classify_intent(normalized_query, detected_topic)
        retrieval_strategy = self._select_strategy(intent_type)

        emit_trace_event(
            "pre_process_completed",
            {
                "query_id": query_id,
                "detected_topic": detected_topic,
                "intent_type": intent_type,
                "retrieval_strategy": retrieval_strategy,
            },
            state,
        )

        return {
            "raw_query": raw_query,
            "query_id": query_id,
            "normalized_query": normalized_query,
            "detected_topic": detected_topic,
            "intent_type": intent_type,
            "retrieval_strategy": retrieval_strategy,
            "has_code_context": has_code_context,
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2: redact credentials a developer pasted into the question.

        Developers routinely paste a whole config block when asking why it does
        not work, so the credential arrives in the question itself. Redact in
        place rather than rejecting: the question is legitimate and the rest of
        the config is what makes it answerable.
        """
        raw = str(state.get("raw_query") or state.get("user_input") or "")
        if not raw:
            return state
        findings = detect_credentials(raw)
        if findings:
            for finding in reversed(findings):
                raw = raw[: finding["start"]] + "[REDACTED]" + raw[finding["end"] :]
            state["raw_query"] = raw
            state["user_input"] = raw
            emit_trace_event("credentials_redacted_from_question", {"finding_count": len(findings)}, state)
        return state

    @staticmethod
    def _input_error(raw_query: str, message: str) -> dict[str, Any]:
        return {
            "raw_query": raw_query,
            "error_message": message,
            "input_error_message": message,
            "input_error_guidance": [
                "Name the area you are asking about — the L1 base, the State schema, HITL, "
                "graph wiring, or a security rule.",
                "Paste the relevant code or config if you are debugging something specific.",
            ],
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip script and handler markup, but preserve code formatting.

        Generic HTML-tag stripping is deliberately not applied: a developer
        question legitimately contains angle brackets in type annotations such
        as `list[dict[str, Any]]` and generics, and removing them would corrupt
        the very code the question is about.
        """
        result = text
        for pattern in _INJECTION_PATTERNS:
            result = pattern.sub("", result)
        return " ".join(result.split())[:_MAX_QUERY_TOKENS]

    @staticmethod
    def _detect_topic(query: str) -> str:
        lower = query.lower()
        for topic, keywords in _TOPIC_KEYWORDS:
            if any(keyword in lower for keyword in keywords):
                return topic
        return "general"

    @staticmethod
    def _classify_intent(query: str, topic: str) -> str:
        # A security question is answered from the rules corpus regardless of
        # how it is phrased — "show me an example" of a security rule still
        # needs the rule, not a generic snippet.
        if topic == "security":
            return "security_rule"
        lower = query.lower()
        if any(term in lower for term in _CODE_INTENT_TERMS):
            return "code_example"
        if any(term in lower for term in _DECISION_INTENT_TERMS):
            return "architecture_decision"
        return "concept"

    @staticmethod
    def _select_strategy(intent_type: str) -> str:
        if intent_type == "security_rule":
            return "security_only"
        if intent_type in ("architecture_decision", "code_example"):
            return "focused"
        return "broad"
