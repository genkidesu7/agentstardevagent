"""Main node for AgentStarDevAgent.

Retrieves AgentCore documentation for the parsed question, filtered by the
detected topic and the selected retrieval strategy.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event


class MainNode(FunctionNode):
    """Retrieve documentation chunks for the developer question."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__()
        # Required scaffold injection point. Retrieval is deterministic today;
        # response synthesis consumes the client in PostProcessNode.
        self._llm = llm

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("input_error_message"):
            return {"status": AgentStatus.SUCCESS.value}

        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("main_skipped", {"reason": "upstream_error"}, result)
            return result

        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        top_k = int(state.get("top_k") or agent_config.get("top_k", 5))
        score_threshold = float(state.get("score_threshold") or agent_config.get("score_threshold", 0.6))

        retrieval_query = str(state.get("normalized_query") or state.get("raw_query") or "")

        filters: dict[str, Any] = {}
        topic = state.get("detected_topic")
        strategy = state.get("retrieval_strategy")
        # A "broad" strategy deliberately applies no topic filter: a general
        # concept question is answered better by the whole corpus than by one
        # slice of it.
        if topic and topic != "general" and strategy != "broad":
            filters["topic"] = topic
        if strategy == "security_only":
            filters["source"] = "security_rules"

        try:
            docs = self._retrieve(
                query=retrieval_query,
                filters=filters,
                top_k=top_k,
                score_threshold=score_threshold,
                config=state.get("config"),
            )
        except Exception as exc:
            error_message = f"MainNode: the documentation index is temporarily unavailable ({type(exc).__name__})"
            emit_trace_event(
                "main_retrieval_error",
                {"query_id": state.get("query_id"), "error_type": type(exc).__name__},
                state,
            )
            return {
                "retrieved_docs": "[]",
                "retrieval_count": 0,
                "error_message": error_message,
                "recoverable_error_code": "retrieval_unavailable",
                # The failure is handled by PostProcessNode so Marketplace receives
                # an actionable response instead of treating the invocation itself
                # as failed.
                "status": AgentStatus.SUCCESS.value,
            }

        retrieval_count = len(docs)
        emit_trace_event(
            "main_completed",
            {
                "query_id": state.get("query_id"),
                "retrieval_count": retrieval_count,
                "filters_applied": list(filters.keys()),
            },
            state,
        )

        return {
            "retrieved_docs": json.dumps(docs),
            "retrieval_count": retrieval_count,
            "error_message": None,
            "recoverable_error_code": None,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _retrieve(
        query: str,
        filters: dict[str, Any],
        top_k: int,
        score_threshold: float,
        config: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        configurable = (config or {}).get("configurable", {})
        retriever = configurable.get("retriever")
        if retriever is not None:
            return cast(
                list[dict[str, Any]],
                retriever.retrieve(
                    query=query,
                    filters=filters,
                    top_k=top_k,
                    score_threshold=score_threshold,
                ),
            )
        # No retriever configured: return no evidence rather than inventing
        # framework APIs. PostProcessNode turns this into a no-results answer.
        return []

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2: PII masking on normalized_query before documentation retrieval."""
        raw = state.get("normalized_query", "")
        if not raw:
            return state
        findings = detect_pii(raw)
        if findings:
            for finding in reversed(findings):
                raw = raw[: finding["start"]] + "[MASKED]" + raw[finding["end"] :]
            state["normalized_query"] = raw
            emit_trace_event("pii_masked_normalized_query", {}, state)
        return state
