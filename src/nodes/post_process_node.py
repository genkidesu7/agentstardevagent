"""Post-process node for AgentStarDevAgent.

Synthesizes the answer from retrieved AgentCore documentation, extracts code
snippets, citations, and any Cat recommendation, and applies the S-3 output gate.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, NamedTuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security.pii_masking import mask_pii
from shared.security.credential_detector import detect_credentials
from shared.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_runtime import complete_text


def _emit_progress(message: str, metadata: dict[str, Any]) -> None:
    """Emit Marketplace progress when the platform event transport is available."""
    try:
        from shared.services.events import emitter
        from shared.services.events.types import EventType
    except (ImportError, ModuleNotFoundError):
        # Lightweight local AgentCore installations may omit Marketplace events.
        # Synthesis and its terminal fallback must remain available regardless.
        return
    emitter().emit_event(event_type=EventType.PROGRESS_UPDATE, message=message, metadata=metadata)


_SYSTEM_PROMPT = (
    "You are an AgentCore platform architect helping a developer building an agent template.\n"
    "Given a developer question and retrieved framework documentation, write a clear, "
    "accurate answer.\n\n"
    "Your response MUST:\n"
    "1. Answer the architecture question directly, naming the actual class, method, or field\n"
    "2. Include Python code in fenced blocks with a language tag when the context provides it\n"
    "3. State the recommended category explicitly as 'Cat 1', 'Cat 2', or 'Cat 3' when the "
    "question asks which category or base class applies\n"
    "4. Cite each source as: [Source: <source>]\n"
    "5. Use only the provided context -- never invent framework APIs, fields, or config keys\n"
    "6. Say so explicitly when the context is insufficient\n"
    "7. Use placeholders such as YOUR_API_KEY for secrets -- never emit a real credential\n\n"
    "Keep the answer concise and suitable for a developer reading it mid-task."
)

_NO_RESULTS_ANSWER = (
    "## No matching documentation found\n\n"
    "No AgentCore documentation matching your question was found. Please name the class, "
    "method, or concept you are asking about and try again.\n\n"
    "_Azure OpenAI synthesis was temporarily unavailable; this response is based on "
    "retrieval status only._"
)

_RETRIEVAL_UNAVAILABLE_ANSWER = (
    "## Documentation index temporarily unavailable\n\n"
    "I couldn't reach the AgentCore documentation index, so I can't safely confirm framework "
    "APIs or architecture guidance right now. Please try again shortly. If the problem "
    "continues, contact support and include the reference below."
)


class _ScanRule(NamedTuple):
    name: str
    pattern: re.Pattern[str]
    value_group: int | None


# PB-CRED-001 rule set. Every quantifier is bounded so a hostile input cannot
# force quadratic-time backtracking.
_SCAN_RULES: list[_ScanRule] = [
    _ScanRule("openai_secret_key", re.compile(r"sk-[A-Za-z0-9\-_]{20,64}"), None),
    _ScanRule("aws_access_key", re.compile(r"AKIA[A-Z0-9]{16}"), None),
    _ScanRule(
        "jwt_token",
        re.compile(r"eyJ[A-Za-z0-9\-_]{10,512}\.[A-Za-z0-9\-_]{10,512}\.[A-Za-z0-9\-_]{10,512}"),
        None,
    ),
    _ScanRule("bearer_token", re.compile(r"Bearer\s{1,3}([A-Za-z0-9\-._~+/]{20,256}={0,2})"), 1),
    _ScanRule(
        "hardcoded_password",
        re.compile(r"password[\"']?\s{0,4}[:=]\s{0,4}[\"']?([^\s\"',;)]{8,128})[\"']?", re.IGNORECASE),
        1,
    ),
    _ScanRule(
        "api_key_assignment",
        re.compile(r"api[_\-]?key[\"']?\s{0,4}[:=]\s{0,4}[\"']?([^\s\"',;)]{8,128})[\"']?", re.IGNORECASE),
        1,
    ),
    _ScanRule(
        "generic_secret_assignment",
        re.compile(
            r"(?:token|secret|auth_key|access_key|private_key)[\"']?\s{0,4}[:=]\s{0,4}"
            r"[\"']?([A-Za-z0-9+/\-_]{20,256}={0,2})[\"']?",
            re.IGNORECASE,
        ),
        1,
    ),
    _ScanRule("url_embedded_credentials", re.compile(r"https?://[^/\s@]{1,128}:([^/\s@]{4,128})@", re.IGNORECASE), 1),
    _ScanRule(
        "db_connection_string",
        re.compile(
            r"(?:postgresql|mysql|mongodb(?:\+srv)?|redis)://[^/\s@]{1,128}:([^/\s@]{4,128})@",
            re.IGNORECASE,
        ),
        1,
    ),
    _ScanRule(
        "long_base64_token",
        re.compile(
            r"(?:key|token|secret|credential)[\"']?\s{0,4}[:=]\s{0,4}[\"']?([A-Za-z0-9+/]{40,256}={0,2})[\"']?",
            re.IGNORECASE,
        ),
        1,
    ),
]

# Documented placeholders are instructional, not secrets. Redacting them would
# destroy the value of a configuration example, which is the capability's purpose.
_PLACEHOLDER_RE = re.compile(
    r"""
    (?:
        YOUR_[A-Z0-9_]{1,64}
      | <[A-Za-z_]{1,64}>
      | \{\{?[A-Za-z0-9_]{1,64}\}?\}
      | \$\{[A-Za-z0-9_]{1,64}\}
      | \$[A-Z_][A-Z0-9_]{1,63}
      | x{4,64}
      | 0{4,64}
      | (?:mock|test|fake|example|dummy|sample)[-_]
      | \.{3,}
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_REDACTED_MARKER = "[REDACTED]"

_FENCE_PATTERN = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]{0,20})[ \t]{0,8}\n(?P<code>[\s\S]{1,8000}?)```")

_LANG_ALIASES: dict[str, str] = {
    "ts": "typescript",
    "typescript": "typescript",
    "js": "javascript",
    "javascript": "javascript",
    "py": "python",
    "python": "python",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "sh": "shell",
    "bash": "shell",
    "shell": "shell",
    "": "unknown",
}

# The shared name detector favors recall: it flags any run of Title Case words as
# a possible person, so framework vocabulary ("Function Node", "Trust Level")
# matches as readily as a real name. A fixed phrase list cannot keep up, because
# the detector's span boundaries shift with surrounding words. Instead, a finding
# is treated as non-personal when every one of its capitalised tokens is framework
# vocabulary — a real name contributes at least one token that is not.
_DOMAIN_VOCABULARY = frozenset(
    {
        # Framework nouns.
        "agent",
        "agentcore",
        "anonymous",
        "autonomous",
        "base",
        "call",
        "cat",
        "checkpoint",
        "class",
        "code",
        "config",
        "context",
        "core",
        "decision",
        "design",
        "edge",
        "example",
        "external",
        "field",
        "finalize",
        "framework",
        "function",
        "gate",
        "graph",
        "guide",
        "hitl",
        "human",
        "initialize",
        "input",
        "internal",
        "invocation",
        "level",
        "loop",
        "main",
        "memory",
        "method",
        "next",
        "node",
        "output",
        "pipeline",
        "post",
        "pre",
        "process",
        "register",
        "resume",
        "review",
        "rule",
        "rules",
        "schema",
        "security",
        "selection",
        "star",
        "state",
        "step",
        "steps",
        "trust",
        "type",
        "verified",
        "wiring",
        # Imperative verbs a documentation answer opens a sentence with. The
        # detector's span absorbs the sentence-initial word, so "Extend Function
        # Node" arrives as one finding.
        "add",
        "avoid",
        "build",
        "call",
        "check",
        "compile",
        "create",
        "declare",
        "define",
        "extend",
        "implement",
        "inherit",
        "invoke",
        "keep",
        "make",
        "note",
        "override",
        "pass",
        "prefer",
        "raise",
        "return",
        "set",
        "use",
        "wire",
        "write",
    }
)

_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{0,63}")

# Cat recommendations are stated in prose, so they are read back out of the
# answer rather than requested as a separate structured field.
_CAT_RE = re.compile(r"\bCat\s{0,2}([123])\b")


def _is_domain_phrase(value: str) -> bool:
    """True when every word in a detected name is framework vocabulary."""
    tokens = _NAME_TOKEN_RE.findall(value)
    if not tokens:
        return False
    return all(token.lower() in _DOMAIN_VOCABULARY for token in tokens)


def _extract_cat_recommendation(answer: str) -> str | None:
    """Return the single Cat the answer recommends, or None when ambiguous.

    Several distinct Cat values in one answer means the answer is comparing
    them rather than recommending one, so no recommendation is recorded.
    """
    found = {match.group(1) for match in _CAT_RE.finditer(answer)}
    return f"Cat{found.pop()}" if len(found) == 1 else None


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(value))


def _redact_credentials(text: str) -> tuple[str, list[str]]:
    """Redact credential values in place; preserve documented placeholders."""
    redacted = text
    findings: list[str] = []
    for rule in _SCAN_RULES:
        for match in rule.pattern.finditer(text):
            value = match.group(rule.value_group) if rule.value_group is not None else match.group(0)
            if not value or _is_placeholder(value):
                continue
            full_match = match.group(0)
            redacted = redacted.replace(full_match, full_match.replace(value, _REDACTED_MARKER))
            # Record only the rule name — the audit trail must never carry the secret.
            findings.append(rule.name)
    return redacted, findings


class PostProcessNode(FunctionNode):
    """Generate the answer, code snippets, citations, and Cat recommendation."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm: Any | None = None,
        *,
        llm_timeout_s: float = 30.0,
        llm_max_retry: int = 3,
    ) -> None:
        super().__init__()
        # Unit-test seam only. Production resolves Azure OpenAI from the
        # invocation-scoped secrets declared by the agent manifest.
        self._llm = llm
        self._llm_timeout_s = llm_timeout_s
        self._llm_max_retry = llm_max_retry

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("input_error_message"):
            message = str(state["input_error_message"])
            guidance = state.get("input_error_guidance", [])
            return {
                "answer": message,
                "citations": "[]",
                "code_snippets": "[]",
                "confidence": "low",
                "formatted_output": json.dumps(
                    {"status": "input_error", "message": message, "guidance": guidance},
                    ensure_ascii=False,
                ),
                "status": AgentStatus.SUCCESS.value,
            }

        if state.get("recoverable_error_code") == "retrieval_unavailable":
            answer = _RETRIEVAL_UNAVAILABLE_ANSWER
            trace_id = str(state.get("trace_id") or "").strip()
            if trace_id:
                answer = f"{answer}\n\nReference: `{trace_id}`"
            emit_trace_event(
                "post_process_retrieval_unavailable",
                {"query_id": state.get("query_id"), "trace_id": trace_id},
                state,
            )
            return {
                "answer": answer,
                "citations": "[]",
                "code_snippets": "[]",
                "confidence": "low",
                "formatted_output": json.dumps(
                    {
                        "status": "temporarily_unavailable",
                        "answer": answer,
                        "citations": [],
                        "code_snippets": [],
                        "confidence": "low",
                        "retrieval_count": 0,
                    },
                    ensure_ascii=False,
                ),
                "error_message": None,
                "recoverable_error_code": None,
                "status": AgentStatus.SUCCESS.value,
            }

        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("post_process_skipped", {"reason": "upstream_error"}, result)
            return result

        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        llm_max_tokens = int(state.get("response_budget") or agent_config.get("llm_max_tokens", 1500))

        docs = self._load_json_list(state.get("retrieved_docs"))
        query = str(state.get("normalized_query") or state.get("raw_query") or "")

        if not docs:
            return self._no_results(state, query, llm_max_tokens)

        user_prompt = f"Question: {query}\n"
        if state.get("tool_name"):
            user_prompt += f"Platform: {str(state['tool_name']).upper()}\n"
        if state.get("intent"):
            user_prompt += f"Intent: {state['intent']}\n"
        user_prompt += f"\n---\nDocumentation:\n{self._build_context(docs)}"

        _emit_progress(
            "Synthesizing the architecture answer with Azure OpenAI...",
            {"stage": "llm_synthesis", "retrieval_count": len(docs)},
        )
        try:
            answer = self._generate(state, _SYSTEM_PROMPT, user_prompt, llm_max_tokens)
            synthesis_mode = "azure_openai"
            _emit_progress(
                "Azure OpenAI synthesis completed.",
                {"stage": "llm_synthesis", "status": "completed", "retrieval_count": len(docs)},
            )
        except Exception as exc:
            emit_trace_event(
                "post_process_llm_fallback",
                {"query_id": state.get("query_id"), "reason": type(exc).__name__},
                state,
            )
            answer = self._fallback_answer(docs)
            synthesis_mode = "evidence_only"
            _emit_progress(
                "Azure OpenAI synthesis is temporarily unavailable. Returning retrieved documentation only.",
                {"stage": "llm_synthesis", "status": "error", "retrieval_count": len(docs)},
            )

        code_snippets = self._extract_code_snippets(answer)
        citations = self._extract_citations(docs)
        confidence = self._compute_confidence(docs)
        cat_recommendation = _extract_cat_recommendation(answer)

        emit_trace_event(
            "post_process_completed",
            {
                "query_id": state.get("query_id"),
                "retrieval_count": len(docs),
                "citations_count": len(citations),
                "snippet_count": len(code_snippets),
                "cat_recommendation": cat_recommendation,
                "confidence": confidence,
            },
            state,
        )

        return {
            "answer": answer,
            "citations": json.dumps(citations),
            "code_snippets": json.dumps(code_snippets),
            "cat_recommendation": cat_recommendation,
            "confidence": confidence,
            "synthesis_mode": synthesis_mode,
            "formatted_output": json.dumps(
                {
                    "answer": answer,
                    "citations": citations,
                    "code_snippets": code_snippets,
                    "cat_recommendation": cat_recommendation,
                    "confidence": confidence,
                    "retrieval_count": len(docs),
                    "synthesis_mode": synthesis_mode,
                },
                ensure_ascii=False,
            ),
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _no_results(self, state: dict[str, Any], query: str, llm_max_tokens: int) -> dict[str, Any]:
        _emit_progress(
            "Generating the response with Azure OpenAI...",
            {"stage": "llm_synthesis", "retrieval_count": 0},
        )
        prompt = (
            f"Question: {query}\n\n"
            "No framework documentation was retrieved. Explain this limitation concisely and "
            "suggest which class, method, or concept the developer should name instead."
        )
        try:
            answer = self._generate(state, _SYSTEM_PROMPT, prompt, min(llm_max_tokens, 400))
            synthesis_mode = "azure_openai"
            _emit_progress(
                "Azure OpenAI synthesis completed.",
                {"stage": "llm_synthesis", "status": "completed", "retrieval_count": 0},
            )
        except Exception as exc:
            answer = _NO_RESULTS_ANSWER
            synthesis_mode = "evidence_only"
            _emit_progress(
                "Azure OpenAI synthesis is temporarily unavailable. Returning retrieval-only guidance.",
                {"stage": "llm_synthesis", "status": "error", "retrieval_count": 0},
            )
            emit_trace_event("post_process_llm_fallback", {"reason": type(exc).__name__}, state)

        emit_trace_event(
            "post_process_no_results",
            {"query_id": state.get("query_id"), "confidence": "low"},
            state,
        )
        code_snippets = self._extract_code_snippets(answer)
        return {
            "answer": answer,
            "citations": "[]",
            "code_snippets": json.dumps(code_snippets),
            "confidence": "low",
            "synthesis_mode": synthesis_mode,
            "formatted_output": json.dumps(
                {
                    "answer": answer,
                    "citations": [],
                    "code_snippets": code_snippets,
                    "confidence": "low",
                    "retrieval_count": 0,
                    "synthesis_mode": synthesis_mode,
                },
                ensure_ascii=False,
            ),
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _generate(self, state: dict[str, Any], system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        config = state.get("config")
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        injected = self._llm or configurable.get("llm")
        return complete_text(
            state,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            injected,
            max_tokens=max_tokens,
            timeout_s=self._llm_timeout_s,
            max_retry=self._llm_max_retry,
        )

    @staticmethod
    def _fallback_answer(docs: list[dict[str, Any]]) -> str:
        lines = [
            "## AgentCore documentation",
            "",
            "LLM synthesis was unavailable; the retrieved documentation follows.",
            "",
        ]
        for doc in docs[:20]:
            content = str(doc.get("content", "")).strip()
            lines.append(f"- [{doc.get('source', 'unknown')}] {content[:400]}")
        return "\n".join(lines)

    @staticmethod
    def _build_context(docs: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{index}] Source: {doc.get('source', 'unknown')}\n    {doc.get('content', '')}"
            for index, doc in enumerate(docs, 1)
        )

    @staticmethod
    def _extract_code_snippets(answer: str) -> list[dict[str, str]]:
        snippets = []
        for index, match in enumerate(_FENCE_PATTERN.finditer(answer)):
            label = match.group("lang").lower()
            snippets.append(
                {
                    "snippet_id": f"snippet-{index + 1}",
                    "language": _LANG_ALIASES.get(label, label or "unknown"),
                    "code": match.group("code").strip(),
                }
            )
        return snippets

    @staticmethod
    def _extract_citations(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        citations = []
        for doc in docs:
            source = str(doc.get("source") or "")
            if source in seen:
                continue
            seen.add(source)
            citations.append({"source": source, "score": str(doc.get("score", ""))})
        return citations

    @staticmethod
    def _compute_confidence(docs: list[dict[str, Any]]) -> str:
        if not docs:
            return "low"
        try:
            scores = [float(doc.get("score", 0.0)) for doc in docs if doc.get("score")]
            avg_score = sum(scores) / len(scores) if scores else 0.0
        except (ValueError, TypeError):
            avg_score = 0.0
        if len(docs) >= 5 and avg_score >= 0.75:
            return "high"
        if len(docs) >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _load_json_list(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, str):
            return []
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []

    def _extra_security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """S-3: PB-CRED-001 redaction, then residual PII masking.

        Redaction happens here, in the node that produces the content, so the
        framework credential scan sees clean output and the invocation ends in a
        redacted success. A downstream masking node would never be reached,
        because this node's own gate raises first.
        """
        answer = result.get("answer", "")
        if not answer:
            return result

        redacted_answer, findings = _redact_credentials(answer)
        snippets = self._load_json_list(result.get("code_snippets"))
        for snippet in snippets:
            code = str(snippet.get("code", ""))
            if code:
                snippet["code"], snippet_findings = _redact_credentials(code)
                findings.extend(snippet_findings)

        if findings:
            redacted_answer = (
                f"{redacted_answer}\n\n_Credential-shaped values were redacted from this response. "
                "Use placeholders such as YOUR_API_KEY in examples._"
            )
            emit_trace_event(
                "post_process_s3_credential_redacted",
                {"redacted_count": len(findings), "rules": sorted(set(findings))},
                result,
            )

        # Any credential surviving redaction is a detector gap, not a safe output.
        surviving = detect_credentials(redacted_answer)
        if surviving:
            emit_trace_event(
                "post_process_s3_credential_blocked",
                {"finding_type": surviving[0]["type"]},
                result,
            )
            raise RuntimeError("S-3 blocked: credential detected in output")

        # Redact PII rather than raising. Raising turns a recoverable output-filter
        # decision into a failed invocation, and the detector's high-recall name
        # heuristic also matches framework vocabulary such as "Function Node".
        pii_findings = [
            finding
            for finding in detect_pii(redacted_answer)
            if finding.get("type") != "name" or not _is_domain_phrase(str(finding.get("value", "")))
        ]
        if pii_findings:
            redacted_answer = mask_pii(redacted_answer, pii_findings)
            redacted_answer = f"{redacted_answer}\n\n_Some personal information was redacted from this response._"
            emit_trace_event(
                "post_process_s3_pii_redacted",
                {
                    "finding_count": len(pii_findings),
                    "finding_types": sorted({str(item["type"]) for item in pii_findings}),
                },
                result,
            )

        result["answer"] = redacted_answer
        result["code_snippets"] = json.dumps(snippets, ensure_ascii=False)
        self._update_formatted_output(result, snippets)
        return result

    @staticmethod
    def _update_formatted_output(result: dict[str, Any], snippets: list[dict[str, Any]]) -> None:
        raw = result.get("formatted_output")
        if not isinstance(raw, str):
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        payload["answer"] = result["answer"]
        payload["code_snippets"] = snippets
        result["formatted_output"] = json.dumps(payload, ensure_ascii=False)
