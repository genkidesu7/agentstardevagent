# Test Specification

## Test Strategy

- Coverage target: 80%
- Test types: unit, framework compliance, proof-of-boundary, integration, and local STG invocation.
- Unit tests may call execute directly for isolated business logic.
- Security-boundary tests call node(state) through FunctionNode.__call__.
- No live Azure OpenAI or vector index is required: the provider is exercised through a recording
  double, and retrieval through an injected stub retriever.

## Framework Compliance Tests

| TC-ID | Test | Expected Result |
|-------|------|-----------------|
| TC-01 | State contract is a flat TypedDict | No Pydantic, dataclass, secret, or runtime client fields |
| TC-02 | Invalid or unsafe input is rejected or sanitized | Safe failure, markup stripped, credential redacted |
| TC-03 | No credential in State | Credential scan reports zero violations |
| TC-04 | InvocationContext construction inside nodes | Only from_state is allowed; authenticated server adapter is exempt |
| TC-05 | No duplicate lifecycle trace events | execute methods do not emit node_start, node_complete, or node_error |
| TC-06 | FunctionNode input security gate is final | Override attempt raises TypeError |
| TC-07 | FunctionNode output security gate is final | Override attempt raises TypeError |
| TC-08 | required_trust_level is enforced through node(state) | Denial happens before execute |
| TC-09 | PreProcessNode and MainNode extra S-2 hooks | Pasted credential redacted; PII in the retrieval query masked |
| TC-10 | PostProcessNode extra S-3 hook | Credential shapes redacted, placeholders preserved, surviving credential blocked |
| TC-11 | Domain tracing | Each execute path emits a domain event |

## Proof-of-Boundary Tests

| PB-ID | Boundary | Expected Result |
|-------|----------|-----------------|
| PB-1 | BaseNode to trace emitter | Domain and lifecycle trace evidence exists |
| PB-2 | State serialization | Only msgpack-safe annotated fields |
| PB-3 | Template to external service | Injected retriever and LLM doubles exercise the boundary |
| PB-4 | Import isolation | No Level 0 imports |
| PB-5 | Checkpoint safety, conditional | Auto-waived — checkpointing disabled |
| PB-6 | Invoke order | S-1 → node_start → S-2 → execute → S-3 → node_complete |
| PB-7 | HITL interrupt propagation, conditional | Auto-waived — non-HITL |

PB-1 through PB-4 and PB-6 are mandatory. PB-5 applies only when memory or HITL enables
checkpointing and the framework provides both ingress hooks. PB-7 applies only when
config/config.yaml sets hitl.enabled to true.

## Business Logic Tests

| TC-ID | Test | Input | Expected Result |
|-------|------|-------|-----------------|
| BL-01 | Topic detection | questions across all six topics | Correct topic, including `general` fallback |
| BL-02 | Security precedence | question naming both a node and a trust level | security topic, security_rule intent, security_only strategy |
| BL-03 | Intent and strategy | how-to, decision, and concept phrasings | Correct intent and the strategy it selects |
| BL-04 | Code-context detection | question with and without code | has_code_context set accordingly |
| BL-05 | Sanitization | script markup and a type annotation | Markup stripped, `list[dict[str, Any]]` preserved |
| BL-06 | Input bounds | over-length and two-word questions | Actionable input guidance, not a failed invocation |
| BL-07 | Pasted credential | question containing an API key | Redacted in place, question still answerable |
| BL-08 | Retrieval filters | focused, broad, and security_only strategies | Topic filter applied, omitted, and source pinned respectively |
| BL-09 | Retrieval outage | retriever raises | Safe guidance, backend detail not disclosed, invocation still succeeds |
| BL-10 | Response synthesis | docs plus injected LLM | Answer, deduplicated citations, snippets, confidence, canonical output |
| BL-11 | Provider failure | LLM raises | Evidence-only answer carrying the retrieved documentation |
| BL-12 | Cat extraction | answers naming one Cat, several Cats, or none | Single recommendation, None, and None respectively |
| BL-13 | Confidence tiers | doc counts and scores | High, medium, or low per policy |
| BL-14 | Output safety | generated credential alongside a documented placeholder | Credential redacted, placeholder preserved |
| BL-15 | JSON config redaction | `{"api_key": "..."}` | Value redacted |
| BL-16 | Non-person name handling | framework vocabulary and a real name in one answer | Vocabulary preserved, personal name masked |
| BL-17 | Azure wiring | Azure secrets present | Request-scoped client built from ctx.secrets, no process-global client |
| BL-18 | Missing secrets | no Azure secrets bound | Terminal evidence-only response with visible Azure-unavailable guidance |
| BL-19 | Bearer mapping | external, runner, wrong, and absent bearer | VERIFIED_EXTERNAL, INTERNAL, HTTP 401, and ANONYMOUS respectively |
| BL-20 | Marketplace shaping | same state with and without conversation_history | Marketplace receives the answer text, the API keeps canonical JSON |

## Test Execution Summary

- Execution date: 2026-09-30
- Framework: agenticstar-agentcore 1.0.3 wheel (not a CI import shim)
- Full suite: 84 passed, 0 failed, 3 skipped
- Local gates: forbidden strings, manifest schema, trust level, audit trace, dependency pinning,
  stub tests, credential scan, OSS license, and Cat consistency all PASS
- Lint and types: ruff format, ruff check, and mypy strict on src/ all clean

The three expected skips are PB-5 (checkpointing disabled) and both PB-7 cases (non-HITL).

One defect was found and fixed during this cycle, covered by BL-16: the personal-data filter
masked "Extend Function Node" as a person name, because the detector's span absorbs the
sentence-initial verb that a documentation answer opens with.
