# Template Design Specification

## Position in AgentCore Architecture

- **Agent Class**: AgentStarDevAgent
- **L1 Base**: AgentBaseGraph
- **Category**: Cat 1
- **Generation Mode**: llm
- **Three-Layer Separation**:
  - State: flat TypedDict fields containing only msgpack-safe data.
  - Node: FunctionNode subclasses override execute and optional extra security hooks.
  - Graph: register_nodes composes the fixed framework pipeline.

## Architecture Overview

| Node | Responsibility | Input State | Output State | Inherits |
|------|----------------|-------------|--------------|----------|
| initialize | Framework invocation metadata | framework-managed | framework-managed | InitializeNode |
| pre_process | Sanitize the question; detect topic, intent, and retrieval strategy | user_input or raw_query | normalized_query and classification fields | FunctionNode |
| main | Retrieve framework documentation using the selected strategy | classified question | retrieved_docs, retrieval_count | FunctionNode |
| post_process | Synthesize or safely degrade, extract snippets, citations, and Cat recommendation | retrieved docs | answer, citations, code_snippets, cat_recommendation, confidence | FunctionNode |
| finalize | Framework response metadata | framework-managed | framework-managed | FinalizeNode |

### Data Flow

    START → initialize → pre_process → main → post_process → finalize → END
                                      ↘ retry → pre_process

Graph registers retrieval and request-scoped synthesis separately:

    self._nodes["main"] = MainNode()
    self._nodes["post_process"] = PostProcessNode(
        llm_timeout_s=config["timeout_s"],
        llm_max_retry=config["max_retry"],
    )

Production creates the Azure client from invocation-scoped secrets inside PostProcessNode.
Constructor injection is a test seam only; clients are never written to State or checkpoint surfaces.

### Classification and Retrieval Strategy

Topic detection checks security first. A question can mention both a node and a trust level while
only the security answer is correct, so the security corpus wins the tie.

Intent follows from the topic and the phrasing: a security topic always yields `security_rule`,
because "show me an example" of a security rule still needs the rule rather than a generic
snippet. Otherwise a how-to phrasing yields `code_example`, a decision phrasing yields
`architecture_decision`, and anything else is a `concept` question.

The intent selects the retrieval strategy, and the strategy — not the topic alone — decides
filtering. A `broad` strategy applies no topic filter at all: a general concept question is
answered better by the whole corpus than by one slice of it. `security_only` additionally pins
the source to the rules corpus.

### Sanitization

Script and event-handler markup is stripped, but generic HTML-tag stripping is deliberately **not**
applied. A developer question legitimately contains angle brackets in type annotations such as
`list[dict[str, Any]]` and in generics, and removing them would corrupt the very code the question
is about.

## State Definition

| Field group | Types | Purpose |
|-------------|-------|---------|
| raw_query, normalized_query, query_id | str | Safe question processing and correlation |
| detected_topic, intent_type, retrieval_strategy | str | Classification driving retrieval |
| has_code_context | bool | Whether the question carries code or a file path |
| retrieved_docs, retrieval_count | str, int | JSON-serialized documentation chunks |
| answer, citations, code_snippets, confidence | str | Domain response |
| cat_recommendation | str or None | Cat1/Cat2/Cat3 when the answer recommends exactly one |
| synthesis_mode | str | azure_openai or evidence_only |
| formatted_output | str | Canonical JSON response exposed by AgentBaseGraph |
| error_message, recoverable_error_code | str or None | Short-circuit and degradation control |
| input_error_message, input_error_guidance | str or None, list[str] | User-facing input guidance |

**State Constraints:**

- Flat TypedDict only; collections contain primitive or JSON-safe values.
- No API keys, bearer tokens, credentials, retrievers, LLM clients, or Pydantic objects.
- A credential pasted into the question is redacted at the S-2 hook, before it reaches State.
- InvocationContext is reconstructed with InvocationContext.from_state(state) inside a node if needed.

## Framework Utilization

- [x] InvocationContext and caller trust at the graph boundary.
- [x] Framework S-1 required_trust_level on every domain node.
- [x] Framework final S-2/S-3 gates remain non-overridden.
- [x] PreProcessNode extends S-2 through _extra_security_gate_input to redact pasted credentials.
- [x] MainNode extends S-2 through _extra_security_gate_input for normalized-query PII masking.
- [x] PostProcessNode extends S-3 through _extra_security_gate_output for generated output checks.
- [x] Every execute method emits a domain trace event on success, failure, or short circuit.
- [x] AgentRegistry and standalone server both load config/config.yaml separately from agent.yaml.
- [x] Marketplace and standalone entrypoints bind secrets per invocation and degrade safely when Azure is unavailable.

## Security and Runtime Decisions

- Minimum caller trust: VERIFIED_EXTERNAL.
- External bearer credentials map only to VERIFIED_EXTERNAL.
- A separately configured STG runner token is required for INTERNAL.
- Bearers are compared in constant time and invalid configured credentials return HTTP 401.
- memory_enabled and hitl.enabled are false, so no checkpointer is created.
- A missing retriever returns zero chunks; no framework APIs, fields, or config keys are invented.
- A credential in the inbound question is redacted rather than rejected, so a debugging question
  stays answerable while the secret never reaches State, the prompt, or the provider.
- S-3 redacts credential shapes inside the node that produces the content, so the framework
  credential scan sees clean output and the invocation ends in a redacted success. The rule set
  matches both the assignment form and the quoted JSON form, since JSON config is a common shape
  in framework answers.
- Personal data in the answer is masked, not raised on. The shared name detector favors recall and
  flags any run of Title Case words, so framework vocabulary such as "Function Node" and
  "Trust Level" match as readily as a real name — and the detector's span absorbs the
  sentence-initial verb, so "Extend Function Node" arrives as a single finding. A finding is
  treated as non-personal only when every one of its capitalised tokens is framework vocabulary
  or a documentation imperative; a real name contributes at least one token that is not.
- Every credential and detection pattern uses bounded quantifiers, so a hostile input cannot
  force quadratic-time backtracking.
- Azure client creation is lazy, so missing provider configuration does not break startup. At
  invocation time, provider failure emits progress and returns `synthesis_mode=evidence_only`
  with the retrieved documentation and no fabricated facts.

### Cat recommendation

The recommendation is read back out of the prose answer rather than requested as a separate
structured field, because the model states it in the text either way and a second call would be
a second chance to disagree with itself. When several distinct Cat values appear, the answer is
comparing them rather than recommending one, so no recommendation is recorded.

## EU AI Act Art.13 Design-Time Evidence

The proposal declares this developer documentation capability outside Annex III scope.
The README documents intended purpose and limitations, and response metadata exposes
no-results, synthesis mode, and confidence information for user transparency.

## Import Isolation Confirmation

- [x] Source imports framework and shared layers, never the Level 0 platform SDK.
- [x] Agent-local imports use the src package path.

## Design Decision Record

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| L1 base | AgentBaseGraph / AutonomousBaseGraph | AgentBaseGraph | Fixed classify, retrieve, answer stages need no autonomous loop |
| Composition | Standalone / subgraph / remote | Standalone | One independently deployable technical capability |
| HITL | Enabled / disabled | Disabled | The capability answers questions and authorizes nothing |
| Memory | Enabled / disabled | Disabled | Each question is self-contained |
| Retrieval backend | Injected retriever / embedded client | Injected retriever | The deployment owns the corpus; the template stays backend-agnostic |
| Broad-strategy filtering | Always filter by topic / no filter when broad | No filter | A general concept question is answered better by the whole corpus |
| Pasted credential | Reject the question / redact in place | Redact | The surrounding config is what makes a debugging question answerable |
| HTML sanitization | Strip all tags / strip script and handlers only | Script and handlers only | Generic stripping corrupts type annotations and generics in the question |
| Cat recommendation | Separate structured LLM call / parse from the answer | Parse | The answer already states it; a second call could contradict the first |
| Non-person name handling | Fixed phrase allowlist / domain-vocabulary token test | Token test | Detector span boundaries shift with context, so a fixed list cannot keep up |
