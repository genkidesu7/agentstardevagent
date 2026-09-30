"""Flat, msgpack-safe state for CMN-C1-130."""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Agent state for AgentStarDevAgent.

    Add only fields specific to this agent below.
    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    """

    # --- Input ---
    raw_query: str
    top_k: int
    score_threshold: float
    response_budget: int
    max_input_length: int

    # --- Pre-process outputs ---
    query_id: str
    normalized_query: str
    detected_topic: str  # level2_selection | state_design | hitl | security | graph_wiring | general
    intent_type: str  # architecture_decision | code_example | concept | security_rule
    retrieval_strategy: str  # broad | focused | security_only
    has_code_context: bool

    # --- Main (retrieval) outputs ---
    retrieved_docs: str  # JSON string: list of doc dicts
    retrieval_count: int

    # --- Post-process outputs ---
    answer: str
    citations: str  # JSON string: list of citation dicts
    code_snippets: str  # JSON string: list of snippet dicts
    cat_recommendation: str | None  # Cat1 | Cat2 | Cat3 | None
    confidence: str  # "high" | "medium" | "low"
    synthesis_mode: str  # "azure_openai" | "evidence_only"
    formatted_output: str

    # --- Error ---
    error_message: str | None
    recoverable_error_code: str | None
    input_error_message: str | None
    input_error_guidance: list[str]
