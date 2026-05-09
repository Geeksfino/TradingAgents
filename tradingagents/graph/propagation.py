from typing import Any, Dict

from tradingagents.agents.utils.agent_states import (
    AgentState,
    OrchestrationState,
    RunContext,
)


class Propagator:
    """Creates the typed runtime state for the explicit orchestrator."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def create_orchestration_state(
        self,
        company_name: str,
        trade_date: str,
        past_context: str = "",
        selected_analysts: list[str] | None = None,
    ) -> OrchestrationState:
        return OrchestrationState(
            context=RunContext(
                company_of_interest=company_name,
                trade_date=str(trade_date),
                config=dict(self.config),
                selected_analysts=list(selected_analysts or []),
                past_context=past_context,
            )
        )

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        past_context: str = "",
        selected_analysts: list[str] | None = None,
    ) -> AgentState:
        """Compatibility wrapper for callers that still expect the legacy flat state."""
        return self.create_orchestration_state(
            company_name,
            trade_date,
            past_context=past_context,
            selected_analysts=selected_analysts,
        ).to_legacy_state()
