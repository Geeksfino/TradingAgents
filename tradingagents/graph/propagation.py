from typing import Any, Dict

from tradingagents.agents.utils.agent_states import OrchestrationState, RunContext


class Propagator:
    """Creates the typed runtime state for the explicit orchestrator."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def create_initial_state(
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
