from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_neutral_debator,
    create_portfolio_manager,
    create_research_manager,
    create_social_media_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import (
    AgentState,
    OrchestrationState,
    RunContext,
)

from .runtime_checkpoint import PhaseCheckpointStore


class LocalCollaborationBackend:
    """Coarse-grained collaboration seam for future A2A integration."""

    def invoke(
        self,
        phase_name: str,
        worker: Callable[[Dict[str, Any]], Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return worker(state)


class TradingOrchestrator:
    """Explicit phase-based runtime with typed handoffs between phases."""

    ANALYST_REPORT_FIELDS = {
        "market": "market_report",
        "social": "sentiment_report",
        "news": "news_report",
        "fundamentals": "fundamentals_report",
    }
    PHASES = (
        "analyst_pack",
        "research_debate",
        "research_manager",
        "trader",
        "risk_debate",
        "portfolio_manager",
    )

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, Any],
        selected_analysts: Iterable[str],
        config: Dict[str, Any],
        *,
        analyst_nodes: Dict[str, Callable] | None = None,
        bull_researcher: Callable | None = None,
        bear_researcher: Callable | None = None,
        research_manager: Callable | None = None,
        trader: Callable | None = None,
        aggressive_analyst: Callable | None = None,
        conservative_analyst: Callable | None = None,
        neutral_analyst: Callable | None = None,
        portfolio_manager: Callable | None = None,
        collaborator: LocalCollaborationBackend | None = None,
        checkpoint_store: PhaseCheckpointStore | None = None,
        debug: bool = False,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.selected_analysts = list(selected_analysts)
        self.config = config
        self.debug = debug
        self.collaborator = collaborator or LocalCollaborationBackend()
        self.checkpoint_store = checkpoint_store or PhaseCheckpointStore(
            self.config["data_cache_dir"]
        )

        self.analyst_nodes = analyst_nodes or {
            "market": create_market_analyst(self.quick_thinking_llm),
            "social": create_social_media_analyst(self.quick_thinking_llm),
            "news": create_news_analyst(self.quick_thinking_llm),
            "fundamentals": create_fundamentals_analyst(self.quick_thinking_llm),
        }
        self.bull_researcher = bull_researcher or create_bull_researcher(
            self.quick_thinking_llm
        )
        self.bear_researcher = bear_researcher or create_bear_researcher(
            self.quick_thinking_llm
        )
        self.research_manager = research_manager or create_research_manager(
            self.deep_thinking_llm
        )
        self.trader = trader or create_trader(self.quick_thinking_llm)
        self.aggressive_analyst = aggressive_analyst or create_aggressive_debator(
            self.quick_thinking_llm
        )
        self.conservative_analyst = (
            conservative_analyst
            or create_conservative_debator(self.quick_thinking_llm)
        )
        self.neutral_analyst = neutral_analyst or create_neutral_debator(
            self.quick_thinking_llm
        )
        self.portfolio_manager = portfolio_manager or create_portfolio_manager(
            self.deep_thinking_llm
        )

    def create_state(
        self, company_name: str, trade_date: str, past_context: str = ""
    ) -> OrchestrationState:
        return OrchestrationState(
            context=RunContext(
                company_of_interest=company_name,
                trade_date=str(trade_date),
                config=dict(self.config),
                selected_analysts=list(self.selected_analysts),
                past_context=past_context,
                trace_id=uuid4().hex,
            )
        )

    def run(
        self, company_name: str, trade_date: str, past_context: str = ""
    ) -> OrchestrationState:
        state = self._load_or_create_state(company_name, trade_date, past_context)

        for phase_name in self.PHASES:
            if phase_name in state.metadata.completed_phases:
                continue
            state.metadata.mark_phase_started(phase_name)
            try:
                getattr(self, f"_run_{phase_name}")(state)
                state.metadata.mark_phase_completed(phase_name)
                self._persist_checkpoint(state)
            except Exception as exc:
                state.metadata.record_failure(phase_name, exc)
                self._persist_checkpoint(state)
                raise

        state.metadata.status = "completed"
        state.metadata.current_phase = "completed"
        self._clear_checkpoint(state)
        return state

    def _load_or_create_state(
        self, company_name: str, trade_date: str, past_context: str
    ) -> OrchestrationState:
        if self.config.get("checkpoint_enabled"):
            restored = self.checkpoint_store.load(company_name, str(trade_date))
            if restored is not None:
                restored.context.past_context = past_context
                restored.context.config = dict(self.config)
                restored.context.selected_analysts = list(self.selected_analysts)
                restored.metadata.status = "running"
                return restored
        return self.create_state(company_name, trade_date, past_context)

    def _persist_checkpoint(self, state: OrchestrationState) -> None:
        if self.config.get("checkpoint_enabled"):
            self.checkpoint_store.save(state)

    def _clear_checkpoint(self, state: OrchestrationState) -> None:
        if self.config.get("checkpoint_enabled"):
            self.checkpoint_store.clear(
                state.context.company_of_interest, state.context.trade_date
            )

    def _invoke_phase_worker(
        self,
        phase_name: str,
        worker: Callable[[Dict[str, Any]], Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self.collaborator.invoke(phase_name, worker, state)
        if self.debug:
            for message in result.get("messages", []):
                if isinstance(message, BaseMessage):
                    message.pretty_print()
        return result

    def _build_analyst_state(self, state: OrchestrationState) -> Dict[str, Any]:
        return {
            "company_of_interest": state.context.company_of_interest,
            "trade_date": state.context.trade_date,
        }

    def _build_research_state(self, state: OrchestrationState) -> AgentState:
        return {
            "company_of_interest": state.context.company_of_interest,
            "market_report": state.analysts.market_report,
            "sentiment_report": state.analysts.sentiment_report,
            "news_report": state.analysts.news_report,
            "fundamentals_report": state.analysts.fundamentals_report,
            "investment_debate_state": state.research.to_dict(),
        }

    def _build_trader_state(self, state: OrchestrationState) -> AgentState:
        return {
            "company_of_interest": state.context.company_of_interest,
            "investment_plan": state.decisions.investment_plan,
        }

    def _build_risk_state(self, state: OrchestrationState) -> AgentState:
        return {
            "company_of_interest": state.context.company_of_interest,
            "market_report": state.analysts.market_report,
            "sentiment_report": state.analysts.sentiment_report,
            "news_report": state.analysts.news_report,
            "fundamentals_report": state.analysts.fundamentals_report,
            "trader_investment_plan": state.decisions.trader_investment_plan,
            "risk_debate_state": state.risk.to_dict(),
        }

    def _build_portfolio_manager_state(self, state: OrchestrationState) -> AgentState:
        payload = self._build_risk_state(state)
        payload["investment_plan"] = state.decisions.investment_plan
        payload["past_context"] = state.context.past_context
        return payload

    def _run_analyst_pack(self, state: OrchestrationState) -> None:
        for analyst_name in self.selected_analysts:
            self._run_single_analyst(analyst_name, state)

    def _run_single_analyst(self, analyst_name: str, state: OrchestrationState) -> None:
        analyst_node = self.analyst_nodes[analyst_name]
        tool_node = self.tool_nodes.get(analyst_name)
        field_name = self.ANALYST_REPORT_FIELDS[analyst_name]
        local_state = self._build_analyst_state(state)
        local_state["messages"] = [HumanMessage(content=state.context.company_of_interest)]

        for _ in range(self.config.get("max_recur_limit", 100)):
            result = self._invoke_phase_worker(
                f"{analyst_name}_analyst", analyst_node, local_state
            )
            self._merge_messages(local_state, result)

            if field_name in result and result[field_name]:
                setattr(state.analysts, field_name, result[field_name])

            latest_message = self._last_message(result.get("messages", []))
            if latest_message is None or not getattr(latest_message, "tool_calls", None):
                return

            if tool_node is None:
                raise RuntimeError(f"Missing tool node for analyst '{analyst_name}'")

            tool_result = tool_node.invoke({"messages": local_state["messages"]})
            self._merge_messages(local_state, tool_result)

        raise RuntimeError(f"Analyst '{analyst_name}' exceeded the recursion limit")

    def _run_research_debate(self, state: OrchestrationState) -> None:
        for _ in range(self.config.get("max_debate_rounds", 1)):
            bull_result = self._invoke_phase_worker(
                "bull_researcher", self.bull_researcher, self._build_research_state(state)
            )
            self._apply_investment_debate_update(state, bull_result)

            bear_result = self._invoke_phase_worker(
                "bear_researcher", self.bear_researcher, self._build_research_state(state)
            )
            self._apply_investment_debate_update(state, bear_result)

    def _run_research_manager(self, state: OrchestrationState) -> None:
        result = self._invoke_phase_worker(
            "research_manager", self.research_manager, self._build_research_state(state)
        )
        if "investment_debate_state" in result:
            self._apply_investment_debate_update(state, result)
        state.decisions = replace(
            state.decisions,
            investment_plan=result.get("investment_plan", state.decisions.investment_plan),
        )

    def _run_trader(self, state: OrchestrationState) -> None:
        result = self._invoke_phase_worker("trader", self.trader, self._build_trader_state(state))
        state.decisions = replace(
            state.decisions,
            trader_investment_plan=result.get(
                "trader_investment_plan", state.decisions.trader_investment_plan
            ),
        )

    def _run_risk_debate(self, state: OrchestrationState) -> None:
        for _ in range(self.config.get("max_risk_discuss_rounds", 1)):
            aggressive_result = self._invoke_phase_worker(
                "aggressive_analyst",
                self.aggressive_analyst,
                self._build_risk_state(state),
            )
            self._apply_risk_debate_update(state, aggressive_result)

            conservative_result = self._invoke_phase_worker(
                "conservative_analyst",
                self.conservative_analyst,
                self._build_risk_state(state),
            )
            self._apply_risk_debate_update(state, conservative_result)

            neutral_result = self._invoke_phase_worker(
                "neutral_analyst",
                self.neutral_analyst,
                self._build_risk_state(state),
            )
            self._apply_risk_debate_update(state, neutral_result)

    def _run_portfolio_manager(self, state: OrchestrationState) -> None:
        result = self._invoke_phase_worker(
            "portfolio_manager",
            self.portfolio_manager,
            self._build_portfolio_manager_state(state),
        )
        if "risk_debate_state" in result:
            self._apply_risk_debate_update(state, result)
        state.decisions = replace(
            state.decisions,
            final_trade_decision=result.get(
                "final_trade_decision", state.decisions.final_trade_decision
            ),
        )

    @staticmethod
    def _merge_messages(local_state: Dict[str, Any], result: Dict[str, Any]) -> None:
        for key, value in result.items():
            if key == "messages":
                local_state.setdefault("messages", [])
                local_state["messages"].extend(value)
            else:
                local_state[key] = value

    @staticmethod
    def _last_message(messages: List[BaseMessage]) -> BaseMessage | None:
        if not messages:
            return None
        return messages[-1]

    @staticmethod
    def _apply_investment_debate_update(
        state: OrchestrationState, result: Dict[str, Any]
    ) -> None:
        payload = result.get("investment_debate_state")
        if not payload:
            return
        state.research = replace(
            state.research,
            bull_history=payload.get("bull_history", state.research.bull_history),
            bear_history=payload.get("bear_history", state.research.bear_history),
            history=payload.get("history", state.research.history),
            current_response=payload.get(
                "current_response", state.research.current_response
            ),
            judge_decision=payload.get("judge_decision", state.research.judge_decision),
            count=payload.get("count", state.research.count),
            bull_thesis=payload.get("bull_history", state.research.bull_thesis),
            bear_thesis=payload.get("bear_history", state.research.bear_thesis),
        )

    @staticmethod
    def _apply_risk_debate_update(
        state: OrchestrationState, result: Dict[str, Any]
    ) -> None:
        payload = result.get("risk_debate_state")
        if not payload:
            return
        state.risk = replace(
            state.risk,
            aggressive_history=payload.get(
                "aggressive_history", state.risk.aggressive_history
            ),
            conservative_history=payload.get(
                "conservative_history", state.risk.conservative_history
            ),
            neutral_history=payload.get("neutral_history", state.risk.neutral_history),
            history=payload.get("history", state.risk.history),
            latest_speaker=payload.get("latest_speaker", state.risk.latest_speaker),
            current_aggressive_response=payload.get(
                "current_aggressive_response",
                state.risk.current_aggressive_response,
            ),
            current_conservative_response=payload.get(
                "current_conservative_response",
                state.risk.current_conservative_response,
            ),
            current_neutral_response=payload.get(
                "current_neutral_response", state.risk.current_neutral_response
            ),
            judge_decision=payload.get("judge_decision", state.risk.judge_decision),
            count=payload.get("count", state.risk.count),
        )
