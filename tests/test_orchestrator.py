from pathlib import Path

import pytest

from tradingagents.graph.orchestrator import TradingOrchestrator
from tradingagents.graph.runtime_checkpoint import PhaseCheckpointStore


def _analyst(report_field: str, value: str):
    def worker(_state):
        return {report_field: value}

    return worker


def _bull(_state):
    return {
        "investment_debate_state": {
            "history": "Bull Analyst: Growth is intact.",
            "bull_history": "Bull Analyst: Growth is intact.",
            "bear_history": "",
            "current_response": "Bull Analyst: Growth is intact.",
            "judge_decision": "",
            "count": 1,
        }
    }


def _bear(_state):
    return {
        "investment_debate_state": {
            "history": "Bull Analyst: Growth is intact.\nBear Analyst: Valuation is rich.",
            "bull_history": "Bull Analyst: Growth is intact.",
            "bear_history": "Bear Analyst: Valuation is rich.",
            "current_response": "Bear Analyst: Valuation is rich.",
            "judge_decision": "",
            "count": 2,
        }
    }


def _research_manager(state):
    assert state["market_report"] == "Market report"
    return {
        "investment_debate_state": {
            **state["investment_debate_state"],
            "judge_decision": "Structured research plan",
            "current_response": "Structured research plan",
        },
        "investment_plan": "Structured research plan",
    }


def _trader(_state):
    return {"trader_investment_plan": "Trader proposal"}


def _aggressive(_state):
    return {
        "risk_debate_state": {
            "history": "Aggressive Analyst: Lean in.",
            "aggressive_history": "Aggressive Analyst: Lean in.",
            "conservative_history": "",
            "neutral_history": "",
            "latest_speaker": "Aggressive",
            "current_aggressive_response": "Aggressive Analyst: Lean in.",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 1,
        }
    }


def _conservative(_state):
    return {
        "risk_debate_state": {
            "history": "Aggressive Analyst: Lean in.\nConservative Analyst: Size down.",
            "aggressive_history": "Aggressive Analyst: Lean in.",
            "conservative_history": "Conservative Analyst: Size down.",
            "neutral_history": "",
            "latest_speaker": "Conservative",
            "current_aggressive_response": "Aggressive Analyst: Lean in.",
            "current_conservative_response": "Conservative Analyst: Size down.",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 2,
        }
    }


def _neutral(_state):
    return {
        "risk_debate_state": {
            "history": (
                "Aggressive Analyst: Lean in.\n"
                "Conservative Analyst: Size down.\n"
                "Neutral Analyst: Keep moderate exposure."
            ),
            "aggressive_history": "Aggressive Analyst: Lean in.",
            "conservative_history": "Conservative Analyst: Size down.",
            "neutral_history": "Neutral Analyst: Keep moderate exposure.",
            "latest_speaker": "Neutral",
            "current_aggressive_response": "Aggressive Analyst: Lean in.",
            "current_conservative_response": "Conservative Analyst: Size down.",
            "current_neutral_response": "Neutral Analyst: Keep moderate exposure.",
            "judge_decision": "",
            "count": 3,
        }
    }


def _portfolio_manager(state):
    assert state["past_context"] == "Past lesson"
    return {
        "risk_debate_state": {
            **state["risk_debate_state"],
            "judge_decision": "Rating: Hold\nWait for confirmation.",
        },
        "final_trade_decision": "Rating: Hold\nWait for confirmation.",
    }


@pytest.mark.unit
def test_orchestrator_runs_explicit_phases_without_global_messages(tmp_path):
    config = {
        "data_cache_dir": str(tmp_path),
        "checkpoint_enabled": False,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 4,
    }
    orchestrator = TradingOrchestrator(
        quick_thinking_llm=None,
        deep_thinking_llm=None,
        tool_nodes={},
        selected_analysts=["market", "social"],
        config=config,
        analyst_nodes={
            "market": _analyst("market_report", "Market report"),
            "social": _analyst("sentiment_report", "Sentiment report"),
        },
        bull_researcher=_bull,
        bear_researcher=_bear,
        research_manager=_research_manager,
        trader=_trader,
        aggressive_analyst=_aggressive,
        conservative_analyst=_conservative,
        neutral_analyst=_neutral,
        portfolio_manager=_portfolio_manager,
    )

    state = orchestrator.run("NVDA", "2026-01-10", past_context="Past lesson")

    assert state.metadata.completed_phases == list(orchestrator.PHASES)
    assert state.analysts.market_report == "Market report"
    assert state.analysts.sentiment_report == "Sentiment report"
    assert state.decisions.investment_plan == "Structured research plan"
    assert state.decisions.trader_investment_plan == "Trader proposal"
    assert state.decisions.final_trade_decision.startswith("Rating: Hold")

    legacy = state.to_legacy_state()
    assert "messages" not in legacy
    assert legacy["investment_debate_state"]["count"] == 2
    assert legacy["risk_debate_state"]["count"] == 3


@pytest.mark.unit
def test_orchestrator_resumes_from_phase_checkpoint(tmp_path):
    config = {
        "data_cache_dir": str(tmp_path),
        "checkpoint_enabled": True,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 4,
    }
    should_fail = {"value": True}

    def flaky_trader(_state):
        if should_fail["value"]:
            raise RuntimeError("temporary failure")
        return {"trader_investment_plan": "Trader proposal"}

    orchestrator = TradingOrchestrator(
        quick_thinking_llm=None,
        deep_thinking_llm=None,
        tool_nodes={},
        selected_analysts=["market"],
        config=config,
        analyst_nodes={"market": _analyst("market_report", "Market report")},
        bull_researcher=_bull,
        bear_researcher=_bear,
        research_manager=_research_manager,
        trader=flaky_trader,
        aggressive_analyst=_aggressive,
        conservative_analyst=_conservative,
        neutral_analyst=_neutral,
        portfolio_manager=_portfolio_manager,
    )

    with pytest.raises(RuntimeError, match="temporary failure"):
        orchestrator.run("NVDA", "2026-01-10", past_context="Past lesson")

    checkpoint_store = PhaseCheckpointStore(str(tmp_path))
    restored = checkpoint_store.load("NVDA", "2026-01-10")
    assert restored is not None
    assert restored.metadata.completed_phases == [
        "analyst_pack",
        "research_debate",
        "research_manager",
    ]

    should_fail["value"] = False
    resumed = orchestrator.run("NVDA", "2026-01-10", past_context="Past lesson")
    assert resumed.decisions.final_trade_decision.startswith("Rating: Hold")
    checkpoint_path = (
        Path(str(tmp_path)) / "orchestrator_checkpoints" / "NVDA" / "2026-01-10.json"
    )
    assert not checkpoint_path.exists()
