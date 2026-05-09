from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, TypedDict


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunContextState(TypedDict):
    company_of_interest: str
    trade_date: str
    config: Dict[str, Any]
    selected_analysts: List[str]
    past_context: str
    trace_id: str


class AnalystArtifactsState(TypedDict):
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str


class InvestDebateState(TypedDict):
    bull_history: str
    bear_history: str
    history: str
    current_response: str
    judge_decision: str
    count: int
    bull_thesis: str
    bear_thesis: str


class RiskDebateState(TypedDict):
    aggressive_history: str
    conservative_history: str
    neutral_history: str
    history: str
    latest_speaker: str
    current_aggressive_response: str
    current_conservative_response: str
    current_neutral_response: str
    judge_decision: str
    count: int


class DecisionArtifactsState(TypedDict):
    investment_plan: str
    trader_investment_plan: str
    final_trade_decision: str


class ExecutionMetadataState(TypedDict):
    status: Literal["pending", "running", "completed", "failed"]
    current_phase: str
    completed_phases: List[str]
    retries: Dict[str, int]
    errors: List[str]
    started_at: str
    updated_at: str


class OrchestrationStateSnapshot(TypedDict):
    context: RunContextState
    analysts: AnalystArtifactsState
    research: InvestDebateState
    decisions: DecisionArtifactsState
    risk: RiskDebateState
    metadata: ExecutionMetadataState


class AgentState(TypedDict, total=False):
    company_of_interest: str
    trade_date: str
    past_context: str
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    investment_debate_state: InvestDebateState
    investment_plan: str
    trader_investment_plan: str
    risk_debate_state: RiskDebateState
    final_trade_decision: str
    sender: str
    messages: List[Any]
    context: RunContextState
    analysts: AnalystArtifactsState
    research: InvestDebateState
    decisions: DecisionArtifactsState
    risk: RiskDebateState
    metadata: ExecutionMetadataState


@dataclass
class RunContext:
    company_of_interest: str
    trade_date: str
    config: Dict[str, Any]
    selected_analysts: List[str]
    past_context: str = ""
    trace_id: str = ""

    def to_dict(self) -> RunContextState:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: RunContextState) -> "RunContext":
        return cls(**payload)


@dataclass
class AnalystArtifacts:
    market_report: str = ""
    sentiment_report: str = ""
    news_report: str = ""
    fundamentals_report: str = ""

    def to_dict(self) -> AnalystArtifactsState:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: AnalystArtifactsState) -> "AnalystArtifacts":
        return cls(**payload)


@dataclass
class InvestmentDebateArtifacts:
    bull_history: str = ""
    bear_history: str = ""
    history: str = ""
    current_response: str = ""
    judge_decision: str = ""
    count: int = 0
    bull_thesis: str = ""
    bear_thesis: str = ""

    def to_dict(self) -> InvestDebateState:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: InvestDebateState) -> "InvestmentDebateArtifacts":
        return cls(**payload)


@dataclass
class RiskDebateArtifacts:
    aggressive_history: str = ""
    conservative_history: str = ""
    neutral_history: str = ""
    history: str = ""
    latest_speaker: str = ""
    current_aggressive_response: str = ""
    current_conservative_response: str = ""
    current_neutral_response: str = ""
    judge_decision: str = ""
    count: int = 0

    def to_dict(self) -> RiskDebateState:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: RiskDebateState) -> "RiskDebateArtifacts":
        return cls(**payload)


@dataclass
class DecisionArtifacts:
    investment_plan: str = ""
    trader_investment_plan: str = ""
    final_trade_decision: str = ""

    def to_dict(self) -> DecisionArtifactsState:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: DecisionArtifactsState) -> "DecisionArtifacts":
        return cls(**payload)


@dataclass
class ExecutionMetadata:
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    current_phase: str = "initialized"
    completed_phases: List[str] = field(default_factory=list)
    retries: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def mark_phase_started(self, phase_name: str) -> None:
        self.status = "running"
        self.current_phase = phase_name
        self.updated_at = _utc_now()

    def mark_phase_completed(self, phase_name: str) -> None:
        if phase_name not in self.completed_phases:
            self.completed_phases.append(phase_name)
        self.current_phase = phase_name
        self.updated_at = _utc_now()

    def record_failure(self, phase_name: str, error: Exception) -> None:
        self.status = "failed"
        self.current_phase = phase_name
        self.errors.append(f"{phase_name}: {error}")
        self.updated_at = _utc_now()

    def to_dict(self) -> ExecutionMetadataState:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: ExecutionMetadataState) -> "ExecutionMetadata":
        return cls(**payload)


@dataclass
class OrchestrationState:
    context: RunContext
    analysts: AnalystArtifacts = field(default_factory=AnalystArtifacts)
    research: InvestmentDebateArtifacts = field(default_factory=InvestmentDebateArtifacts)
    decisions: DecisionArtifacts = field(default_factory=DecisionArtifacts)
    risk: RiskDebateArtifacts = field(default_factory=RiskDebateArtifacts)
    metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)

    def to_dict(self) -> OrchestrationStateSnapshot:
        return {
            "context": self.context.to_dict(),
            "analysts": self.analysts.to_dict(),
            "research": self.research.to_dict(),
            "decisions": self.decisions.to_dict(),
            "risk": self.risk.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: OrchestrationStateSnapshot) -> "OrchestrationState":
        return cls(
            context=RunContext.from_dict(payload["context"]),
            analysts=AnalystArtifacts.from_dict(payload["analysts"]),
            research=InvestmentDebateArtifacts.from_dict(payload["research"]),
            decisions=DecisionArtifacts.from_dict(payload["decisions"]),
            risk=RiskDebateArtifacts.from_dict(payload["risk"]),
            metadata=ExecutionMetadata.from_dict(payload["metadata"]),
        )

    def to_legacy_state(self) -> AgentState:
        """Flatten artifacts for compatibility with existing loggers and tests."""
        return {
            "company_of_interest": self.context.company_of_interest,
            "trade_date": self.context.trade_date,
            "past_context": self.context.past_context,
            "market_report": self.analysts.market_report,
            "sentiment_report": self.analysts.sentiment_report,
            "news_report": self.analysts.news_report,
            "fundamentals_report": self.analysts.fundamentals_report,
            "investment_debate_state": self.research.to_dict(),
            "investment_plan": self.decisions.investment_plan,
            "trader_investment_plan": self.decisions.trader_investment_plan,
            "risk_debate_state": self.risk.to_dict(),
            "final_trade_decision": self.decisions.final_trade_decision,
            "context": self.context.to_dict(),
            "analysts": self.analysts.to_dict(),
            "research": self.research.to_dict(),
            "decisions": self.decisions.to_dict(),
            "risk": self.risk.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
