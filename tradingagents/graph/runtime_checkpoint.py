from __future__ import annotations

import json
from pathlib import Path

from tradingagents.agents.utils.agent_states import OrchestrationState
from tradingagents.dataflows.utils import safe_ticker_component


class PhaseCheckpointStore:
    """Persist orchestrator state between coarse-grained phases."""

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)

    def _path(self, ticker: str, trade_date: str) -> Path:
        safe_ticker = safe_ticker_component(ticker).upper()
        directory = self._data_dir / "orchestrator_checkpoints" / safe_ticker
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{trade_date}.json"

    def load(self, ticker: str, trade_date: str) -> OrchestrationState | None:
        path = self._path(ticker, trade_date)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return OrchestrationState.from_dict(payload)

    def save(self, state: OrchestrationState) -> Path:
        path = self._path(state.context.company_of_interest, state.context.trade_date)
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        return path

    def clear(self, ticker: str, trade_date: str) -> None:
        path = self._path(ticker, trade_date)
        if path.exists():
            path.unlink()
