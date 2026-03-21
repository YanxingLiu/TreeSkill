"""Trace Storage — append-only JSONL persistence for interaction traces.

Each line in the JSONL file is a self-contained JSON object representing
a single ``Trace`` record.  This makes it safe for concurrent appenders
and trivially streamable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from evoskill.config import StorageConfig
from evoskill.schema import Trace


class TraceStorage:
    """Append-only JSONL store for ``Trace`` objects.

    Parameters
    ----------
    config : StorageConfig
        Must include ``trace_path`` pointing to the JSONL file location.
    """

    def __init__(self, config: StorageConfig) -> None:
        self._path = Path(config.trace_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, trace: Trace) -> None:
        """Serialize *trace* and append it as a single JSON line."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(trace.model_dump_json() + "\n")

    def upsert(self, trace: Trace) -> None:
        """Persist *trace*, replacing an existing record with the same ID."""
        traces = self.load_all()
        replaced = False

        for index, existing in enumerate(traces):
            if existing.id == trace.id:
                traces[index] = trace
                replaced = True
                break

        if not replaced:
            traces.append(trace)

        self._write_all(traces)

    def load_all(self) -> List[Trace]:
        """Read every trace from the JSONL file, deduplicated by trace ID."""
        if not self._path.exists():
            return []
        traces_by_id: Dict[str, Trace] = {}
        trace_order: List[str] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    trace = Trace.model_validate_json(line)
                    if trace.id not in traces_by_id:
                        trace_order.append(trace.id)
                    traces_by_id[trace.id] = trace
        return [traces_by_id[trace_id] for trace_id in trace_order]

    def get_feedback_samples(
        self,
        min_score: float = 0.0,
        max_score: float = 0.5,
    ) -> List[Trace]:
        """Return traces whose feedback score falls in [min_score, max_score].

        This surfaces the "bad" examples that the APO optimizer should
        learn from.  Traces without feedback are silently skipped.
        """
        return [
            t
            for t in self.load_all()
            if t.feedback is not None
            and min_score <= t.feedback.score <= max_score
        ]

    def _write_all(self, traces: List[Trace]) -> None:
        with self._path.open("w", encoding="utf-8") as fh:
            for trace in traces:
                fh.write(trace.model_dump_json() + "\n")
