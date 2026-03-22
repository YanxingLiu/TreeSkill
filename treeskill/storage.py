"""Trace Storage — JSONL persistence for interaction traces.

Each line in the JSONL file is a self-contained JSON object representing
a single ``Trace`` record. A sidecar lock file serializes readers and
writers so append and rewrite flows stay consistent across processes.

DPO export: traces with ``feedback.correction`` can be exported as
preference pairs for Direct Preference Optimization fine-tuning.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from treeskill.config import StorageConfig
from treeskill.schema import Message, Trace

logger = logging.getLogger(__name__)


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
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, trace: Trace) -> None:
        """Serialize *trace* and append it as a single JSON line."""
        with self._exclusive_lock():
            self._append_unlocked(trace)

    def upsert(self, trace: Trace) -> None:
        """Persist *trace*, replacing an existing record with the same ID."""
        with self._exclusive_lock():
            traces = self.load_all(lock=False)
            replaced = False

            for index, existing in enumerate(traces):
                if existing.id == trace.id:
                    traces[index] = trace
                    replaced = True
                    break

            if not replaced:
                traces.append(trace)

            self._write_all(traces)

    def load_all(self, *, lock: bool = True) -> List[Trace]:
        """Read every trace from the JSONL file, deduplicated by trace ID."""
        if lock:
            with self._exclusive_lock():
                return self.load_all(lock=False)
        return self._load_all_unlocked()

    def _load_all_unlocked(self) -> List[Trace]:
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

    # ------------------------------------------------------------------
    # DPO Export
    # ------------------------------------------------------------------

    def get_dpo_pairs(self) -> List[Dict[str, Any]]:
        """Extract DPO preference pairs from stored traces.

        A valid DPO pair requires:
        - ``feedback.correction`` (the human-provided ideal response = chosen)
        - ``prediction`` (the model's original response = rejected)

        Deduplicates by trace id (the /rewrite flow re-appends the same
        trace, so the JSONL may contain duplicates).

        Returns a list of dicts with keys:
        ``prompt``, ``chosen``, ``rejected``, ``score``, ``critique``.
        """
        all_traces = self.load_all()

        # Deduplicate: keep the last occurrence of each trace id
        # (the one with feedback attached)
        seen: Dict[str, Trace] = {}
        for t in all_traces:
            seen[t.id] = t

        pairs: List[Dict[str, Any]] = []
        for t in seen.values():
            if t.feedback is None or not t.feedback.correction:
                continue

            prompt = _messages_to_chatml(t.inputs)
            rejected = _message_content_to_str(t.prediction.content)
            chosen = t.feedback.correction

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "score": t.feedback.score,
                "critique": t.feedback.critique,
            })

        return pairs
    
    def _append_unlocked(self, trace: Trace) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(trace.model_dump_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _write_all(self, traces: List[Trace]) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f"{self._path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            for trace in traces:
                fh.write(trace.model_dump_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        try:
            tmp_path.replace(self._path)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise

    @contextlib.contextmanager
    def _exclusive_lock(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as fh:
            _lock_file(fh)
            try:
                yield
            finally:
                _unlock_file(fh)

    def export_dpo(
        self,
        output_path: Union[str, Path],
        *,
        include_system: bool = True,
    ) -> int:
        """Export DPO preference pairs to a JSONL file.

        Parameters
        ----------
        output_path : str | Path
            Where to write the DPO JSONL file.
        include_system : bool
            Whether to include system messages in the prompt field.

        Returns
        -------
        int
            Number of pairs exported.
        """
        pairs = self.get_dpo_pairs()
        if not include_system:
            for pair in pairs:
                pair["prompt"] = [
                    m for m in pair["prompt"]
                    if m["role"] != "system"
                ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info("Exported %d DPO pairs to %s", len(pairs), output_path)
        return len(pairs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _message_content_to_str(content) -> str:
    """Extract plain text from Message content (str or List[ContentPart])."""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if hasattr(part, "text"):
            texts.append(part.text)
    return " ".join(texts) if texts else ""


def _messages_to_chatml(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert a list of Message objects to ChatML dicts."""
    return [
        {"role": m.role, "content": _message_content_to_str(m.content)}
        for m in messages
    ]


if os.name == "nt":
    import msvcrt

    def _lock_file(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_file(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock_file(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
