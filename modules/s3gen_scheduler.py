"""Length-aware scheduling policy for batched S3Gen inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence


@dataclass(frozen=True)
class S3GenScheduleItem:
    """One completed T3 result available for S3Gen decoding."""

    item_id: str
    token_length: int
    payload: Any


@dataclass(frozen=True)
class S3GenBatchCapacity:
    """Empirical padded-length limits for each supported batch size."""

    max_padded_tokens: Dict[int, int]
    max_batch_size: int

    def accepts(self, batch: Sequence[S3GenScheduleItem]) -> bool:
        """Return whether batch's padded shape fits measured capacity."""
        size = len(batch)
        if size <= 1:
            return True
        limit = self.max_padded_tokens.get(size)
        if limit is None:
            return False
        return max(item.token_length for item in batch) <= limit


class S3GenScheduler:
    """Choose globally efficient neighboring batches from completed outputs."""

    def __init__(
        self,
        capacity: S3GenBatchCapacity,
        cost_model: Callable[[int, int], float] | None = None,
    ) -> None:
        """Initialize scheduler with empirical capacity and optional cost model."""
        self.capacity = capacity
        self.cost_model = cost_model or self._estimated_wall_time

    @staticmethod
    def _estimated_wall_time(batch_size: int, longest_tokens: int) -> float:
        """Estimate decode time from parallel rows and padded length.

        A measured shape-to-time profile can replace this proxy through
        ``cost_model`` without changing scheduling policy.
        """
        return float(longest_tokens) / max(1, batch_size)

    def schedule(
        self, items: Sequence[S3GenScheduleItem]
    ) -> List[List[S3GenScheduleItem]]:
        """Return minimum-cost contiguous partitions after length sorting.

        Candidate batches are contiguous in sorted-length order, so every batch
        contains neighboring lengths. Dynamic programming evaluates all legal
        batch sizes and therefore avoids committing to a locally convenient
        group that makes the remaining queue slower.
        """
        ordered = sorted(items, key=lambda item: item.token_length)
        count = len(ordered)
        if not count:
            return []

        best_cost = [float("inf")] * (count + 1)
        best_batches = [float("inf")] * (count + 1)
        choices: List[tuple[int, int] | None] = [None] * (count + 1)
        best_cost[0] = 0.0
        best_batches[0] = 0

        for end in range(1, count + 1):
            start_min = max(0, end - self.capacity.max_batch_size)
            for start in range(start_min, end):
                candidate = ordered[start:end]
                if not self.capacity.accepts(candidate):
                    continue
                candidate_cost = best_cost[start] + self.cost_model(
                    len(candidate), candidate[-1].token_length
                )
                candidate_batches = best_batches[start] + 1
                if (candidate_cost, candidate_batches) < (
                    best_cost[end],
                    best_batches[end],
                ):
                    best_cost[end] = candidate_cost
                    best_batches[end] = candidate_batches
                    choices[end] = (start, end)

        if choices[count] is None:
            raise RuntimeError("S3Gen scheduler could not partition completed outputs")

        result: List[List[S3GenScheduleItem]] = []
        end = count
        while end:
            start, selected_end = choices[end]  # type: ignore[misc]
            result.append(ordered[start:selected_end])
            end = start
        result.reverse()
        return result
