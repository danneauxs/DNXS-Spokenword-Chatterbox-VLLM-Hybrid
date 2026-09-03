from __future__ import annotations

import functools
import json
import logging
import sys
import time
import gc
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from modules.pause_utils import parse_pause_tags

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

vllm_root = project_root / "chatterbox-vllm" / "src"
sys.path.insert(0, str(vllm_root))

from chatterbox_vllm.tts import ChatterboxTTS
from config import config as runtime_config

logger = logging.getLogger(__name__)


def _extract_supported_params(tts_params: dict) -> dict:
    """Return per-chunk sampling params, dropping book-level keys vLLM cannot vary.

    cfg_weight is engine-load CFG. language_id/language are T3_LANGUAGE, not
    VADER per-chunk fields.
    """
    skip = {"cfg_weight", "language_id", "language"}
    return {k: v for k, v in tts_params.items() if k not in skip}


def _group_chunks_by_params(chunk_metadata: list) -> dict:
    """Groups chunk metadata by supported TTS parameters."""
    groups = defaultdict(list)
    for meta in chunk_metadata:
        supported_params = _extract_supported_params(meta["tts_params"])
        param_key = tuple(sorted(supported_params.items()))
        groups[param_key].append(meta)
    return dict(groups)


class _ProgressRelay:
    """tqdm-compatible stand-in for vLLM's own progress bar(s).

    vllm's `use_tqdm` callable is used at TWO separate call sites with different
    calling conventions (both in vllm/entrypoints/llm.py):
      - LLM._validate_and_add_requests: `tqdm_func(it, desc="Adding requests")` --
        wraps an iterable positionally; real tqdm proxies iteration while showing a
        bar. This is just request bookkeeping (near-instant), not generation
        progress, so this class passes it through unchanged with no progress relay.
      - LLM._run_engine: `tqdm_func(total=N, desc=..., dynamic_ncols=..., postfix=...)`,
        then `.update(n)` after every individual finished request, `.close()` at the
        end. THIS is the real per-request generation progress we want, relayed to a
        GUI callback instead of drawing a terminal bar -- the only way to get live
        Phase 1 progress, since vLLM's generate() is otherwise synchronous and gives
        nothing back until the *entire* batch is done.
    """

    _MIN_INTERVAL = 0.2  # seconds; avoids flooding the GUI's cross-thread signal queue

    def __init__(self, iterable=None, total=None, desc=None, dynamic_ncols=None,
                 postfix=None, on_update=None, base_count=0, grand_total=0):
        """Stores vLLM's tqdm-construction args plus this relay's own progress context."""
        self._iterable = iterable
        self.total = total
        self.n = 0
        self.postfix = postfix
        self._on_update = on_update
        self._base_count = base_count
        self._grand_total = grand_total or total
        self._start = time.time()
        self._last_emit = 0.0

    def __iter__(self):
        """Passthrough for the "Adding requests" enqueue step -- not generation
        progress, so this relays nothing, just lets iteration proceed unchanged."""
        return iter(self._iterable) if self._iterable is not None else iter(())

    @property
    def format_dict(self):
        """Minimal subset of tqdm's format_dict that vLLM actually reads (elapsed time)."""
        return {"elapsed": time.time() - self._start}

    def update(self, n=1):
        """Called by vLLM after each completed request; relays progress, throttled."""
        self.n += n
        now = time.time()
        is_last = bool(self.total) and self.n >= self.total
        if is_last or (now - self._last_emit) >= self._MIN_INTERVAL:
            self._last_emit = now
            current, total = self._base_count + self.n, self._grand_total
            elapsed = now - self._start
            pct = (current / total * 100) if total else 0.0
            rate = (self.n / elapsed) if elapsed > 0 else 0.0
            # vLLM sets self.postfix (est. speed input/output toks/s) right before
            # calling update() -- see _run_engine in vllm/entrypoints/llm.py -- so
            # the same token-throughput numbers its own bar showed are already here.
            postfix_str = f" | {self.postfix}" if self.postfix else ""
            # Always print, independent of the GUI callback below -- replacing
            # vLLM's own tqdm bar with this object silently killed terminal
            # visibility for Phase 1; both channels must work regardless of
            # whether a GUI is attached. Restores the same stats vLLM's own bar
            # showed (percentage, count, rate, token throughput) minus the ASCII
            # bar graphic, which is tqdm's own rendering with no data to reuse.
            print(f"[T3] Phase 1: {pct:.0f}% | {current}/{total} | {rate:.1f} it/s{postfix_str}")
            if self._on_update:
                self._on_update(current, total)

    def refresh(self):
        """No-op; vLLM calls this once all requests finish."""

    def close(self):
        """No-op; vLLM calls this when generation finishes."""


class VllmBatchProcessor:
    """Phase 1 batch processor using T3-only vLLM model.

    Loads ChatterboxTTS with load_t3_only=True (skips S3Gen/VE, saves ~0.5GB VRAM).
    Groups chunks by TTS parameters, batch-generates speech tokens, saves
    chunks_tokens.json, and returns in-memory token dict.

    Usage:
        processor = VllmBatchProcessor(ckpt_dir, device="cuda")
        tokens_dict = processor.process_chunks(chunks_info_path)
        # tokens_dict = {chunk_id: [token_list_for_each_segment], ...}
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        target_device: str = "cuda",
        max_model_len: int = 1000,
        max_batch_size: int = 10,
        variant: str = "english",
        language_id: Optional[str] = None,
        vllm_kwargs: Optional[dict] = None,
    ):
        """Initializes a VllmBatchProcessor instance with model configuration and parameters.

        Args:
            ckpt_dir: Directory passed to ChatterboxTTS.from_local for VE/S3Gen/conds.
            target_device: CUDA device string for T3 cond pieces.
            max_model_len: vLLM max model length.
            max_batch_size: Hint for vLLM memory heuristic.
            variant: "english" or "multilingual"; multilingual prepends <language_id>.
            language_id: ISO code for multilingual T3 (default config.T3_LANGUAGE / en).
            vllm_kwargs: Extra kwargs forwarded to ChatterboxTTS.from_local.
        """
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.max_batch_size = max_batch_size
        self.variant = variant
        self.language_id = (
            language_id or getattr(runtime_config, "T3_LANGUAGE", "en")
        ).lower()
        self.vllm_kwargs = vllm_kwargs or {}
        self.ckpt_dir = str(ckpt_dir)

        logger.info(
            "Loading ChatterboxTTS (T3-only) from %s on %s",
            ckpt_dir,
            target_device,
        )
        self.model = ChatterboxTTS.from_local(
            ckpt_dir=self.ckpt_dir,
            target_device=self.target_device,
            max_model_len=self.max_model_len,
            max_batch_size=self.max_batch_size,
            variant=self.variant,
            load_t3_only=True,
            **self.vllm_kwargs,
        )
        logger.info("T3-only model loaded successfully")

    def process_chunks(
        self,
        json_path: Path,
        cond_emb: Optional[torch.Tensor] = None,
        use_vader: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Dict[str, List[List[int]]], List[dict]]:
        """Process all chunks through T3 token generation.

        Args:
            json_path: Path to chunks_info.json.
            cond_emb: Pre-computed condition embedding (from compute_conditionals).
            use_vader: Apply VADER emotion analysis.
            progress_callback: fn(segments_processed, total_segments), called live
                as vLLM finishes each individual request (throttled), via _ProgressRelay.

        Returns:
            Tuple of (tokens_dict, chunks_data):
                tokens_dict: {chunk_id: [list_of_token_lists_per_segment]}
                chunks_data: Updated chunks_data list with speech_tokens embedded.
        """
        if not json_path.exists():
            raise FileNotFoundError(f"chunks_info.json not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        if not chunks_data:
            logger.warning("chunks_info.json is empty")
            return {}, []

        logger.info("Loaded %d chunks from %s", len(chunks_data), json_path)

        chunk_metadata = []
        for chunk_info in chunks_data:
            # NOTE: ids start at 0, so falsy checks ("or", "not id") would drop chunk 0
            chunk_id = chunk_info.get("chunk_id")
            if chunk_id is None:
                chunk_id = chunk_info.get("index")
            text = chunk_info.get("text", "")
            if chunk_id is None or not text:
                logger.warning("Skipping chunk with missing id or text")
                continue

            tts_params = (chunk_info.get("tts_params") or {}).copy()
            tts_params.setdefault("exaggeration", 0.5)
            tts_params.setdefault("temperature", 0.8)
            tts_params.setdefault("top_p", 0.8)
            tts_params.setdefault("repetition_penalty", 2.0)

            boundary_type = chunk_info.get("boundary_type", "period")

            # Parsed once here (not per-batch below) so total_segments -- the real
            # unit vLLM counts progress in -- is known before generation starts.
            # `pauses` (one fewer entry than `segments`) survives into chunks_data
            # below so VllmDecoder can insert real silence between segments.
            segments, pauses = parse_pause_tags(text)

            chunk_metadata.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "segments": segments,
                    "pauses": pauses,
                    "tts_params": tts_params,
                    "boundary_type": boundary_type,
                    "original_index": chunk_info.get("index"),
                }
            )

        param_groups = _group_chunks_by_params(chunk_metadata)
        logger.info(
            "Grouped %d chunks into %d param groups",
            len(chunk_metadata),
            len(param_groups),
        )

        tokens_dict: Dict[str, List[List[int]]] = {}
        total_segments = sum(len(m["segments"]) for m in chunk_metadata)
        segments_done_so_far = 0

        for batch_num, (param_key, batch_chunks) in enumerate(param_groups.items(), 1):
            batch_params = dict(param_key)
            batch_size = len(batch_chunks)
            logger.info(
                "Processing batch %d/%d: %d chunks",
                batch_num,
                len(param_groups),
                batch_size,
            )

            all_segments = []
            chunk_segment_counts = []

            for chunk in batch_chunks:
                segments = chunk["segments"]
                chunk_segment_counts.append(len(segments))
                all_segments.extend(segments)

            try:
                turbo_like_params = batch_params.copy()
                turbo_like_params.setdefault("exaggeration", 0.0)
                # Book-level language comes from T3_LANGUAGE, not per-chunk VADER params.
                turbo_like_params.pop("language_id", None)
                turbo_like_params.pop("language", None)

                # Real per-request progress from vLLM itself, relayed to progress_callback
                # as generation happens -- not an after-the-fact estimate. Falls back to
                # vLLM's own terminal bar when nothing is listening (e.g. CLI/debug use).
                # Turbo T3 conds are 376 tokens each. One vLLM generate() of a
                # whole book overflows the ~8192-token encoder cache and CUDA-aborts.
                gen_batch = 8 if self.variant == "turbo" else len(all_segments)
                all_token_lists = []
                for seg_i in range(0, len(all_segments), gen_batch):
                    piece = all_segments[seg_i : seg_i + gen_batch]
                    use_tqdm = (
                        functools.partial(
                            _ProgressRelay,
                            on_update=progress_callback,
                            base_count=segments_done_so_far + seg_i,
                            grand_total=total_segments,
                        )
                        if progress_callback
                        else True
                    )
                    all_token_lists.extend(
                        self.model.generate_speech_tokens(
                            piece,
                            cond_emb=cond_emb,
                            use_tqdm=use_tqdm,
                            language_id=self.language_id,
                            **turbo_like_params,
                        )
                    )

                segment_idx = 0
                for idx, chunk in enumerate(batch_chunks):
                    segment_count = chunk_segment_counts[idx]
                    chunk_tokens = []
                    for _ in range(segment_count):
                        tokens = all_token_lists[segment_idx]
                        chunk_tokens.append(tokens)
                        segment_idx += 1

                    tokens_dict[chunk["chunk_id"]] = chunk_tokens

            except Exception as e:
                logger.error(
                    "Failed to generate tokens for batch %d: %s",
                    batch_num,
                    e,
                    exc_info=True,
                )
                raise

            segments_done_so_far += len(all_segments)

            if batch_num % 10 == 0:
                gc.collect()

        meta_by_id = {m["chunk_id"]: m for m in chunk_metadata}

        for chunks_data_item in chunks_data:
            # Same falsy-zero hazard as above: id 0 must not be dropped
            cid = chunks_data_item.get("chunk_id")
            if cid is None:
                cid = chunks_data_item.get("index")
            if cid in tokens_dict:
                chunks_data_item["speech_tokens"] = [
                    t.tolist() if torch.is_tensor(t) else t for t in tokens_dict[cid]
                ]
                m = meta_by_id.get(cid, {})
                chunks_data_item["pauses"] = m.get("pauses", [])
                # Same pre-split segments Phase 1 generated from -- Phase 3 regen
                # reuses these directly instead of re-parsing pause tags itself.
                chunks_data_item["segments"] = m.get("segments", [])

        tokens_json_path = json_path.parent / "chunks_tokens.json"
        with open(tokens_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        logger.info(
            "Saved tokens for %d chunks to %s", len(tokens_dict), tokens_json_path
        )

        return tokens_dict, chunks_data

    def shutdown(self):
        """Release one Phase-1 model even after an earlier pipeline failure."""
        model = getattr(self, "model", None)
        try:
            if model is not None:
                model.shutdown()
        finally:
            # Make repeat cleanup safe for both normal phase transitions and errors.
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
        logger.info("VllmBatchProcessor shut down")
