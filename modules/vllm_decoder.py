from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchaudio
from pydub import AudioSegment

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

vllm_root = project_root / "chatterbox-vllm" / "src"
sys.path.insert(0, str(vllm_root))

from config.config import (
    ENABLE_AUDIO_TRIMMING,
    TURBO_S3GEN_BATCH_SIZE,
    TURBO_S3GEN_BATCH_CAPACITY,
    TURBO_S3GEN_ENABLE_BATCHING,
)
from modules.audio_processor import trim_audio_endpoint, add_contextual_silence_memory
from modules.s3gen_scheduler import S3GenBatchCapacity, S3GenScheduleItem, S3GenScheduler

logger = logging.getLogger(__name__)


def _tensor_to_audio_segment(tensor: torch.Tensor, sample_rate: int = 24000) -> AudioSegment:
    """Converts tensor to AudioSegment object. Builds Turbo wrapper."""
    if tensor.dim() == 2:
        tensor = tensor.squeeze(0)
    audio_np = tensor.cpu().numpy()
    audio_np = (audio_np * 32767).astype("int16")
    return AudioSegment(
        audio_np.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )


def _build_turbo_wrapper(s3gen, ve, device, voice_path):
    """Build TurboS3GenWrapper and load conditionals.

    Matches token_to_audio.py Turbo loading logic.
    """
    class _TurboWrapper:
        """Wrapper class for handling TurboS3Gen and VoiceEncoder."""
        def __init__(self, s3gen_model, ve_model, dev):
            """Initializes wrapper for S3Gen and VoiceEncoder models."""
            self.s3gen = s3gen_model
            self.ve = ve_model
            self.device = dev
            self.sr = 24000
            self._cached_s3gen_ref = None

        def load_conditionals(self, vpath):
            """Loads voice conditionals for Turbo backend."""
            vpath_str = str(vpath)
            logger.info("Turbo loading voice from: %s", vpath_str)
            audio, sr = torchaudio.load(vpath_str)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            with torch.inference_mode():
                s3gen_ref = self.s3gen.embed_ref(
                    ref_wav=audio.squeeze(0),
                    ref_sr=16000,
                    device=self.device,
                )
            self._cached_s3gen_ref = s3gen_ref
            return s3gen_ref

        def shutdown(self):
            """Cleans up resources by deleting models and clearing cache."""
            del self.s3gen
            del self.ve
            torch.cuda.empty_cache()
            gc.collect()

    wrapper = _TurboWrapper(s3gen, ve, device)
    if voice_path:
        wrapper.load_conditionals(voice_path)
    return wrapper


class VllmDecoder:
    """Unified S3Gen decoder supporting standard and turbo backends.

    Standard decoder: full ChatterboxTTS S3Gen (10-25 diffusion steps).
    Turbo decoder: Turbo S3Gen meanflow (2 diffusion steps, 5x faster).
    """

    def __init__(
        self,
        decoder_type: str,
        ckpt_dir: str | Path,
        target_device: str = "cuda",
        voice_path: Optional[Path] = None,
        diffusion_steps: int = 10,
        turbo_diffusion_steps: int = 2,
        turbo_ckpt_dir: Optional[str | Path] = None,
    ):
        """Initializes VLLMDecoder with parameters."""
        if decoder_type not in ("standard", "turbo"):
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        self.decoder_type = decoder_type
        self.target_device = target_device
        self.voice_path = voice_path
        self.diffusion_steps = diffusion_steps
        self.turbo_diffusion_steps = turbo_diffusion_steps
        self._model = None
        self._s3gen_ref = None
        self._sample_rate = 24000

        if decoder_type == "standard":
            self._init_standard(ckpt_dir)
        else:
            self._init_turbo(ckpt_dir, turbo_ckpt_dir)

    def _init_standard(self, ckpt_dir: str | Path):
        """Initializes standard decoder from checkpoint directory."""
        from chatterbox_vllm.tts import ChatterboxTTS

        logger.info("Loading standard decoder from %s", ckpt_dir)
        model = ChatterboxTTS.from_local(
            ckpt_dir=ckpt_dir,
            target_device=self.target_device,
            load_t3_only=False,
        )
        self._model = model
        self._sample_rate = model.sr
        if voice_path := self.voice_path:
            self._s3gen_ref, _ = model.get_audio_conditionals(wav_fpath=str(voice_path))
        else:
            self._s3gen_ref = model.default_conds.gen
        logger.info("Standard decoder ready")

    def _init_turbo(self, ckpt_dir: str | Path, turbo_ckpt_dir: Optional[str | Path]):
        """Initializes TurboS3Gen with checkpoint directory."""
        from src.chatterbox_turbo.models.s3gen import S3Gen as TurboS3Gen
        from src.chatterbox_turbo.models.voice_encoder import VoiceEncoder
        from safetensors.torch import load_file as st_load_file

        if turbo_ckpt_dir is None:
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            candidates = list(hf_cache.glob("models--ResembleAI--chatterbox-turbo/snapshots/*"))
            if not candidates:
                raise RuntimeError("Turbo model not found. Set turbo_ckpt_dir or download.")
            turbo_ckpt_dir = candidates[0]

        turbo_ckpt_dir = Path(turbo_ckpt_dir)
        if not turbo_ckpt_dir.exists():
            raise RuntimeError(f"Turbo checkpoint not found: {turbo_ckpt_dir}")

        device = self.target_device
        logger.info("Loading turbo decoder from %s", turbo_ckpt_dir)

        s3gen = TurboS3Gen(meanflow=True)
        weights_path = turbo_ckpt_dir / "s3gen_meanflow.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"Turbo weights not found: {weights_path}")
        weights = st_load_file(weights_path)
        s3gen.load_state_dict(weights, strict=True)
        s3gen.to(device).eval()

        ve = VoiceEncoder()
        ve.load_state_dict(st_load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        self._model = _build_turbo_wrapper(s3gen, ve, device, self.voice_path)
        self._s3gen_ref = self._model._cached_s3gen_ref
        logger.info("Turbo decoder ready")

    def _decode_turbo_segment_batch(
        self,
        token_lists: List[List[int]],
    ) -> List[AudioSegment]:
        """Decode padded single-segment token lists in one Turbo S3Gen call.

        Args:
            token_lists: Token sequences belonging to one length-compatible batch.

        Returns:
            One raw AudioSegment per input token sequence, with padding removed.

        Raises:
            RuntimeError: If called for the standard decoder.
        """
        if self.decoder_type != "turbo":
            raise RuntimeError("Turbo segment batching requires the Turbo decoder")
        if not token_lists:
            return []

        lengths = torch.tensor(
            [len(tokens) for tokens in token_lists],
            dtype=torch.long,
            device=self.target_device,
        )
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(tokens, dtype=torch.long) for tokens in token_lists],
            batch_first=True,
            padding_value=0,
        ).to(self.target_device)
        ref = self._model._cached_s3gen_ref or self._s3gen_ref
        wavs, _ = self._model.s3gen.inference(
            speech_tokens=padded,
            speech_token_lens=lengths,
            ref_dict=ref,
            n_cfm_timesteps=self.turbo_diffusion_steps,
        )

        max_tokens = max(len(tokens) for tokens in token_lists)
        segments = []
        for index, token_length in enumerate(lengths.tolist()):
            # HiFiGAN output length scales linearly with mel/token length. The
            # batched model returns the padded maximum, so remove its tail here.
            sample_count = max(1, round(wavs.shape[-1] * token_length / max_tokens))
            segments.append(
                _tensor_to_audio_segment(wavs[index, :sample_count], self._sample_rate)
            )
        return segments

    def _decode_tokens_turbo_batched(
        self,
        tokens_dict: Dict[str, List[List[int]]],
        audio_output_dir: Path,
        chunk_meta: Optional[Dict[str, dict]],
        progress_callback: Optional[Callable],
        asr_client,
        asr_threshold: float,
        enable_quality_scoring: bool,
    ) -> Tuple[int, int, Dict[str, float]]:
        """Decode Turbo chunks with length-aware batching and safe fallback.

        Single-segment chunks are scheduled globally by token length. Chunks
        containing inline pause segments remain sequential so pause assembly
        stays exactly on the proven path. A CUDA OOM splits only the failed
        batch, preserving any smaller safe batch opportunity.
        """
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        skipped = 0
        completed = 0
        total = len(tokens_dict)
        local_scores: Dict[str, float] = {}
        pending = []

        for chunk_id, segments in tokens_dict.items():
            chunk_id_int = int(chunk_id)
            chunk_id_str = f"{chunk_id_int:05d}"
            out_path = audio_output_dir / f"chunk_{chunk_id_str}.wav"
            chunk_label = f"chunk_{chunk_id_str}"
            if out_path.exists():
                skipped += 1
                completed += 1
                logger.info("[%s] Already exists, skipping", chunk_label)
                continue
            pending.append((chunk_id, segments, out_path, chunk_label))

        def finish_chunk(chunk_id, segments, out_path, chunk_label, raw_segments):
            """Assemble, export, score, and submit one decoded chunk."""
            nonlocal completed, generated
            meta = (chunk_meta or {}).get(chunk_id, {})
            combined, _debug = self.decode_and_assemble_chunk(
                segments,
                meta.get("pauses", []),
                meta.get("boundary_type", "none"),
                chunk_label=chunk_label,
                predecoded_segments=raw_segments,
            )
            if combined is not None:
                combined.export(str(out_path), format="wav")
                if enable_quality_scoring:
                    from modules.audio_processor import evaluate_chunk_quality
                    local_scores[chunk_id] = evaluate_chunk_quality(
                        combined, reference_text=None, include_spectral=True
                    )
                if asr_client is not None:
                    asr_client.submit(
                        str(chunk_id), out_path, meta.get("text", ""), asr_threshold
                    )
            generated += 1
            completed += 1
            logger.info("[%s] Generated by Turbo S3Gen (%d/%d)", chunk_label, completed, total)
            print(f"[S3Gen] {chunk_label}: batch progress {completed}/{total}")
            if progress_callback:
                progress_callback(completed, total, f"Decoded {completed}/{total}")

        batchable = [item for item in pending if len(item[1]) == 1]
        sequential = [item for item in pending if len(item[1]) != 1]
        scheduler = S3GenScheduler(
            S3GenBatchCapacity(
                max_padded_tokens=dict(TURBO_S3GEN_BATCH_CAPACITY),
                max_batch_size=max(1, int(TURBO_S3GEN_BATCH_SIZE)),
            )
        )
        schedule_items = [
            S3GenScheduleItem(
                item_id=str(item[0]),
                token_length=len(item[1][0]),
                payload=item,
            )
            for item in batchable
        ]
        scheduled_batches = scheduler.schedule(schedule_items)

        for scheduled_batch in scheduled_batches:
            group = [entry.payload for entry in scheduled_batch]
            token_lists = [item[1][0] for item in group]
            if len(group) == 1:
                sequential.extend(group)
                continue
            logger.info(
                "[S3Gen] Turbo batch size=%d token_range=%d-%d",
                len(group), len(token_lists[0]), len(token_lists[-1]),
            )
            try:
                raw_segments = self._decode_turbo_segment_batch(token_lists)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "Turbo S3Gen batch OOM at token lengths %d-%d; "
                    "splitting failed batch",
                    len(token_lists[0]),
                    len(token_lists[-1]),
                )
                torch.cuda.empty_cache()
                gc.collect()
                if len(group) == 2:
                    sequential.extend(group)
                else:
                    # Preserve batching for any pair that still fits after a
                    # runtime memory change or allocator fragmentation.
                    sequential.extend(group[:1])
                    retry_item = group[1:]
                    retry_lists = [item[1][0] for item in retry_item]
                    try:
                        raw_segments = self._decode_turbo_segment_batch(retry_lists)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        gc.collect()
                        sequential.extend(retry_item)
                    else:
                        for item, raw_segment in zip(retry_item, raw_segments):
                            finish_chunk(*item, [raw_segment])
                continue
            for item, raw_segment in zip(group, raw_segments):
                finish_chunk(*item, [raw_segment])

        for item in sequential:
            chunk_id, segments, out_path, chunk_label = item
            finish_chunk(chunk_id, segments, out_path, chunk_label, None)

        logger.info("Decoded %d generated, %d skipped", generated, skipped)
        return generated, skipped, local_scores

    def decode_and_assemble_chunk(
        self,
        segments: List[List[int]],
        pauses: List[float],
        boundary_type: str,
        chunk_label: str = "chunk",
        predecoded_segments: Optional[List[AudioSegment]] = None,
    ) -> Tuple[Optional[AudioSegment], dict]:
        """Decodes one chunk's token segments to a single assembled AudioSegment.

        Shared by decode_tokens (Phase 2's main loop) and the Phase 3 regeneration
        path, so both always assemble audio identically: each raw segment is
        trimmed immediately after S3Gen decode (before any deliberate silence
        exists to accidentally strip), THEN pause-tag silence is inserted between
        the now-trimmed segments, THEN contextual boundary silence is appended
        once at the very end.

        Args:
            segments: One token list per text segment (from pause-tag splitting).
            pauses: Pause durations in seconds between segments (len == len(segments)-1).
            boundary_type: Chapter/paragraph/punctuation boundary for trailing silence.
            chunk_label: Used only in the debug summary line.
            predecoded_segments: Optional raw audio segments already decoded by
                a batch call; when supplied, token inference is skipped.

        Returns:
            (assembled_audio_or_None, debug_stats). debug_stats has trimmed_ms,
            pauses_inserted, boundary_ms for the always-on debug print/log.
        """
        sr = self._sample_rate
        combined = None
        trimmed_ms_total = 0
        pauses_inserted = []
        for seg_idx, seg_tokens in enumerate(segments):
            if predecoded_segments is not None:
                seg = predecoded_segments[seg_idx]
            else:
                st = torch.tensor(seg_tokens, device=self.target_device)
                if self.decoder_type == "standard":
                    wav, _ = self._model.s3gen.inference(
                        speech_tokens=st,
                        ref_dict=self._s3gen_ref,
                        n_timesteps=self.diffusion_steps,
                    )
                else:
                    ref = self._model._cached_s3gen_ref or self._s3gen_ref
                    wav, _ = self._model.s3gen.inference(
                        speech_tokens=st,
                        ref_dict=ref,
                        n_cfm_timesteps=self.turbo_diffusion_steps,
                    )
                seg = _tensor_to_audio_segment(wav.cpu(), sr)

            # Trim this raw segment's own endpoint artifacts before any
            # deliberate silence is added anywhere near it.
            if ENABLE_AUDIO_TRIMMING:
                pre_trim_len = len(seg)
                seg = trim_audio_endpoint(seg)
                trimmed_ms_total += pre_trim_len - len(seg)

            if combined is None:
                combined = seg
            else:
                pause_idx = seg_idx - 1
                if pause_idx < len(pauses):
                    pause_ms = int(pauses[pause_idx] * 1000)
                    if pause_ms > 0:
                        combined += AudioSegment.silent(duration=pause_ms)
                        pauses_inserted.append(pause_ms)
                combined += seg

        boundary_ms = 0
        if combined is not None:
            pre_boundary_len = len(combined)
            combined = add_contextual_silence_memory(combined, boundary_type)
            boundary_ms = len(combined) - pre_boundary_len
            # Always-on debug summary -- prints/logs are cheap and this is the
            # only way to see pause/trim/boundary-silence behavior without a
            # manual ASR word-timestamp analysis. Comment out (or gate behind a
            # future config debug flag) if this gets too noisy for large books.
            debug_line = (
                f"[{chunk_label}] trim=-{trimmed_ms_total}ms | "
                f"pauses={pauses_inserted}ms | "
                f"boundary={boundary_type}(+{boundary_ms}ms)"
            )
            logger.info(debug_line)
            print(debug_line)

        return combined, {
            "trimmed_ms": trimmed_ms_total,
            "pauses_inserted": pauses_inserted,
            "boundary_ms": boundary_ms,
        }

    def decode_tokens(
        self,
        tokens_dict: Dict[str, List[List[int]]],
        audio_output_dir: Path,
        chunk_meta: Optional[Dict[str, dict]] = None,
        progress_callback: Optional[Callable] = None,
        asr_client=None,
        asr_threshold: float = 0.75,
        enable_quality_scoring: bool = False,
    ) -> Tuple[int, int, Dict[str, float]]:
        """Decode tokens to audio WAV files.

        Args:
            tokens_dict: {chunk_id: [list_of_token_segment_lists]}.
            audio_output_dir: Output directory for WAVs.
            chunk_meta: {chunk_id: {"pauses": [...], "boundary_type": ..., "text": ...}},
                from VllmBatchProcessor.process_chunks's chunks_data return value.
                Missing/absent chunk_id or field falls back to no pause / no
                boundary silence / no ASR comparison for that chunk.
            progress_callback: fn(chunks_done, total_chunks, message).
            asr_client: Optional ASRDaemonClient; when given, each chunk's WAV is
                submitted for background transcription immediately after export
                (non-blocking -- decode continues to the next chunk right away).
            asr_threshold: Passed through to the ASR daemon's task record only
                (it doesn't gate anything here; the caller decides pass/fail
                after collecting results).
            enable_quality_scoring: When True, computes a local (non-ASR) composite
                quality score per chunk via audio_processor.evaluate_chunk_quality.

        Returns:
            (generated_count, skipped_count, local_scores). local_scores is
            {chunk_id: float}, empty unless enable_quality_scoring is True.
        """
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        skipped = 0
        total = len(tokens_dict)
        local_scores: Dict[str, float] = {}

        if self.decoder_type == "turbo" and TURBO_S3GEN_ENABLE_BATCHING:
            return self._decode_tokens_turbo_batched(
                tokens_dict=tokens_dict,
                audio_output_dir=audio_output_dir,
                chunk_meta=chunk_meta,
                progress_callback=progress_callback,
                asr_client=asr_client,
                asr_threshold=asr_threshold,
                enable_quality_scoring=enable_quality_scoring,
            )

        for idx, (chunk_id, segments) in enumerate(tokens_dict.items()):
            chunk_id_int = int(chunk_id)
            chunk_id_str = f"{chunk_id_int:05d}"
            out_path = audio_output_dir / f"chunk_{chunk_id_str}.wav"
            chunk_start = time.time()
            chunk_label = f"chunk_{chunk_id_str}"
            if out_path.exists():
                skipped += 1
                logger.info("[%s] Already exists, skipping", chunk_label)
                continue

            logger.info("[%s] Generating audio (%d segments)...", chunk_label, len(segments))

            meta = (chunk_meta or {}).get(chunk_id, {})
            pauses = meta.get("pauses", [])
            boundary_type = meta.get("boundary_type", "none")

            combined, _debug = self.decode_and_assemble_chunk(
                segments, pauses, boundary_type, chunk_label=chunk_label
            )

            if combined is not None:
                combined.export(str(out_path), format="wav")

                if enable_quality_scoring:
                    from modules.audio_processor import evaluate_chunk_quality
                    local_scores[chunk_id] = evaluate_chunk_quality(
                        combined, reference_text=None, include_spectral=True
                    )

                if asr_client is not None:
                    expected_text = meta.get("text", "")
                    asr_client.submit(str(chunk_id), out_path, expected_text, asr_threshold)

            generated += 1
            chunk_elapsed = time.time() - chunk_start
            realtime_ratio = chunk_elapsed / (len(combined) / 1000.0) if combined else 0
            logger.info(
                "[%s] Generated in %.2fs (%.2fx realtime, %.2f sec audio)",
                chunk_label, chunk_elapsed, realtime_ratio,
                len(combined) / 1000.0 if combined else 0,
            )
            print(
                f"[S3Gen] {chunk_label}: {chunk_elapsed:.2f}s | "
                f"{realtime_ratio:.2f}x realtime | "
                f"Total {idx + 1}/{total}"
            )

            if progress_callback:
                progress_callback(idx + 1, total, f"Decoded {idx + 1}/{total}")

            if idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        logger.info("Decoded %d generated, %d skipped", generated, skipped)
        return generated, skipped, local_scores

    def shutdown(self):
        """Shuts down the model and releases resources."""
        if self._model is not None:
            self._model.shutdown()
        self._model = None
        self._s3gen_ref = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Decoder shut down")
