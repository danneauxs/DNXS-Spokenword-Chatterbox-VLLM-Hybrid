"""Build chapter maps and export chapterized M4B / MP3 / WAV deliverables.

M4B is encoded from the ordered chunk WAV list (no intermediate full WAV)
unless write_wav was requested. Multi-chapter M4B encodes chapters in
parallel then remuxes. Peak normalization is a constant gain, not EBU
loudnorm.
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.audio_export import (
    aac_encode_args,
    concat_wavs,
    export_worker_count,
    measure_pcm_peak,
    peak_volume_filter,
    run_ffmpeg_checked,
    wav_duration_seconds,
    write_concat_list,
)
from modules.chapter_headers import assign_chapter_ids

logger = logging.getLogger(__name__)

CHAPTER_MODE_HEADINGS_ONLY = "headings_only"
CHAPTER_MODE_HEADINGS_OR_MINUTES = "headings_or_minutes"
CHAPTER_MODE_HEADINGS_WITH_MAX = "headings_with_max"
_CHAPTER_MODES = {
    CHAPTER_MODE_HEADINGS_ONLY,
    CHAPTER_MODE_HEADINGS_OR_MINUTES,
    CHAPTER_MODE_HEADINGS_WITH_MAX,
}


def resolve_chapter_mode(
    chapter_mode: Optional[str], max_chapter_minutes: float | int
) -> str:
    """Return a supported chapter strategy, mapping legacy minute-cap configs.

    Args:
        chapter_mode: Explicit strategy name, or empty for legacy inference.
        max_chapter_minutes: Positive values used to mean split-long-chapters.

    Returns:
        One of headings_only, headings_or_minutes, headings_with_max.
    """
    mode = str(chapter_mode or "").strip().lower()
    if mode in _CHAPTER_MODES:
        return mode
    return (
        CHAPTER_MODE_HEADINGS_WITH_MAX
        if float(max_chapter_minutes or 0) > 0
        else CHAPTER_MODE_HEADINGS_ONLY
    )


def build_chunk_timeline(
    chunks: Sequence[Dict[str, Any]],
    audio_chunks_dir: Path,
) -> List[Dict[str, Any]]:
    """Attach start/end seconds to each chunk from per-chunk WAV durations.

    Args:
        chunks: Ordered chunk records with an integer index.
        audio_chunks_dir: Directory containing chunk_XXXXX.wav files.

    Returns:
        Timeline rows including wav_path, start_s, end_s, duration_s.

    Raises:
        FileNotFoundError: A required chunk WAV is missing.
    """
    audio_chunks_dir = Path(audio_chunks_dir)
    timeline: List[Dict[str, Any]] = []
    cursor = 0.0
    for chunk in chunks:
        index = int(chunk.get("index", len(timeline)))
        wav_path = audio_chunks_dir / f"chunk_{index:05d}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio chunk: {wav_path}")
        duration = wav_duration_seconds(wav_path)
        row = dict(chunk)
        row["start_s"] = cursor
        row["end_s"] = cursor + duration
        row["duration_s"] = duration
        row["wav_path"] = str(wav_path)
        timeline.append(row)
        cursor += duration
    return timeline


def group_chapters_from_timeline(
    timeline: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse ordered timeline rows into chapter ranges (id 0, 1, 2, …).

    Args:
        timeline: Rows from build_chunk_timeline.

    Returns:
        Chapter dicts with title, times, chunk indices, and wav_paths.
    """
    if not timeline:
        return []
    groups: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for row in timeline:
        cid = int(row.get("chapter_id") or 0)
        title = str(
            row.get("chapter_title")
            or ("Front matter" if cid == 0 else f"Chapter {cid}")
        )
        if current is None or current["chapter_id"] != cid:
            current = {
                "chapter_id": cid,
                "title": title if cid > 0 else "Front matter",
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "start_chunk": int(row["index"]),
                "end_chunk": int(row["index"]),
                "chunk_indices": [int(row["index"])],
                "wav_paths": [row["wav_path"]],
                "boundary_types": [row.get("boundary_type") or ""],
            }
            groups.append(current)
        else:
            current["end_s"] = float(row["end_s"])
            current["end_chunk"] = int(row["index"])
            current["chunk_indices"].append(int(row["index"]))
            current["wav_paths"].append(row["wav_path"])
            current["boundary_types"].append(row.get("boundary_type") or "")
    return groups


def subdivide_chapters_by_minutes(
    timeline: Sequence[Dict[str, Any]],
    chapters: Sequence[Dict[str, Any]],
    max_minutes: float,
) -> List[Dict[str, Any]]:
    """Split long chapters near paragraph/sentence ends, else hard cut.

    Args:
        timeline: Chunk rows with start_s/end_s/boundary_type.
        chapters: Chapter groups from group_chapters_from_timeline.
        max_minutes: Soft max length; 0 disables subdivision.

    Returns:
        Possibly expanded chapter list with continuation titles.
    """
    if not max_minutes or max_minutes <= 0:
        return list(chapters)

    max_s = float(max_minutes) * 60.0
    by_index = {int(row["index"]): row for row in timeline}
    expanded: List[Dict[str, Any]] = []

    for chapter in chapters:
        duration = float(chapter["end_s"]) - float(chapter["start_s"])
        if duration <= max_s + 0.5:
            expanded.append(dict(chapter))
            continue

        indices = list(chapter["chunk_indices"])
        part = 1
        seg_start_idx = 0
        seg_start_s = float(chapter["start_s"])
        target_end = seg_start_s + max_s

        def emit_segment(end_i: int) -> None:
            """Append one segment covering indices [seg_start_idx, end_i]."""
            nonlocal part, seg_start_idx, seg_start_s, target_end
            slice_idx = indices[seg_start_idx : end_i + 1]
            if not slice_idx:
                return
            first = by_index[slice_idx[0]]
            last = by_index[slice_idx[-1]]
            base_title = chapter["title"]
            title = base_title if part == 1 else f"{base_title} (cont. {part})"
            expanded.append(
                {
                    "chapter_id": chapter["chapter_id"],
                    "title": title,
                    "start_s": float(first["start_s"]),
                    "end_s": float(last["end_s"]),
                    "start_chunk": slice_idx[0],
                    "end_chunk": slice_idx[-1],
                    "chunk_indices": slice_idx,
                    "wav_paths": [by_index[i]["wav_path"] for i in slice_idx],
                    "boundary_types": [
                        by_index[i].get("boundary_type") or "" for i in slice_idx
                    ],
                    "artificial": part > 1,
                }
            )
            part += 1
            seg_start_idx = end_i + 1
            if seg_start_idx < len(indices):
                seg_start_s = float(by_index[indices[seg_start_idx]]["start_s"])
                target_end = seg_start_s + max_s

        i = 0
        while i < len(indices):
            row = by_index[indices[i]]
            end_s = float(row["end_s"])
            is_last = i == len(indices) - 1
            if end_s + 0.01 >= target_end or is_last:
                cut = i
                if not is_last:
                    for j in range(i, min(len(indices), i + 40)):
                        bt = str(by_index[indices[j]].get("boundary_type") or "")
                        if bt in {"paragraph_end", "paragraph_break", "chapter_start"}:
                            cut = j
                            break
                        if bt in {"period", "sentence_end"}:
                            cut = j
                emit_segment(cut)
                i = cut + 1
                continue
            i += 1

        if seg_start_idx < len(indices):
            emit_segment(len(indices) - 1)

    for order, chapter in enumerate(expanded):
        chapter["order"] = order
    return expanded


def build_chapter_export_plan(
    chunks: List[Dict[str, Any]],
    audio_chunks_dir: Path,
    *,
    chapterize: bool,
    max_chapter_minutes: float | int,
    chapter_mode: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, bool]:
    """Build export timeline and chapter groups for the selected strategy.

    headings_or_minutes keeps heading chapters intact and uses minutes only
    when no heading exists. headings_with_max splits long headed chapters.

    Args:
        chunks: Chunk records (chapter_id stamped if chapterize is True).
        audio_chunks_dir: Directory of chunk_XXXXX.wav files.
        chapterize: Enable a chapter TOC rather than one full-book chapter.
        max_chapter_minutes: Minute value for the selected strategy.
        chapter_mode: headings_only, headings_or_minutes, or headings_with_max.

    Returns:
        Timeline rows, chapter groups, resolved strategy name, headings_found.
    """
    resolved_mode = resolve_chapter_mode(chapter_mode, max_chapter_minutes)
    planned_chunks = [dict(chunk) for chunk in chunks]
    headings_found = False
    if chapterize:
        planned_chunks = assign_chapter_ids(planned_chunks)
        headings_found = any(
            int(chunk.get("chapter_id") or 0) > 0 for chunk in planned_chunks
        )
    else:
        for chunk in planned_chunks:
            chunk["chapter_id"] = 0
            chunk["chapter_title"] = "Full book"
            chunk["chapter_number"] = None

    timeline = build_chunk_timeline(planned_chunks, audio_chunks_dir)
    chapters = group_chapters_from_timeline(timeline)
    should_split = (
        chapterize
        and float(max_chapter_minutes or 0) > 0
        and (
            resolved_mode == CHAPTER_MODE_HEADINGS_WITH_MAX
            or (resolved_mode == CHAPTER_MODE_HEADINGS_OR_MINUTES and not headings_found)
        )
    )
    if should_split:
        chapters = subdivide_chapters_by_minutes(
            timeline, chapters, float(max_chapter_minutes)
        )
    return timeline, chapters, resolved_mode, headings_found


def write_ffmetadata(
    chapters: Sequence[Dict[str, Any]],
    path: Path,
    *,
    title: str = "",
    artist: str = "",
    extra_tags: Optional[Dict[str, str]] = None,
) -> Path:
    """Write an FFMETADATA1 file with chapter start/end times in milliseconds.

    Args:
        chapters: Chapter groups with start_s/end_s/title.
        path: Destination chapters.ffmetadata path.
        title: Book title tag.
        artist: Artist tag.
        extra_tags: Optional extra key/value metadata from book.nfo.

    Returns:
        The written path.
    """
    path = Path(path)
    lines = [";FFMETADATA1"]
    if title:
        lines.append(f"title={_ffmeta_escape(title)}")
    if artist:
        lines.append(f"artist={_ffmeta_escape(artist)}")
    for key, value in (extra_tags or {}).items():
        if key.lower() in {"title", "artist"}:
            continue
        if value:
            lines.append(f"{key}={_ffmeta_escape(str(value))}")
    for chapter in chapters:
        start_ms = int(round(float(chapter["start_s"]) * 1000))
        end_ms = int(round(float(chapter["end_s"]) * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={_ffmeta_escape(str(chapter.get('title') or 'Chapter'))}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ffmeta_escape(value: str) -> str:
    """Escape special characters for FFMETADATA1 values."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", " ")
    )


def parse_nfo_tags(nfo_path: Optional[Path]) -> Dict[str, str]:
    """Parse key: value lines from a book.nfo file into metadata tags.

    Args:
        nfo_path: Optional path to book.nfo.

    Returns:
        Dict of tag names to values. Empty if the file is missing.
    """
    tags: Dict[str, str] = {}
    if nfo_path is None or not Path(nfo_path).exists():
        return tags
    with open(nfo_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            key = key.strip()
            val = val.strip()
            if key and val:
                tags[key] = val
    return tags


def _normalization_af(
    *,
    enable_normalization: bool,
    normalization_type: str,
    target_peak_db: float,
    speed: float,
    peak: Optional[float] = None,
) -> Optional[str]:
    """Return ffmpeg -af string for configured normalization plus optional atempo.

    Peak uses a constant gain from the measured PCM peak. Loudness keeps EBU
    loudnorm. Simple is a fixed volume offset.

    Args:
        enable_normalization: Master toggle from config.
        normalization_type: none, peak, loudness, or simple.
        target_peak_db: Target dBFS for peak/simple.
        speed: Playback atempo multiplier; 1.0 skips the filter.
        peak: Measured 0..1 peak used only for peak mode.

    Returns:
        Combined -af string, or None when no filter is needed.
    """
    filters: List[str] = []
    norm = str(normalization_type or "peak").lower()
    if enable_normalization and norm not in {"none", ""}:
        if norm == "peak":
            vol = peak_volume_filter(float(peak or 0.0), target_peak_db)
            if vol:
                filters.append(vol)
        elif norm == "simple":
            filters.append(f"volume={target_peak_db}dB")
        elif norm == "loudness":
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if abs(float(speed) - 1.0) >= 0.01:
        filters.append(f"atempo={speed}")
    return ",".join(filters) if filters else None


def mux_m4b(
    audio_input: Path,
    output_m4b: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    chapters_ffmetadata: Optional[Path] = None,
    cover_path: Optional[Path] = None,
    concat_list: bool = False,
    already_aac: bool = False,
    adts: bool = False,
    sample_rate: int = 24000,
    normalization_af: Optional[str] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Encode or remux audio into an M4B, optionally with chapters and cover.

    Args:
        audio_input: WAV path, AAC path, or concat list when concat_list is True.
        output_m4b: Destination M4B path.
        ffmpeg_path: FFmpeg executable.
        chapters_ffmetadata: Optional FFMETADATA1 file.
        cover_path: Optional JPEG/PNG cover.
        concat_list: Treat audio_input as an ffmpeg concat demuxer list.
        already_aac: Stream-copy audio instead of encoding.
        adts: Apply aac_adtstoasc when copying concatenated ADTS AAC.
        sample_rate: AAC sample rate; 0 leaves the source rate.
        normalization_af: Optional volume/loudnorm/atempo filter.
        extra_metadata: Extra -metadata key=value tags.

    Returns:
        The written output_m4b path.
    """
    output_m4b = Path(output_m4b)
    output_m4b.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path, "-y"]
    if concat_list:
        cmd.extend(["-f", "concat", "-safe", "0", "-i", str(audio_input)])
    else:
        cmd.extend(["-i", str(audio_input)])
    metadata_index = None
    if chapters_ffmetadata is not None and Path(chapters_ffmetadata).exists():
        metadata_index = 1
        cmd.extend(["-i", str(chapters_ffmetadata)])
    cover_index = None
    if cover_path is not None and Path(cover_path).exists():
        cover_index = 2 if metadata_index is not None else 1
        cmd.extend(["-i", str(cover_path)])
        cmd.extend(["-map", "0:a:0", "-map", f"{cover_index}:v:0"])
    if metadata_index is not None:
        cmd.extend(["-map_metadata", str(metadata_index)])
    if already_aac:
        if cover_index is not None:
            cmd.extend(
                ["-c:a", "copy", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
            )
        else:
            cmd.extend(["-c:a", "copy"])
        if adts:
            cmd.extend(["-bsf:a", "aac_adtstoasc"])
    else:
        if normalization_af:
            cmd.extend(["-af", normalization_af])
        if cover_index is not None:
            cmd.extend(["-c:v", "copy", "-disposition:v:0", "attached_pic"])
        cmd.extend(aac_encode_args(ffmpeg_path, sample_rate))
    for key, value in (extra_metadata or {}).items():
        if value:
            cmd.extend(["-metadata", f"{key}={value}"])
    cmd.append(str(output_m4b))
    run_ffmpeg_checked(cmd)
    return output_m4b


def _encode_chapter_aac(
    chapter: Dict[str, Any],
    dest_aac: Path,
    *,
    ffmpeg_path: str,
    sample_rate: int,
    normalization_af: Optional[str],
) -> Path:
    """Encode one chapter's chunk WAV list to ADTS AAC for later stream-copy mux.

    Args:
        chapter: Chapter dict containing wav_paths.
        dest_aac: Destination .aac path.
        ffmpeg_path: FFmpeg executable.
        sample_rate: Requested AAC sample rate.
        normalization_af: Optional volume/loudnorm filter shared across chapters.

    Returns:
        The written dest_aac path.
    """
    dest_aac = Path(dest_aac)
    dest_aac.parent.mkdir(parents=True, exist_ok=True)
    list_path = dest_aac.with_suffix(".concat.txt")
    try:
        write_concat_list(chapter["wav_paths"], list_path)
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if normalization_af:
            cmd.extend(["-af", normalization_af])
        cmd.extend(aac_encode_args(ffmpeg_path, sample_rate))
        cmd.append(str(dest_aac))
        run_ffmpeg_checked(cmd)
    finally:
        list_path.unlink(missing_ok=True)
    return dest_aac


def mux_m4b_from_parallel_chapters(
    chapters: Sequence[Dict[str, Any]],
    chapters_ffmetadata: Optional[Path],
    output_m4b: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    sample_rate: int = 24000,
    normalization_af: Optional[str] = None,
    cover_path: Optional[Path] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Encode chapters to AAC in parallel, then remux one chapterized M4B.

    Args:
        chapters: Planned chapter groups with wav_paths.
        chapters_ffmetadata: FFMETADATA1 file with chapter timestamps.
        output_m4b: Destination M4B path.
        ffmpeg_path: FFmpeg executable.
        sample_rate: AAC sample rate.
        normalization_af: Shared filter so chapter loudness stays consistent.
        cover_path: Optional JPEG/PNG cover art.
        extra_metadata: Extra ffmpeg metadata tags.

    Returns:
        The written output_m4b path.
    """
    output_m4b = Path(output_m4b)
    output_m4b.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_m4b.parent / f".{output_m4b.stem}.m4b_parts"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        dests = [work_dir / f"ch_{index:03d}.aac" for index in range(len(chapters))]

        def _job(item: Tuple[Dict[str, Any], Path]) -> Path:
            """Encode one chapter AAC part."""
            chapter, dest = item
            return _encode_chapter_aac(
                chapter,
                dest,
                ffmpeg_path=ffmpeg_path,
                sample_rate=sample_rate,
                normalization_af=normalization_af,
            )

        items = list(zip(chapters, dests))
        if len(items) <= 1:
            for item in items:
                _job(item)
        else:
            with ThreadPoolExecutor(max_workers=export_worker_count(len(items))) as pool:
                futures = [pool.submit(_job, item) for item in items]
                for future in as_completed(futures):
                    future.result()
        list_path = work_dir / "chapters.concat.txt"
        write_concat_list([str(path) for path in dests], list_path)
        return mux_m4b(
            list_path,
            output_m4b,
            ffmpeg_path=ffmpeg_path,
            chapters_ffmetadata=chapters_ffmetadata,
            cover_path=cover_path,
            concat_list=True,
            already_aac=True,
            adts=True,
            extra_metadata=extra_metadata,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def encode_mp3_from_chunks(
    wav_paths: Sequence[str],
    output_mp3: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    title: str = "",
    album: str = "",
    artist: str = "",
    cover_path: Optional[Path] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Encode one MP3 by concatenating chunk WAVs in a single ffmpeg pass.

    Args:
        wav_paths: Ordered source WAV paths.
        output_mp3: Destination MP3 path.
        ffmpeg_path: FFmpeg executable.
        title: Title tag.
        album: Album tag.
        artist: Artist tag.
        cover_path: Optional attached cover art.
        extra_metadata: Extra ffmpeg metadata tags.

    Returns:
        The written output_mp3 path.
    """
    output_mp3 = Path(output_mp3)
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_mp3.with_suffix(".concat.txt")
    write_concat_list(wav_paths, list_path)
    try:
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if cover_path is not None and Path(cover_path).exists():
            cmd.extend(["-i", str(cover_path), "-map", "0:a:0", "-map", "1:v:0"])
        cmd.extend(["-codec:a", "libmp3lame", "-q:a", "2"])
        if cover_path is not None and Path(cover_path).exists():
            cmd.extend(["-codec:v", "copy", "-disposition:v:0", "attached_pic"])
        if title:
            cmd.extend(["-metadata", f"title={title}"])
        if album:
            cmd.extend(["-metadata", f"album={album}"])
        if artist:
            cmd.extend(["-metadata", f"artist={artist}"])
        for key, value in (extra_metadata or {}).items():
            if value:
                cmd.extend(["-metadata", f"{key}={value}"])
        cmd.append(str(output_mp3))
        run_ffmpeg_checked(cmd)
    finally:
        list_path.unlink(missing_ok=True)
    return output_mp3


def export_chapter_mp3s(
    chapters: Sequence[Dict[str, Any]],
    mp3_dir: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    album: str = "",
    artist: str = "",
    cover_path: Optional[Path] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Encode one MP3 per chapter in parallel under mp3_dir.

    Args:
        chapters: Chapter groups with wav_paths and titles.
        mp3_dir: Destination directory (created if missing).
        ffmpeg_path: FFmpeg executable.
        album: Album tag, typically the book title.
        artist: Artist tag.
        cover_path: Optional cover art.
        extra_metadata: Extra ffmpeg metadata tags.

    Returns:
        Ordered list of written MP3 paths as strings.
    """
    mp3_dir = Path(mp3_dir)
    mp3_dir.mkdir(parents=True, exist_ok=True)

    def _one(order: int, chapter: Dict[str, Any]) -> Tuple[int, Path]:
        """Encode a single chapter MP3 and return its order and path."""
        cid = int(chapter.get("chapter_id") or 0)
        title = str(chapter.get("title") or f"Chapter {cid}")
        safe = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in title)
        dest = mp3_dir / f"{order + 1:02d}_{safe.strip() or f'chapter_{cid}'}.mp3"
        encode_mp3_from_chunks(
            chapter["wav_paths"],
            dest,
            ffmpeg_path=ffmpeg_path,
            title=title,
            album=album,
            artist=artist,
            cover_path=cover_path,
            extra_metadata={**(extra_metadata or {}), "track": str(order + 1)},
        )
        return order, dest

    written: Dict[int, Path] = {}
    jobs = list(enumerate(chapters))
    if len(jobs) <= 1:
        for order, chapter in jobs:
            idx, path = _one(order, chapter)
            written[idx] = path
    else:
        with ThreadPoolExecutor(max_workers=export_worker_count(len(jobs))) as pool:
            futures = [pool.submit(_one, order, chapter) for order, chapter in jobs]
            for future in as_completed(futures):
                idx, path = future.result()
                written[idx] = path
    return [str(written[i]) for i in range(len(chapters))]


def _load_chunks_records(chunks_json: Path) -> List[Dict[str, Any]]:
    """Load chunk dicts from chunks_info.json, skipping the _metadata sentinel.

    Args:
        chunks_json: Path to chunks_info.json.

    Returns:
        List of text-chunk records with an index field.
    """
    data = json.loads(Path(chunks_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"chunks JSON must be a list: {chunks_json}")
    records = []
    for item in data:
        if not isinstance(item, dict) or item.get("_metadata"):
            continue
        if "text" not in item:
            continue
        records.append(item)
    return records


def export_book_chapters(
    book_dir: Path,
    *,
    chunks_json: Optional[Path] = None,
    audio_chunks_dir: Optional[Path] = None,
    chapterize: bool = False,
    max_chapter_minutes: float | int = 0.0,
    chapter_mode: Optional[str] = None,
    write_m4b: bool = True,
    write_mp3: bool = False,
    write_wav: bool = False,
    title: str = "",
    cover_path: Optional[Path] = None,
    nfo_path: Optional[Path] = None,
    ffmpeg_path: str = "ffmpeg",
    sample_rate: int = 24000,
    enable_normalization: bool = True,
    normalization_type: str = "peak",
    target_peak_db: float = -1.5,
    speed: float = 1.0,
    output_stem: Optional[str] = None,
) -> Dict[str, Any]:
    """Build chapter map and export only the requested formats.

    Args:
        book_dir: Main book output directory (parent of TTS/).
        chunks_json: Path to chunks_info.json.
        audio_chunks_dir: Directory of chunk_XXXXX.wav files.
        chapterize: Enable a chapter TOC rather than one full-book chapter.
        max_chapter_minutes: Minute value for the selected chapter strategy.
        chapter_mode: headings_only, headings_or_minutes, or headings_with_max.
        write_m4b / write_mp3 / write_wav: Which deliverables to create.
        title: Book title fallback.
        cover_path: Optional cover art.
        nfo_path: Optional book.nfo for metadata tags.
        ffmpeg_path: FFmpeg executable.
        sample_rate: Requested AAC sample rate.
        enable_normalization: Master normalization toggle.
        normalization_type: none, peak, loudness, or simple.
        target_peak_db: Peak/simple target in dBFS.
        speed: Playback atempo multiplier.
        output_stem: Filename stem for WAV/M4B/MP3 (without extension).

    Returns:
        Dict with paths and chapter summary.
    """
    book_dir = Path(book_dir)
    tts_dir = book_dir / "TTS"
    chunks_json = Path(chunks_json or tts_dir / "text_chunks" / "chunks_info.json")
    audio_chunks_dir = Path(audio_chunks_dir or tts_dir / "audio_chunks")
    extra_tags = parse_nfo_tags(nfo_path)
    chunks = _load_chunks_records(chunks_json)
    if not chunks:
        raise ValueError(f"No chunks in {chunks_json}")

    timeline, chapters, resolved_mode, headings_found = build_chapter_export_plan(
        chunks,
        audio_chunks_dir,
        chapterize=chapterize,
        max_chapter_minutes=max_chapter_minutes,
        chapter_mode=chapter_mode,
    )
    book_title = title or extra_tags.get("title") or book_dir.name
    book_title = Path(str(book_title)).stem
    stem = output_stem or book_title
    total_duration = timeline[-1]["end_s"] if timeline else 0.0
    all_wav_paths = [row["wav_path"] for row in timeline]
    artist = extra_tags.get("artist") or extra_tags.get("author") or ""

    chapters_json_path = tts_dir / "chapters.json"
    chapters_json_path.write_text(
        json.dumps(
            {
                "title": book_title,
                "total_duration_s": total_duration,
                "chapter_mode": resolved_mode,
                "headings_found": headings_found,
                "chapters": [
                    {
                        "order": i,
                        "chapter_id": c["chapter_id"],
                        "title": c["title"],
                        "start_s": c["start_s"],
                        "end_s": c["end_s"],
                        "start_chunk": c["start_chunk"],
                        "end_chunk": c["end_chunk"],
                    }
                    for i, c in enumerate(chapters)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ffmeta_path = tts_dir / "chapters.ffmetadata"
    write_ffmetadata(
        chapters if chapterize else [],
        ffmeta_path,
        title=book_title,
        artist=artist,
        extra_tags=extra_tags,
    )

    result: Dict[str, Any] = {
        "chapters_json": str(chapters_json_path),
        "ffmetadata": str(ffmeta_path),
        "chapter_count": len(chapters),
        "total_duration_s": total_duration,
        "m4b_path": None,
        "mp3_dir": None,
        "mp3_files": [],
        "wav_path": None,
        "chapter_mode": resolved_mode,
        "headings_found": headings_found,
    }

    wav_path: Optional[Path] = None
    if write_wav:
        wav_path = book_dir / f"{stem}.wav"
        logger.info("Building full WAV from %s chunk files…", len(all_wav_paths))
        concat_wavs(all_wav_paths, wav_path, ffmpeg_path=ffmpeg_path)
        result["wav_path"] = str(wav_path)
        logger.info("Full WAV: %s", wav_path)

    encode_rate = int(sample_rate or 0)
    try:
        import wave as _wave

        with _wave.open(str(all_wav_paths[0]), "rb") as probe:
            if probe.getframerate() == encode_rate:
                encode_rate = 0
    except Exception:
        pass

    peak: Optional[float] = None
    if write_m4b and enable_normalization and str(normalization_type) == "peak":
        logger.info("Measuring PCM peak across %s chunk WAV(s)…", len(all_wav_paths))
        peak = measure_pcm_peak(all_wav_paths)
        logger.info("Measured PCM peak=%.4f", peak)
    af = _normalization_af(
        enable_normalization=enable_normalization,
        normalization_type=normalization_type,
        target_peak_db=target_peak_db,
        speed=speed,
        peak=peak,
    )

    if write_m4b:
        m4b_path = book_dir / f"{stem}.m4b"
        if chapterize and len(chapters) > 1:
            mux_m4b_from_parallel_chapters(
                chapters,
                ffmeta_path,
                m4b_path,
                ffmpeg_path=ffmpeg_path,
                sample_rate=encode_rate,
                normalization_af=af,
                cover_path=cover_path,
                extra_metadata=extra_tags,
            )
            logger.info(
                "M4B written from %s parallel chapter encodes: %s",
                len(chapters),
                m4b_path,
            )
        else:
            list_path = tts_dir / ".m4b_concat.txt"
            try:
                write_concat_list(all_wav_paths, list_path)
                mux_m4b(
                    list_path,
                    m4b_path,
                    ffmpeg_path=ffmpeg_path,
                    chapters_ffmetadata=ffmeta_path if chapterize else None,
                    cover_path=cover_path,
                    concat_list=True,
                    sample_rate=encode_rate,
                    normalization_af=af,
                    extra_metadata=extra_tags,
                )
            finally:
                list_path.unlink(missing_ok=True)
            logger.info("M4B written from chunk list: %s", m4b_path)
        result["m4b_path"] = str(m4b_path)

    if write_mp3:
        if chapterize and len(chapters) > 1:
            mp3_dir = book_dir / "chapters"
            files = export_chapter_mp3s(
                chapters,
                mp3_dir,
                ffmpeg_path=ffmpeg_path,
                album=book_title,
                artist=artist,
                cover_path=cover_path,
                extra_metadata=extra_tags,
            )
            result["mp3_dir"] = str(mp3_dir)
            result["mp3_files"] = files
        else:
            mp3_path = book_dir / f"{stem}.mp3"
            encode_mp3_from_chunks(
                all_wav_paths,
                mp3_path,
                ffmpeg_path=ffmpeg_path,
                title=book_title,
                album=book_title,
                artist=artist,
                cover_path=cover_path,
                extra_metadata=extra_tags,
            )
            result["mp3_files"] = [str(mp3_path)]
    return result
