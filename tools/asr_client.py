"""
ASR Client - Communicates with ASR daemon via file-based queue.
Used by main process to submit validation tasks and collect results.

This client manages the ASR daemon subprocess and uses file-based IPC
for efficient cross-environment communication.
"""
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Optional


class ASRDaemonClient:
    """Client for communicating with ASR worker daemon."""

    def __init__(
        self,
        base_dir: Path,
        num_workers: int = 4,
        model_size: str = "base",
        device: str = "cpu",
        backend: str = "faster_whisper",
    ):
        """Configures the ASR daemon client (does not start the subprocess yet).
        Args:
        base_dir: Book-specific working dir; queue/results/log files live under it.
        num_workers: Worker process count for the daemon's pool.
        model_size: Whisper model name (tiny/base/small/medium/large*), typically
            resolved from the GUI's SAFE/MODERATE/INSANE tier.
        device: "cpu" (default, never contends with TTS for VRAM) or "cuda"
            (Tab 2's "Run ASR on GPU" checkbox; the daemon falls back to CPU on
            its own if CUDA load fails for any reason).
        backend: faster_whisper, whisper_cpp, or parakeet (Stage 1 only).
        Returns:
        None
        """
        self.base_dir = Path(base_dir)
        self.queue_dir = self.base_dir / "asr_queue"
        self.results_dir = self.base_dir / "asr_results"
        self.num_workers = num_workers
        self.model_size = model_size
        self.device = device
        self.backend = backend or "faster_whisper"
        self.daemon_process = None
        self.pending_chunks = set()

    def start_daemon(self):
        """Start the ASR daemon subprocess (runs in ASR venv)."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Clear old results
        for f in self.results_dir.glob("*.result.json"):
            f.unlink()

        # Clear old shutdown signal (prevents immediate daemon shutdown)
        shutdown_file = self.base_dir / "asr_daemon.shutdown"
        if shutdown_file.exists():
            shutdown_file.unlink()
            logging.info("Removed stale shutdown signal file")

        # Find ASR daemon script and Python (use main venv — unified environment)
        asr_dir = Path(__file__).parent.parent / "ASR"
        daemon_script = asr_dir / "asr_daemon.py"
        asr_python = sys.executable

        if not daemon_script.exists():
            raise RuntimeError(f"ASR daemon script not found: {daemon_script}")

        # Start daemon
        cmd = [
            str(asr_python),
            str(daemon_script),
            "--queue-dir", str(self.queue_dir),
            "--results-dir", str(self.results_dir),
            "--workers", str(self.num_workers),
            "--model-size", str(self.model_size),
            "--device", str(self.device),
            "--backend", str(self.backend),
        ]

        logging.info(
            "Starting ASR daemon with %s workers (backend: %s, model: %s, device: %s)...",
            self.num_workers,
            self.backend,
            self.model_size,
            self.device,
        )
        logging.info(f"Command: {' '.join(cmd)}")

        # Create log file for daemon output (prevents pipe blocking)
        log_file = self.base_dir / "asr_daemon.log"
        logging.info(f"Daemon logs: {log_file}")

        # Stage 2 replaces Stage 1 in one run; append so its start cannot erase
        # the failure traceback needed to diagnose the preceding daemon.
        self.daemon_log_file = open(log_file, "a", buffering=1)  # Line buffered
        self.daemon_log_file.write(
            "\n=== ASR daemon start "
            f"backend={self.backend} model={self.model_size} "
            f"requested_device={self.device} workers={self.num_workers} ===\n"
        )

        self.daemon_process = subprocess.Popen(
            cmd,
            stdout=self.daemon_log_file,    # Write to file, no pipe blocking
            stderr=subprocess.STDOUT,        # Merge stderr to stdout
            text=True
        )

        # Wait for daemon to be ready (check for PID file)
        pid_file = self.base_dir / "asr_daemon.pid"
        max_wait = 30
        start = time.time()
        while not pid_file.exists() and (time.time() - start) < max_wait:
            # Check if process is still running
            if self.daemon_process.poll() is not None:
                # Process exited, close log and read it
                self.daemon_log_file.close()
                with open(log_file, 'r') as f:
                    output = f.read()
                raise RuntimeError(f"ASR daemon failed to start. Log:\n{output}")
            time.sleep(0.5)

        if not pid_file.exists():
            raise RuntimeError("ASR daemon failed to start (timeout)")

        logging.info("✅ ASR daemon started successfully")

    def restart_with_model(
        self,
        model_size: str,
        num_workers: Optional[int] = None,
        backend: Optional[str] = None,
    ):
        """Shut down the current daemon and start another engine/size.

        Used to swap Stage 1 for Stage 2, or to bring ASR back after regen with
        a single worker so it does not sit next to T3.

        Args:
            model_size: Whisper model name or Parakeet id for the replacement daemon.
            num_workers: Optional worker-process count override for the new daemon.
            backend: Optional engine swap (faster_whisper, whisper_cpp, parakeet).
        """
        self.shutdown()
        self.model_size = model_size
        if num_workers is not None:
            self.num_workers = int(num_workers)
        if backend:
            self.backend = backend
        self.pending_chunks = set()
        self.start_daemon()

    def submit(self, chunk_id: str, wav_path: Path, expected_text: str, threshold: float):
        """Submit a validation task (non-blocking)."""
        task_data = {
            'chunk_id': chunk_id,
            'wav_path': str(wav_path.resolve()),
            'expected_text': expected_text,
            'threshold': threshold
        }

        task_file = self.queue_dir / f"{chunk_id}.task.json"
        task_file.write_text(json.dumps(task_data, indent=2))
        self.pending_chunks.add(chunk_id)
        logging.debug(f"Submitted ASR task: {chunk_id}")

    def daemon_alive(self) -> bool:
        """Return True when the daemon subprocess is still running."""
        if self.daemon_process is None:
            return False
        return self.daemon_process.poll() is None

    def get_result(self, chunk_id: str, timeout: float = 120) -> Optional[dict]:
        """Get result for a chunk (blocking with timeout)."""
        result_file = self.results_dir / f"{chunk_id}.result.json"
        start = time.time()

        while (time.time() - start) < timeout:
            if result_file.exists():
                result = json.loads(result_file.read_text())
                result_file.unlink()  # Clean up
                self.pending_chunks.discard(chunk_id)
                return result
            if not self.daemon_alive():
                return {
                    "chunk_id": chunk_id,
                    "passed": False,
                    "score": 0.0,
                    "error": "ASR daemon exited before returning a result",
                    "asr_text": "",
                }
            time.sleep(0.1)

        return None  # Timeout

    def shutdown(self):
        """Shutdown the ASR daemon."""
        try:
            if self.daemon_process:
                shutdown_file = self.base_dir / "asr_daemon.shutdown"
                shutdown_file.touch()
                logging.info("Waiting for ASR daemon to shutdown...")
                try:
                    self.daemon_process.wait(timeout=10)
                    logging.info("ASR daemon stopped")
                except subprocess.TimeoutExpired:
                    logging.warning("ASR daemon did not stop within 10s, terminating...")
                    self.daemon_process.terminate()
                    try:
                        self.daemon_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logging.warning("ASR daemon did not terminate, killing...")
                        self.daemon_process.kill()
                        self.daemon_process.wait()
        finally:
            self.daemon_process = None
            if hasattr(self, 'daemon_log_file'):
                try:
                    self.daemon_log_file.close()
                except Exception:
                    pass
