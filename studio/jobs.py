import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from studio.hardware import hardware_tracker

HANG_THRESHOLD_SECONDS = 30.0


class JobCancelledError(Exception):
    """Excepción lanzada cuando el usuario cancela un trabajo."""


class JobRecord:
    def __init__(self, kind: str, title: str) -> None:
        now = time.time()
        self.id: str = uuid.uuid4().hex[:10]
        self.kind: str = kind
        self.title: str = title
        self.status: str = "queued"  # queued | running | completed | failed | cancelled
        self.progress: float = 0.0
        self.current_step: str = "En cola..."
        self.speed_text: str = ""
        self.created_ts: float = now
        self.started_ts: Optional[float] = None
        self.finished_ts: Optional[float] = None
        self.last_heartbeat_ts: float = now
        self.logs: List[Dict[str, Any]] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cancel_requested: bool = False
        self._lock = threading.Lock()
        self.log(f"Trabajo creado: {title}")

    def log(
        self,
        message: str,
        level: str = "info",
        progress: Optional[float] = None,
        step: Optional[str] = None,
        speed_text: Optional[str] = None,
    ) -> None:
        now = time.time()
        t_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        with self._lock:
            self.last_heartbeat_ts = now
            if progress is not None:
                self.progress = max(0.0, min(100.0, float(progress)))
            if step is not None:
                self.current_step = step
            if speed_text is not None:
                self.speed_text = speed_text
            self.logs.append(
                {
                    "ts": round(now, 2),
                    "time": t_str,
                    "level": level,
                    "message": message,
                }
            )
            if len(self.logs) > 500:
                self.logs = self.logs[-500:]
        hardware_tracker.mark_activity()

    def heartbeat(
        self,
        progress: Optional[float] = None,
        step: Optional[str] = None,
        speed_text: Optional[str] = None,
    ) -> None:
        """Actualiza progreso/velocidad sin saturar de líneas el registro."""
        now = time.time()
        with self._lock:
            self.last_heartbeat_ts = now
            if progress is not None:
                self.progress = max(0.0, min(100.0, float(progress)))
            if step is not None:
                self.current_step = step
            if speed_text is not None:
                self.speed_text = speed_text
        hardware_tracker.mark_activity()

    def check_cancelled(self) -> None:
        if self.cancel_requested:
            raise JobCancelledError("Cancelado manualmente por el usuario.")

    def to_dict(
        self,
        include_logs: bool = True,
        hang_threshold: float = HANG_THRESHOLD_SECONDS,
    ) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            if self.started_ts is not None:
                end_ref = self.finished_ts if self.finished_ts is not None else now
                elapsed = max(0.0, end_ref - self.started_ts)
            else:
                elapsed = 0.0

            sec_since_update = max(0.0, now - self.last_heartbeat_ts)
            possibly_hung = bool(
                self.status == "running" and sec_since_update >= hang_threshold
            )

            data: Dict[str, Any] = {
                "id": self.id,
                "kind": self.kind,
                "title": self.title,
                "status": self.status,
                "progress": round(self.progress, 1),
                "current_step": self.current_step,
                "speed_text": self.speed_text,
                "created_ts": round(self.created_ts, 2),
                "started_ts": round(self.started_ts, 2) if self.started_ts else None,
                "finished_ts": round(self.finished_ts, 2) if self.finished_ts else None,
                "elapsed_seconds": round(elapsed, 1),
                "seconds_since_update": round(sec_since_update, 1),
                "possibly_hung": possibly_hung,
                "hang_warning": (
                    f"Posible cuelgue: este proceso lleva {int(sec_since_update)} s sin emitir actividad."
                    if possibly_hung
                    else None
                ),
                "cancel_requested": self.cancel_requested,
                "result": self.result,
                "error": self.error,
                "log_count": len(self.logs),
            }
            if include_logs:
                data["logs"] = list(self.logs)
            return data


class JobManager:
    def __init__(self, max_workers: int = 3) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="studio-job"
        )

    def submit(
        self,
        kind: str,
        title: str,
        fn: Callable[[JobRecord], Dict[str, Any]],
    ) -> JobRecord:
        job = JobRecord(kind=kind, title=title)
        with self._lock:
            self._jobs[job.id] = job
            self._order.insert(0, job.id)
            if len(self._order) > 100:
                old_id = self._order.pop()
                self._jobs.pop(old_id, None)

        def _runner() -> None:
            job.status = "running"
            job.started_ts = time.time()
            job.log("Iniciando ejecución...", progress=1.0, step="Iniciando...")
            try:
                res = fn(job)
                job.check_cancelled()
                job.result = res or {}
                job.status = "completed"
                job.finished_ts = time.time()
                job.speed_text = ""
                job.log(
                    "Proceso completado y verificado correctamente.",
                    level="success",
                    progress=100.0,
                    step="Completado",
                )
            except JobCancelledError as ce:
                job.status = "cancelled"
                job.finished_ts = time.time()
                job.error = str(ce)
                job.speed_text = ""
                job.log(str(ce), level="warning", step="Cancelado")
            except Exception as exc:
                job.status = "failed"
                job.finished_ts = time.time()
                job.error = str(exc)
                job.speed_text = ""
                tb_last = traceback.format_exc().strip().splitlines()[-4:]
                job.log(f"Error: {exc}", level="error", step="Error")
                for ln in tb_last:
                    job.log(ln, level="error")

        self._executor.submit(_runner)
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[JobRecord]:
        job = self.get(job_id)
        if not job:
            return None
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_ts = time.time()
            job.log("Cancelado antes de iniciar.", level="warning", step="Cancelado")
        else:
            job.log("Solicitud de cancelación enviada...", level="warning")
        return job

    def list_jobs(self, limit: int = 30, include_logs: bool = True) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(self._order[:limit])
            records = [self._jobs[jid] for jid in ids if jid in self._jobs]
        return [r.to_dict(include_logs=include_logs) for r in records]

    def active_count(self) -> int:
        with self._lock:
            return sum(
                1 for j in self._jobs.values() if j.status in ("queued", "running")
            )


job_manager = JobManager()
