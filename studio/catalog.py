import io
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download
from tqdm.std import tqdm as std_tqdm

from studio.config import MODELS_DIR, STATE_DIR
from studio.hardware import colab_console_log, query_gpu_info
from studio.jobs import JobRecord

CUSTOM_MODELS_FILE = STATE_DIR / "custom_models.json"
DISK_SAFETY_MARGIN_GB = 1.0

DEFAULT_CATALOG: List[Dict[str, Any]] = [
    # ==================== IMAGEN ====================
    {
        "id": "qwen-image-edit-2511-pack",
        "name": "Qwen-Image-Edit 2511 (4-bit) + LoRA Lightning (4 pasos)",
        "category": "image",
        "is_default": True,
        "badge": "Predeterminado ★ · 4 pasos",
        "description": (
            "Pack principal todo-en-uno: descarga y carga juntos Qwen-Image-Edit-2511 cuantizado en 4-bit "
            "y su LoRA de aceleración Lightning (edición, referencias multi-imagen e inpainting en 4 pasos)."
        ),
        "estimated_gb": 16.6,
        "min_vram_gb": 14.0,
        "recommended_vram_gb": 22.0,
        "engine": "qwen_image_edit",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "ovedrive/Qwen-Image-Edit-2511-4bit",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
            },
            {
                "role": "lora_lightning",
                "repo_id": "lightx2v/Qwen-Image-Edit-2511-Lightning",
                "subdir": "lora_lightning",
                "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "verify_file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            },
        ],
    },
    {
        "id": "qwen-lora-nsfw",
        "name": "LoRA Extra: Qwen-Image-Edit Plus (Sin censura / Realismo)",
        "category": "lora",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "mcnl-nsfw-v1",
        "badge": "Complemento LoRA · Opcional",
        "description": (
            "LoRA opcional compatible con Qwen-Image-Edit-2511. Puedes descargarlo aparte y activarlo "
            "o desactivarlo con un interruptor al editar."
        ),
        "estimated_gb": 0.35,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora",
                "subdir": "lora_nsfw",
                "filename": "qwen-image-edit-plus-nsfw-lora.safetensors",
                "verify_file": "qwen-image-edit-plus-nsfw-lora.safetensors",
            }
        ],
    },
    # ==================== VÍDEO ====================
    {
        "id": "wan22-ti2v-5b",
        "name": "Wan 2.2 TI2V-5B (Imagen / Texto a Vídeo)",
        "category": "video",
        "is_default": True,
        "badge": "Recomendado Vídeo ★",
        "description": (
            "Convierte tus imágenes editadas en clips de vídeo fluidos. Equilibrio ideal entre calidad "
            "y consumo de VRAM para GPUs L4 (22.5 GB) y A100."
        ),
        "estimated_gb": 10.4,
        "min_vram_gb": 14.5,
        "recommended_vram_gb": 22.0,
        "engine": "wan_ti2v",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
            }
        ],
    },
    {
        "id": "wan22-s2v-14b",
        "name": "Wan 2.2 S2V-14B (Vídeo con Diálogo · Audio guía duración)",
        "category": "video",
        "is_default": False,
        "badge": "Sincronía de Voz · Pesado",
        "description": (
            "Genera vídeo animado guiado por una pista de voz/audio e imagen base. "
            "En L4 requiere CPU offload agresivo y liberar antes cualquier otro modelo de VRAM."
        ),
        "estimated_gb": 29.5,
        "min_vram_gb": 22.0,
        "recommended_vram_gb": 40.0,
        "engine": "wan_s2v",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.2-S2V-14B-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
            }
        ],
    },
    {
        "id": "ltx-video-2b",
        "name": "LTX-Video 2B (Imagen a Vídeo ultrarrápido)",
        "category": "video",
        "is_default": False,
        "badge": "Ligero · Compatible T4/L4",
        "description": (
            "Modelo ligero de vídeo de Lightricks en Hugging Face. Ideal para pruebas rápidas o sesiones con GPU T4 (15 GB)."
        ),
        "estimated_gb": 6.8,
        "min_vram_gb": 10.0,
        "recommended_vram_gb": 15.0,
        "engine": "ltx_video",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Lightricks/LTX-Video",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
            }
        ],
    },
    # ==================== VOZ / TTS ====================
    {
        "id": "edge-tts-cloud",
        "name": "Edge-TTS Neural (Álvaro, Elvira y +12 voces · Instantáneo)",
        "category": "tts",
        "is_default": True,
        "builtin_no_download": True,
        "badge": "0 GB · Listo sin descargar",
        "description": (
            "Síntesis de voz neuronal en español e idiomas sin ocupar disco ni VRAM. "
            "Incluye es-ES-AlvaroNeural y es-ES-ElviraNeural."
        ),
        "estimated_gb": 0.0,
        "min_vram_gb": 0.0,
        "recommended_vram_gb": 0.0,
        "engine": "edge_tts",
        "artifacts": [],
    },
    {
        "id": "kokoro-82m-tts",
        "name": "Kokoro-82M Multilingüe (Voz local ligera HF)",
        "category": "tts",
        "is_default": False,
        "badge": "Local HF · 0.35 GB",
        "description": (
            "Modelo abierto ultraligero de Hugging Face para generar voz localmente en GPU o CPU."
        ),
        "estimated_gb": 0.35,
        "min_vram_gb": 0.5,
        "recommended_vram_gb": 2.0,
        "engine": "kokoro_tts",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "hexgrad/Kokoro-82M",
                "subdir": "base",
                "filename": None,
                "verify_file": "config.json",
            }
        ],
    },
    {
        "id": "qwen3-tts-1.7b",
        "name": "Qwen3-TTS 1.7B CustomVoice (Voz expresiva local HF)",
        "category": "tts",
        "is_default": False,
        "badge": "Local HF · Expresivo",
        "description": (
            "Modelo Qwen de texto a voz en Hugging Face para generación local de alta expresividad en GPU."
        ),
        "estimated_gb": 3.6,
        "min_vram_gb": 4.5,
        "recommended_vram_gb": 8.0,
        "engine": "qwen_tts",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "subdir": "base",
                "filename": None,
                "verify_file": "config.json",
            }
        ],
    },
]


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                if not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                pass
    return total


class CatalogManager:
    def __init__(self) -> None:
        self._custom_models: List[Dict[str, Any]] = self._load_custom_models()

    def _load_custom_models(self) -> List[Dict[str, Any]]:
        if CUSTOM_MODELS_FILE.exists():
            try:
                return json.loads(CUSTOM_MODELS_FILE.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_custom_models(self) -> None:
        CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_MODELS_FILE.write_text(
            json.dumps(self._custom_models, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_all_definitions(self) -> List[Dict[str, Any]]:
        return DEFAULT_CATALOG + self._custom_models

    def get_definition(self, model_id: str) -> Optional[Dict[str, Any]]:
        for item in self.get_all_definitions():
            if item["id"] == model_id:
                return item
        return None

    def model_dir(self, model_id: str) -> Path:
        return MODELS_DIR / model_id

    def marker_file(self, model_id: str) -> Path:
        return self.model_dir(model_id) / ".download_complete.json"

    def is_downloaded(self, model_id: str) -> bool:
        defn = self.get_definition(model_id)
        if not defn:
            return False
        if defn.get("builtin_no_download"):
            return True
        m_dir = self.model_dir(model_id)
        if not self.marker_file(model_id).exists():
            return False
        for art in defn.get("artifacts", []):
            sub = m_dir / art["subdir"]
            verify_file = art.get("verify_file") or art.get("filename")
            if verify_file:
                target = sub / verify_file
                if not target.exists() or target.stat().st_size == 0:
                    return False
            elif not sub.exists():
                return False
        return True

    def disk_usage_gb(self, model_id: str) -> float:
        m_dir = self.model_dir(model_id)
        return round(_dir_size_bytes(m_dir) / (1024**3), 2)

    def compatibility_badge(self, defn: Dict[str, Any], gpu_info: Dict[str, Any]) -> Dict[str, str]:
        if defn.get("builtin_no_download") or defn.get("min_vram_gb", 0) == 0:
            return {"status": "ok", "label": "Compatible (CPU o cualquier GPU)"}
        if not gpu_info.get("has_gpu"):
            return {
                "status": "cpu_only",
                "label": f"Requiere GPU (~{defn['min_vram_gb']} GB VRAM) para inferencia rápida",
            }
        vram = float(gpu_info.get("vram_total_gb", 0.0))
        rec = float(defn.get("recommended_vram_gb", defn.get("min_vram_gb", 0.0)))
        min_v = float(defn.get("min_vram_gb", 0.0))
        if vram >= rec:
            return {"status": "ok", "label": f"Óptimo en tu {gpu_info['gpu_tier']} ({vram:.1f} GB)"}
        if vram >= min_v:
            return {
                "status": "offload",
                "label": f"Cabe en tu {gpu_info['gpu_tier']} ({vram:.1f} GB) con CPU Offload",
            }
        return {
            "status": "tight",
            "label": f"Muy justo para {vram:.1f} GB VRAM (pide ~{min_v:.0f} GB, requiere offload secuencial)",
        }

    def list_catalog(self, loaded_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        loaded_set = set(loaded_ids or [])
        gpu_info = query_gpu_info()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        free_gb = round(shutil.disk_usage(MODELS_DIR).free / (1024**3), 2)

        items: List[Dict[str, Any]] = []
        for defn in self.get_all_definitions():
            mid = defn["id"]
            downloaded = self.is_downloaded(mid)
            actual_gb = self.disk_usage_gb(mid) if downloaded and not defn.get("builtin_no_download") else 0.0
            compat = self.compatibility_badge(defn, gpu_info)
            items.append(
                {
                    **defn,
                    "downloaded": downloaded,
                    "actual_disk_gb": actual_gb,
                    "loaded_in_memory": mid in loaded_set,
                    "free_disk_gb": free_gb,
                    "compatibility": compat,
                }
            )
        return items

    def add_custom_hf_model(
        self,
        name: str,
        repo_id: str,
        category: str = "lora",
        filename: Optional[str] = None,
        estimated_gb: float = 1.0,
        adapter_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        repo_id = repo_id.strip()
        if "/" not in repo_id:
            raise ValueError("El repositorio de Hugging Face debe tener formato 'autor/modelo'.")
        slug = repo_id.lower().replace("/", "-").replace("_", "-")
        if filename:
            slug += "-" + Path(filename).stem.lower()[:24]
        if self.get_definition(slug):
            raise ValueError(f"Ya existe este modelo en el catálogo ({slug}).")

        clean_fn = filename.strip() if filename and filename.strip() else None
        engine_map = {
            "image": "qwen_image_edit",
            "lora": "qwen_lora",
            "video": "wan_ti2v",
            "tts": "kokoro_tts",
        }
        entry: Dict[str, Any] = {
            "id": slug,
            "name": name.strip() or repo_id,
            "category": category,
            "is_default": False,
            "is_custom": True,
            "parent_model": "qwen-image-edit-2511-pack" if category == "lora" else None,
            "adapter_name": adapter_name or slug[:20],
            "badge": "Añadido por ti (HF)",
            "description": f"Repositorio Hugging Face: {repo_id}" + (f" ({clean_fn})" if clean_fn else ""),
            "estimated_gb": round(max(0.05, float(estimated_gb)), 2),
            "min_vram_gb": 0.5 if category == "lora" else 8.0,
            "recommended_vram_gb": 1.0 if category == "lora" else 16.0,
            "engine": engine_map.get(category, "custom"),
            "artifacts": [
                {
                    "role": "lora" if category == "lora" else "base",
                    "repo_id": repo_id,
                    "subdir": "custom",
                    "filename": clean_fn,
                    "verify_file": clean_fn,
                }
            ],
        }
        self._custom_models.append(entry)
        self._save_custom_models()
        return entry

    def check_disk_space(
        self,
        model_id: str,
        required_bytes_override: Optional[int] = None,
        free_bytes_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        defn = self.get_definition(model_id)
        if not defn:
            raise ValueError(f"Modelo desconocido: {model_id}")
        if defn.get("builtin_no_download"):
            return {"ok": True, "missing_gb": 0.0, "suggestions": []}

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        free_bytes = (
            free_bytes_override
            if free_bytes_override is not None
            else shutil.disk_usage(MODELS_DIR).free
        )
        already_bytes = _dir_size_bytes(self.model_dir(model_id))
        est_bytes = (
            required_bytes_override
            if required_bytes_override is not None
            else int(float(defn.get("estimated_gb", 1.0)) * (1024**3))
        )
        needed_bytes = max(0, est_bytes - already_bytes)
        safety_bytes = int(DISK_SAFETY_MARGIN_GB * (1024**3))
        total_required = needed_bytes + safety_bytes

        if free_bytes >= total_required:
            return {
                "ok": True,
                "free_gb": round(free_bytes / (1024**3), 2),
                "required_gb": round(total_required / (1024**3), 2),
                "missing_gb": 0.0,
                "suggestions": [],
            }

        missing_gb = round((total_required - free_bytes) / (1024**3), 2)
        suggestions: List[Dict[str, Any]] = []
        for other in self.get_all_definitions():
            oid = other["id"]
            if oid == model_id or other.get("builtin_no_download"):
                continue
            used = _dir_size_bytes(self.model_dir(oid))
            if used > 0:
                suggestions.append(
                    {
                        "id": oid,
                        "name": other["name"],
                        "frees_gb": round(used / (1024**3), 2),
                    }
                )
        suggestions.sort(key=lambda x: x["frees_gb"], reverse=True)

        if suggestions:
            sug_str = ", ".join(f"«{s['name']}» (libera {s['frees_gb']} GB)" for s in suggestions)
            msg = (
                f"Espacio insuficiente en disco: faltan {missing_gb} GB libres "
                f"(quedan {free_bytes / (1024**3):.2f} GB y se necesitan ~{total_required / (1024**3):.2f} GB con margen). "
                f"Sugerencia para borrar: {sug_str}."
            )
        else:
            msg = (
                f"Espacio insuficiente en disco: faltan {missing_gb} GB libres "
                f"(quedan {free_bytes / (1024**3):.2f} GB y se necesitan ~{total_required / (1024**3):.2f} GB). "
                f"No hay otros modelos descargados; puedes limpiar archivos antiguos de salidas o subidas."
            )

        return {
            "ok": False,
            "free_gb": round(free_bytes / (1024**3), 2),
            "required_gb": round(total_required / (1024**3), 2),
            "missing_gb": missing_gb,
            "suggestions": suggestions,
            "message": msg,
        }

    def download_model(
        self,
        model_id: str,
        job: JobRecord,
        progress_range: Tuple[float, float] = (0.0, 100.0),
    ) -> Dict[str, Any]:
        p_min, p_max = float(progress_range[0]), float(progress_range[1])

        def _scale_pct(raw_pct: float) -> float:
            clamped = max(0.0, min(100.0, raw_pct))
            return round(p_min + (clamped / 100.0) * (p_max - p_min), 1)

        defn = self.get_definition(model_id)
        if not defn:
            raise ValueError(f"Modelo desconocido: {model_id}")
        if defn.get("builtin_no_download"):
            job.log("Este motor no requiere descargar archivos.", progress=_scale_pct(100.0), step="Listo")
            return {"model_id": model_id, "downloaded_gb": 0.0}

        precheck = self.check_disk_space(model_id)
        if not precheck["ok"]:
            raise RuntimeError(precheck["message"])

        api = HfApi()
        m_dir = self.model_dir(model_id)
        m_dir.mkdir(parents=True, exist_ok=True)

        # Paso 1: Consultar lista exacta de ficheros y tamaños en Hugging Face
        job.log(
            f"Consultando metadatos y tamaño exacto en Hugging Face para «{defn['name']}»...",
            progress=_scale_pct(2.0),
            step="Consultando Hugging Face...",
        )

        plan_files: List[Dict[str, Any]] = []
        total_bytes = 0
        already_bytes = 0

        for art in defn.get("artifacts", []):
            job.check_cancelled()
            repo_id = art["repo_id"]
            subdir = m_dir / art["subdir"]
            subdir.mkdir(parents=True, exist_ok=True)
            single_fn = art.get("filename")

            info = api.model_info(repo_id=repo_id, files_metadata=True)
            siblings = info.siblings or []

            for sib in siblings:
                rfilename = sib.rfilename
                if single_fn and rfilename != single_fn:
                    continue
                if rfilename.startswith(".git"):
                    continue
                fsize = int(getattr(sib, "size", None) or 0)
                local_target = subdir / rfilename
                exists_already = local_target.exists() and (
                    fsize == 0 or local_target.stat().st_size == fsize
                )
                total_bytes += fsize
                if exists_already:
                    already_bytes += fsize
                plan_files.append(
                    {
                        "repo_id": repo_id,
                        "rfilename": rfilename,
                        "subdir": subdir,
                        "size": fsize,
                        "exists": exists_already,
                    }
                )

        if not plan_files:
            raise RuntimeError("No se encontraron archivos descargables en el repositorio indicado.")

        # Segunda comprobación con el peso exacto reportado por Hugging Face
        exact_check = self.check_disk_space(model_id, required_bytes_override=total_bytes)
        if not exact_check["ok"]:
            raise RuntimeError(exact_check["message"])

        total_gb = total_bytes / (1024**3)
        job.log(
            f"Plan de descarga verificado: {len(plan_files)} archivo(s), peso real {total_gb:.2f} GB.",
            progress=_scale_pct(4.0),
            step=f"0 / {len(plan_files)} archivos (0.00 / {total_gb:.2f} GB)",
        )

        completed_files_bytes = already_bytes
        dl_clock = {
            "start_ts": time.time(),
            "net_bytes": 0,
            "last_ui_ts": 0.0,
        }

        for idx, item in enumerate(plan_files, start=1):
            job.check_cancelled()
            rfn = item["rfilename"]
            short_name = Path(rfn).name
            fsize = int(item["size"] or 0)

            if item["exists"]:
                raw_pct = (completed_files_bytes / total_bytes * 100.0) if total_bytes > 0 else (idx / len(plan_files) * 100.0)
                job.log(
                    f"[{idx}/{len(plan_files)}] Ya en disco: {short_name}",
                    progress=_scale_pct(min(99.0, max(4.0, raw_pct))),
                    step=f"[{idx}/{len(plan_files)}] Verificado {short_name}",
                )
                continue

            file_mb = fsize / (1024**2)
            base_raw_pct = (completed_files_bytes / total_bytes * 100.0) if total_bytes > 0 else ((idx - 1) / len(plan_files) * 100.0)
            job.log(
                f"[{idx}/{len(plan_files)}] Descargando {short_name} ({file_mb:.1f} MB)...",
                progress=_scale_pct(min(99.0, max(4.0, base_raw_pct))),
                step=f"[{idx}/{len(plan_files)}] {short_name} ({completed_files_bytes / (1024**3):.2f} / {total_gb:.2f} GB)",
            )

            file_progress = {
                "reconstructed": 0,
                "transfer": 0,
                "done": False,
            }

            def _emit_live_progress(force: bool = False) -> None:
                now = time.time()
                if not force and (now - dl_clock["last_ui_ts"] < 0.35):
                    return
                dl_clock["last_ui_ts"] = now

                # En descargas hf_xet usamos el avance mayor entre transferencia de red y reconstrucción
                est_file_done = max(file_progress["reconstructed"], file_progress["transfer"])
                if fsize > 0:
                    est_file_done = min(fsize, est_file_done)

                cur_total_done = completed_files_bytes + est_file_done
                raw_pct = (
                    (cur_total_done / total_bytes) * 100.0
                    if total_bytes > 0
                    else ((idx - 0.5) / len(plan_files)) * 100.0
                )
                dt = max(0.05, now - dl_clock["start_ts"])
                mbps = (dl_clock["net_bytes"] / (1024**2)) / dt
                done_gb = cur_total_done / (1024**3)
                rem_bytes = max(0, total_bytes - cur_total_done)
                bps = dl_clock["net_bytes"] / dt
                eta_str = f" · ETA ~{int(rem_bytes / bps)}s" if bps > 64 * 1024 else ""
                spd_str = f"{mbps:.1f} MB/s{eta_str}" if dl_clock["net_bytes"] > 0 else "Conectando con HF..."

                job.heartbeat(
                    progress=_scale_pct(min(99.2, max(4.0, raw_pct))),
                    step=f"[{idx}/{len(plan_files)}] {short_name} ({done_gb:.2f} / {total_gb:.2f} GB)",
                    speed_text=spd_str,
                )

            class _JobTqdm(std_tqdm):  # type: ignore
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    desc = str(kwargs.get("desc") or "")
                    self._is_xet_transfer = desc == "Downloading bytes"
                    kwargs["file"] = io.StringIO()
                    kwargs["disable"] = False
                    kwargs.pop("name", None)
                    super().__init__(*args, **kwargs)

                def update(self, n: Optional[int | float] = 1) -> Optional[bool]:
                    job.check_cancelled()
                    res = super().update(n)
                    inc = int(n or 0)
                    if inc > 0:
                        if self._is_xet_transfer:
                            file_progress["transfer"] += inc
                            dl_clock["net_bytes"] += inc
                        else:
                            prev_rec = file_progress["reconstructed"]
                            file_progress["reconstructed"] += inc
                            # Si no hay barra xet separada (descarga HTTP normal), contabilizar velocidad aquí
                            if file_progress["transfer"] == 0:
                                dl_clock["net_bytes"] += inc
                            elif file_progress["reconstructed"] > file_progress["transfer"]:
                                dl_clock["net_bytes"] += max(0, file_progress["reconstructed"] - max(prev_rec, file_progress["transfer"]))
                        _emit_live_progress(force=False)
                    return res

            # Hilo vigilante cada 1.5s para mantener el pulso vivo incluso durante verificación de hashes
            def _pulse_watcher() -> None:
                while not file_progress["done"] and not job.cancel_requested:
                    time.sleep(1.5)
                    if not file_progress["done"]:
                        _emit_live_progress(force=True)

            watcher = threading.Thread(target=_pulse_watcher, daemon=True)
            watcher.start()
            try:
                downloaded_path = hf_hub_download(
                    repo_id=item["repo_id"],
                    filename=rfn,
                    local_dir=str(item["subdir"]),
                    tqdm_class=_JobTqdm,
                )
            finally:
                file_progress["done"] = True

            p_check = Path(downloaded_path)
            if not p_check.exists() or p_check.stat().st_size == 0:
                raise RuntimeError(f"Verificación fallida: {short_name} no existe o está vacío tras descargar.")

            completed_files_bytes += fsize if fsize > 0 else p_check.stat().st_size
            pct_done = (completed_files_bytes / total_bytes * 100.0) if total_bytes > 0 else (idx / len(plan_files) * 100.0)
            job.log(
                f"[{idx}/{len(plan_files)}] Completado: {short_name} ({completed_files_bytes / (1024**3):.2f} / {total_gb:.2f} GB)",
                progress=_scale_pct(min(99.5, max(4.0, pct_done))),
                step=f"[{idx}/{len(plan_files)}] {short_name} listo",
            )

        # Limpiar carpeta temporal .cache interna de huggingface dentro de local_dir para no duplicar espacio
        for art in defn.get("artifacts", []):
            hf_local_cache = m_dir / art["subdir"] / ".cache"
            if hf_local_cache.exists():
                shutil.rmtree(hf_local_cache, ignore_errors=True)

        actual_gb = self.disk_usage_gb(model_id)
        self.marker_file(model_id).write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "completed_at": time.time(),
                    "actual_disk_gb": actual_gb,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if not self.is_downloaded(model_id):
            raise RuntimeError("Faltan archivos clave tras la descarga; no se marca como válido.")

        job.log(
            f"Descarga en disco verificada: «{defn['name']}» ({actual_gb:.2f} GB).",
            level="success",
            progress=_scale_pct(100.0),
            step="Archivos verificados en disco",
        )
        return {"model_id": model_id, "actual_disk_gb": actual_gb}

    def delete_model(
        self,
        model_id: str,
        unload_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        defn = self.get_definition(model_id)
        if not defn:
            raise ValueError(f"Modelo desconocido: {model_id}")
        if defn.get("builtin_no_download"):
            raise ValueError("Este motor está integrado y no ocupa espacio en disco.")

        if unload_callback:
            unload_callback(model_id)

        m_dir = self.model_dir(model_id)
        before_bytes = _dir_size_bytes(m_dir)
        if m_dir.exists():
            shutil.rmtree(m_dir)
        freed_gb = round(before_bytes / (1024**3), 2)
        free_now_gb = round(shutil.disk_usage(MODELS_DIR).free / (1024**3), 2)
        msg = f"«{defn['name']}» eliminado del disco ({freed_gb} GB liberados). Espacio libre actual: {free_now_gb} GB."
        colab_console_log(msg, icon="🗑️")
        return {
            "ok": True,
            "model_id": model_id,
            "freed_gb": freed_gb,
            "free_disk_gb": free_now_gb,
            "message": msg,
        }


catalog_manager = CatalogManager()
