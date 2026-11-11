"""Bounded, metadata-only Hugging Face discovery with an honest offline index.

No model weights, repository code, README instructions, pip installs or loaders
are executed here. Public metadata only: session tokens/private results are not
sent to the provider and can never leak through a shared persistent cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin, urlparse

import requests
from huggingface_hub.utils import validate_repo_id

from studio.config import STATE_DIR
from studio.model_compatibility import assess_definition, assessment_context
from studio.model_recipes import ENGINE_CONTRACTS

INDEX_FILE = STATE_DIR / "model_discovery.json"
RESEARCH_FILE = Path(__file__).with_name("data") / "upstream_models.json"
CATEGORIES = ("all", "image", "video", "lora", "tts", "audio", "vision", "lipsync")
TASKS = {
    "image": ["text-to-image", "image-to-image"],
    "video": ["image-to-video", "text-to-video", "video-to-video"],
    "lora": ["lora"], "tts": ["text-to-speech"], "audio": ["text-to-audio", "audio-to-audio"],
    "vision": ["image-text-to-text", "object-detection"], "lipsync": [None],
    "all": [None],
}
MAX_RECORDS = 2500
MAX_QUERIES = 80
MAX_QUERY_ITEMS = 240
MAX_NETWORK_QUERIES = 2
CACHE_TTL_SECONDS = 900
SCHEMA_VERSION = 1


def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc).isoformat(timespec="seconds")


def clean_repo_id(value: str) -> str:
    value = str(value or "").strip()
    validate_repo_id(value)
    if value.count("/") != 1:
        raise ValueError("Usa un repositorio exacto con formato autor/modelo, no una URL.")
    return value


def clean_filename(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).strip()
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute() or any(p in (".", "..") for p in value.split("/"))
            or len(value) > 240 or any(ord(c) < 32 for c in value)):
        raise ValueError("El archivo debe ser una ruta relativa segura dentro del repositorio.")
    return value


def _query(query: str, category: str, sort: str) -> tuple[str, str, str]:
    query, category = str(query or "").strip(), str(category or "video")
    if len(query) > 160:
        raise ValueError("La búsqueda admite como máximo 160 caracteres.")
    if category not in CATEGORIES:
        raise ValueError(f"Categoría no válida. Usa: {', '.join(CATEGORIES)}.")
    if sort not in ("downloads", "lastModified"):
        raise ValueError("Orden permitido: downloads o lastModified.")
    return query, category, sort


class MetadataError(RuntimeError):
    def __init__(self, code: str, message: str, retry_after: int = 0) -> None:
        super().__init__(message)
        self.code, self.retry_after = code, retry_after


class _PublicMetadataAuth(requests.auth.AuthBase):
    def __call__(self, request: Any) -> Any:
        # Also prevent implicit ~/.netrc authentication: this cache is PUBLIC.
        request.headers.pop("Authorization", None)
        return request


class HuggingFaceMetadataProvider:
    """The documented Hub API, with real Link pagination and hard response limits."""
    host = "https://huggingface.co"

    @staticmethod
    def _safe_api_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.path != "/api/models" or parsed.fragment or parsed.username:
            raise MetadataError("invalid_pagination", "Hugging Face devolvió una paginación no permitida.")
        return url

    def _get_json(self, url: str, *, params: Any = None, max_bytes: int = 4_000_000, redirects: int = 0,
                  deadline: Optional[float] = None) -> tuple[Any, Dict[str, str]]:
        deadline = time.monotonic() + 12 if deadline is None else deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MetadataError("offline", "Tiempo de consulta de metadatos agotado; se conserva la caché.")
        if redirects > 3:
            raise MetadataError("redirect_blocked", "Demasiadas redirecciones de metadatos.")
        try:
            # Disable redirects for API calls. Only fixed, small config-file URLs
            # may redirect to the same Hub host (never arbitrary external hosts).
            with requests.get(url, params=params, headers={"User-Agent": "Studio-IA-model-discovery/1.0"},
                              timeout=(min(3.05, remaining), min(6.0, remaining)), stream=True, allow_redirects=False, auth=_PublicMetadataAuth()) as response:
                if response.status_code == 429:
                    try:
                        retry = max(30, min(3600, int(response.headers.get("Retry-After", "60"))))
                    except ValueError:
                        retry = 60
                    raise MetadataError("rate_limited", "Límite de consultas de Hugging Face; se conserva la caché.", retry)
                if response.status_code in (401, 403):
                    raise MetadataError("access_required", "Metadatos/archivo no accesibles públicamente; un token no prueba la aceptación de la licencia.")
                if response.status_code == 404:
                    raise MetadataError("not_found", "Repositorio o metadato no encontrado públicamente.")
                if response.status_code in (301, 302, 303, 307, 308):
                    location = urljoin(url, response.headers.get("Location", ""))
                    parsed = urlparse(location)
                    if parsed.scheme == "https" and parsed.netloc == "huggingface.co" and parsed.path.startswith("/api/resolve-cache/"):
                        return self._get_json(location, max_bytes=max_bytes, redirects=redirects + 1, deadline=deadline)
                    raise MetadataError("redirect_blocked", "Redirección no permitida para una consulta de metadatos.")
                response.raise_for_status()
                if int(response.headers.get("Content-Length") or 0) > max_bytes:
                    raise MetadataError("metadata_too_large", "Respuesta de metadatos mayor que el límite permitido.")
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=4096):
                    if time.monotonic() > deadline:
                        raise MetadataError("offline", "Tiempo de lectura de metadatos agotado; se conserva la caché.")
                    size += len(chunk)
                    if size > max_bytes:
                        raise MetadataError("metadata_too_large", "Respuesta de metadatos truncada por seguridad.")
                    chunks.append(chunk)
                return json.loads(b"".join(chunks)), dict(response.headers)
        except MetadataError:
            raise
        except (requests.RequestException, ValueError, TypeError) as exc:
            # Never persist raw request errors/headers/credentials.
            raise MetadataError("offline", "No se pudo consultar Hugging Face; se muestran caché y referencias locales.") from exc

    def search_page(self, query: str, category: str, sort: str, cursors: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        items, next_cursors = [], {}
        deadline = time.monotonic() + 30
        tasks = TASKS[category]
        for task in tasks:
            key = task or "unfiltered"
            if cursors is not None and not cursors.get(key):
                continue
            if cursors is not None:
                url, params = self._safe_api_url(str(cursors[key])), None
            else:
                url = self.host + "/api/models"
                search = query or ("lipsync" if category == "lipsync" else "")
                params = [("search", search), ("sort", sort), ("direction", "-1"), ("limit", "24")]
                if task:
                    params.append(("filter", task))
                # No full/cardData flags together with expand (Hub API contract).
                for field in ("sha", "pipeline_tag", "library_name", "tags", "downloads", "likes", "gated", "lastModified", "cardData", "safetensors", "config"):
                    params.append(("expand", field))
            data, headers = self._get_json(url, params=params, deadline=deadline)
            if not isinstance(data, list):
                raise MetadataError("invalid_response", "El proveedor no devolvió una lista de metadatos.")
            items.extend(data[:100])
            link = headers.get("Link", headers.get("link", ""))
            match = re.search(r'<([^>]+)>;\s*rel="?next"?', link)
            next_cursors[key] = self._safe_api_url(match.group(1)) if match else None
        return {"items": items, "cursors": next_cursors}

    def inspect(self, repo_id: str) -> Dict[str, Any]:
        repo_id = clean_repo_id(repo_id)
        deadline = time.monotonic() + 30
        data, _ = self._get_json(f"{self.host}/api/models/{quote(repo_id, safe='/')}", params={"blobs": "true"}, deadline=deadline)
        if not isinstance(data, dict):
            raise MetadataError("invalid_response", "Metadatos de repositorio no válidos.")
        if data.get("private"):
            raise MetadataError("private_not_indexed", "No se indexan repositorios privados ni credenciales.")
        revision = str(data.get("sha") or "")
        siblings = data.get("siblings") if isinstance(data.get("siblings"), list) else []
        filenames = {f.get("rfilename") for f in siblings if isinstance(f, dict)}
        configs, issues = {}, []
        # Pin all small JSON reads to the same public commit; no README or Python.
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            for filename in ("model_index.json", "adapter_config.json", "config.json", "transformer/config.json"):
                if filename not in filenames:
                    continue
                try:
                    config, _ = self._get_json(f"{self.host}/{repo_id}/resolve/{revision}/{filename}", max_bytes=262144, deadline=deadline)
                    if isinstance(config, dict):
                        configs[filename] = config
                except MetadataError as exc:
                    issues.append({"file": filename, "code": exc.code, "message": str(exc)})
        data["_small_configs"], data["_inspection_issues"] = configs, issues
        return data


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        result = int(value)
        return result if 0 <= result <= 10**18 else None
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_metadata(raw: Dict[str, Any], category: str, *, now: str, inspected: bool = False) -> Dict[str, Any]:
    """Keep only bounded diagnostic fields; model cards/configs are untrusted data."""
    if not isinstance(raw, dict) or raw.get("private"):
        raise ValueError("Solo se indexan metadatos públicos de repositorios.")
    repo_id = clean_repo_id(raw.get("id") or raw.get("modelId") or raw.get("repo_id") or "")

    def text(value: Any, maximum: int = 160) -> Optional[str]:
        return value[:maximum] if isinstance(value, str) and value else None

    tags_raw = raw.get("tags") if isinstance(raw.get("tags"), list) else []
    tags = [v[:160] for v in tags_raw[:100] if isinstance(v, str)]
    card = raw.get("cardData") or raw.get("card_data")
    card = card if isinstance(card, dict) else {}
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    configs_raw = raw.get("_small_configs")
    small_configs = {k: v for k, v in configs_raw.items() if k in ("model_index.json", "adapter_config.json", "config.json", "transformer/config.json") and isinstance(v, dict)} if isinstance(configs_raw, dict) else {}
    pipeline = text(raw.get("pipeline_tag"), 120)
    if any(t.lower() in ("lora", "peft", "adapter") for t in tags) or "adapter_config.json" in small_configs:
        category = "lora"
    else:
        category = next((c for c, tasks in TASKS.items() if c != "all" and pipeline and pipeline in tasks), "unknown" if category == "all" else category)
    license_id = card.get("license")
    if isinstance(license_id, list):
        license_id = ", ".join(v[:80] for v in license_id[:5] if isinstance(v, str))
    if not isinstance(license_id, str):
        license_id = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    base = card.get("base_model") or []
    base = [base] if isinstance(base, str) else base
    base = [v[:194] for v in base[:12] if isinstance(v, str)] if isinstance(base, list) else []
    st = raw.get("safetensors")
    st = st if isinstance(st, dict) else {}
    parameters = st.get("parameters")
    params = {k[:16]: _int_or_none(v) for k, v in list(parameters.items())[:32] if isinstance(k, str)} if isinstance(parameters, dict) else {}
    widths = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}
    weight_bytes = sum(widths.get(k, 0) * (n or 0) for k, n in params.items())
    files = []
    siblings = raw.get("siblings") if isinstance(raw.get("siblings"), list) else []
    for sib in siblings[:1500]:
        if not isinstance(sib, dict):
            continue
        try:
            filename = clean_filename(sib.get("rfilename"))
        except ValueError:
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
        if filename:
            size = sib.get("size") if sib.get("size") is not None else lfs.get("size")
            files.append({"name": filename, "size": _int_or_none(size)})
    known_sizes = [f["size"] for f in files if f["size"] is not None]
    manifest = small_configs.get("model_index.json", {})
    manifest_class = manifest.get("_class_name")
    remote_code = bool(config.get("auto_map") or any(v.get("auto_map") for v in small_configs.values()) or manifest.get("custom_pipeline"))
    for component, spec in manifest.items():
        if component.startswith("_"):
            continue
        # Pipeline scalar options (force_zeros_for_empty_prompt,
        # requires_safety_checker, add_watermarker...) are not Python modules.
        if isinstance(spec, (list, tuple)) and len(spec) == 2 and isinstance(spec[0], str):
            if spec[0] not in ("diffusers", "transformers", "stable_diffusion", "stable_diffusion_xl"):
                remote_code = True
    if manifest_class is not None and not isinstance(manifest_class, str):
        remote_code = True
        manifest_class = None
    quant = []
    for filename, cfg in small_configs.items():
        if isinstance(cfg.get("quantization_config"), dict):
            qc = cfg["quantization_config"]
            quant.append({"component": filename, "method": text(qc.get("quant_method"), 80) or "unknown",
                          "load_in_4bit": qc.get("load_in_4bit") is True, "load_in_8bit": qc.get("load_in_8bit") is True})
    issues = raw.get("_inspection_issues")
    issues = [{"file": text(i.get("file")), "code": text(i.get("code"), 80), "message": text(i.get("message"), 300)}
              for i in issues[:4] if isinstance(i, dict)] if isinstance(issues, list) else []
    return {
        "repo_id": repo_id, "name": repo_id.split("/")[-1], "author": repo_id.split("/")[0],
        "category": category, "pipeline_tag": pipeline, "library_name": text(raw.get("library_name"), 80),
        "tags": tags, "license": text(license_id, 300), "base_models": base,
        "gated": raw.get("gated") if isinstance(raw.get("gated"), (bool, str)) else None,
        "downloads": _int_or_none(raw.get("downloads")), "likes": _int_or_none(raw.get("likes")),
        "revision": text(raw.get("sha"), 80), "last_modified": text(raw.get("lastModified") or raw.get("last_modified"), 80),
        "parameter_count": _int_or_none(st.get("total")), "parameter_dtypes": params,
        "tensor_storage_gb": round(weight_bytes / 1024**3, 3) if weight_bytes else None,
        "repository_gb": round(sum(known_sizes) / 1024**3, 3) if known_sizes else None,
        "file_sizes_complete": bool(files and len(files) == len(known_sizes) == len(siblings)),
        "files": files[:256], "inventory_file_count": len(siblings),
        "inventory_preview_truncated": len(files) > 256 or len(siblings) > 1500,
        "pipeline_class": text(manifest_class), "has_model_index": bool(manifest) or any(f["name"] == "model_index.json" for f in files),
        "requires_remote_code": remote_code, "quantization": quant,
        "adapter_base": text(small_configs.get("adapter_config.json", {}).get("base_model_name_or_path"), 194),
        "metadata_level": "inspected" if inspected else "summary", "metadata_checked_at": now,
        "inspection_checked_at": now if inspected else None,
        "source": "huggingface_api", "source_urls": [f"https://huggingface.co/{repo_id}"],
        "inspection_issues": issues,
    }

class ModelDiscovery:
    def __init__(self, *, path: Path = INDEX_FILE, provider: Any = None, catalog: Any = None,
                 clock: Any = time.time) -> None:
        self.path, self.provider, self._catalog, self.clock = Path(path), provider or HuggingFaceMetadataProvider(), catalog, clock
        self._lock = threading.RLock()
        self._inflight: set[str] = set()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._refresh_running = False
        self._next_refresh_ts: Optional[float] = None
        self._cache_warning: Optional[str] = None
        self._state: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "records": {}, "queries": {},
            "settings": {"auto_refresh": os.environ.get("STUDIO_MODEL_AUTO_REFRESH", "1") != "0", "interval_minutes": 360},
            "last_attempt_at": None, "last_success_at": None, "last_error": None, "failures": 0, "cooldown_until": 0}
        self._load()
        # An explicit launch opt-out must win over a previously persisted default.
        # The UI may still change the setting deliberately during this process.
        if "STUDIO_MODEL_AUTO_REFRESH" in os.environ:
            self._state["settings"]["auto_refresh"] = os.environ["STUDIO_MODEL_AUTO_REFRESH"] != "0"
        try:
            self.research = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.research = {"families": []}

    @property
    def catalog(self) -> Any:
        if self._catalog is None:
            from studio.catalog import catalog_manager
            return catalog_manager
        return self._catalog

    def _load(self) -> None:
        if not self.path.exists():
            return
        defaults = copy.deepcopy(self._state)
        try:
            if self.path.stat().st_size > 12_000_000:
                raise ValueError("oversized")
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("invalid index")
            if data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("records"), dict) or not isinstance(data.get("queries"), dict):
                raise ValueError("invalid schema")
            self._state.update({k: data[k] for k in self._state if k in data and k != "settings"})
            settings = data.get("settings") or {}
            if not isinstance(settings, dict) or not isinstance(settings.get("auto_refresh", True), bool):
                raise ValueError("invalid settings")
            self._state["settings"].update(settings)
            self._state["settings"]["interval_minutes"] = max(15, min(1440, int(self._state["settings"]["interval_minutes"])))
            if not isinstance(self._state["cooldown_until"], (int, float)) or not math.isfinite(self._state["cooldown_until"]):
                raise ValueError("invalid cooldown")
            if not isinstance(self._state["failures"], int) or not 0 <= self._state["failures"] <= 8:
                raise ValueError("invalid failures")
            self._state["cooldown_until"] = min(self._state["cooldown_until"], self.clock() + 3600)
            for stamp in ("last_attempt_at", "last_success_at"):
                if self._state[stamp] is not None and not isinstance(self._state[stamp], str):
                    raise ValueError("invalid timestamp")
            if self._state["last_error"] is not None and not isinstance(self._state["last_error"], dict):
                raise ValueError("invalid error")
            for entry in self._state["queries"].values():
                if (not isinstance(entry, dict) or not isinstance(entry.get("ids", []), list)
                        or not isinstance(entry.get("cursors", {}), dict)
                        or not isinstance(entry.get("attempt_ts", 0), (float, int))
                        or not isinstance(entry.get("success_ts", 0), (float, int))):
                    raise ValueError("invalid query")
                _query(entry.get("query", ""), entry.get("category", "video"), entry.get("sort", "downloads"))
                if entry.get("error") is not None and not isinstance(entry["error"], dict):
                    raise ValueError("invalid query error")
                for stamp in ("attempt_ts", "success_ts"):
                    value = entry.get(stamp, 0)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid query timestamp")
                    if stamp in entry:
                        entry[stamp] = min(value, self.clock())
                if any(not isinstance(rid, str) for rid in entry.get("ids", [])):
                    raise ValueError("invalid query ids")
                for cursor in entry.get("cursors", {}).values():
                    if cursor is not None:
                        HuggingFaceMetadataProvider._safe_api_url(cursor)
            self._prune()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, MetadataError):
            self._cache_warning = "Índice persistente inválido/no legible; se usa el catálogo local sin borrar pesos."
            self._state = defaults

    def _prune(self) -> None:
        records, queries = self._state["records"], self._state["queries"]
        for key in list(records):
            if not isinstance(records[key], dict) or records[key].get("source") != "huggingface_api":
                del records[key]
                continue
            try:
                repo_id = clean_repo_id(records[key].get("repo_id", ""))
                record = records[key]
                if key != repo_id.lower():
                    raise ValueError("invalid key")
                if record.get("category") not in (*CATEGORIES, "unknown") or record.get("metadata_level") not in ("summary", "inspected"):
                    raise ValueError("invalid record state")
                for field in ("metadata_checked_at", "inspection_checked_at", "last_modified", "pipeline_class", "revision", "license"):
                    if record.get(field) is not None and not isinstance(record[field], str):
                        raise ValueError("invalid text field")
                for field in ("downloads", "likes", "parameter_count", "inventory_file_count"):
                    if record.get(field) is not None and _int_or_none(record[field]) != record[field]:
                        raise ValueError("invalid numeric field")
                for field in ("files", "tags", "source_urls", "base_models", "quantization", "inspection_issues"):
                    if field in record and not isinstance(record[field], list):
                        raise ValueError("invalid list field")
                for field in ("tags", "source_urls", "base_models", "families"):
                    if field in record and (not isinstance(record[field], list) or any(not isinstance(v, str) for v in record[field])):
                        raise ValueError("invalid text list")
                for field in ("tensor_storage_gb", "repository_gb"):
                    value = record.get(field)
                    if value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
                        raise ValueError("invalid storage number")
                for file in record.get("files", []):
                    if not isinstance(file, dict) or not clean_filename(file.get("name")):
                        raise ValueError("invalid file")
                stamp = record.get("inspection_ts", 0)
                if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                    raise ValueError("invalid inspection time")
                if "inspection_ts" in record:
                    record["inspection_ts"] = min(stamp, self.clock())
            except (ValueError, TypeError):
                del records[key]
        newest = sorted(records, key=lambda k: records[k].get("metadata_checked_at") or "", reverse=True)[:MAX_RECORDS]
        self._state["records"] = {k: records[k] for k in newest}
        valid_queries = [(k, v) for k, v in queries.items() if isinstance(v, dict)]
        self._state["queries"] = dict(sorted(valid_queries, key=lambda row: row[1].get("attempt_ts", 0), reverse=True)[:MAX_QUERIES])
        for query in self._state["queries"].values():
            query["ids"] = [rid for rid in query.get("ids", []) if isinstance(rid, str) and rid in self._state["records"]][:MAX_QUERY_ITEMS]
            order = query.get("display_ids", [])
            query["display_ids"] = [rid for rid in order[:MAX_QUERY_ITEMS + 600] if isinstance(rid, str)] if isinstance(order, list) else []

    def _save(self) -> None:
        # Caller owns the RLock. Atomic replacement also tolerates interrupted Colab sessions.
        self._prune()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            payload = json.dumps(self._state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            # Bound disk footprint as well as counts (large file manifests/tags).
            while len(payload.encode("utf-8")) > 8_000_000 and self._state["records"]:
                self._state["records"].pop(next(reversed(self._state["records"])))
                self._prune()
                payload = json.dumps(self._state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self.path)
        except (OSError, ValueError, TypeError):
            self._cache_warning = "No se pudo persistir el índice; la caché sigue en memoria. No se han tocado pesos."

    def configure(self, *, auto_refresh: bool, interval_minutes: int = 360) -> Dict[str, Any]:
        if not isinstance(auto_refresh, bool) or type(interval_minutes) is not int or not 15 <= interval_minutes <= 1440:
            raise ValueError("Actualización automática: booleano; intervalo: 15–1440 minutos.")
        with self._lock:
            self._state["settings"] = {"auto_refresh": auto_refresh, "interval_minutes": interval_minutes}
            self._save()
        self._next_refresh_ts = self.clock() if auto_refresh else None
        self._wake.set()
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            state = self._state
            families = self.research.get("families", [])
            return {"ok": True, "settings": dict(state["settings"]), "indexed_repositories": len(state["records"]),
                "cached_queries": len(state["queries"]), "research_families": len(families),
                "research_repositories": len({r for family in families for r in family.get("repo_ids", [])}),
                "research_date": self.research.get("researched_at"), "research_revision": self.research.get("upstream_revision"),
                "last_attempt_at": state["last_attempt_at"], "last_success_at": state["last_success_at"],
                "last_error": copy.deepcopy(state["last_error"]), "refresh_running": self._refresh_running,
                "scheduler_running": bool(self._thread and self._thread.is_alive()),
                "next_refresh_at": _iso(self._next_refresh_ts) if state["settings"]["auto_refresh"] and self._next_refresh_ts is not None and self._thread and self._thread.is_alive() else None,
                "retry_after_seconds": max(0, math.ceil(state["cooldown_until"] - self.clock())),
                "cache_warning": self._cache_warning, "ttl_seconds": CACHE_TTL_SECONDS,
                "limits": {"repositories": MAX_RECORDS, "queries": MAX_QUERIES, "repositories_per_query": MAX_QUERY_ITEMS, "persistent_bytes": 8_000_000, "concurrent_network_queries": MAX_NETWORK_QUERIES},
                "policy": "Metadatos públicos, sin pesos ni código remoto. Catálogo revisado ≠ candidato ≠ backend no soportado. No es un censo exhaustivo ni validación GPU."}

    def _seed_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for family in self.research.get("families", []):
            for repo in family.get("repo_ids", []):
                try:
                    clean_repo_id(repo)
                except ValueError:
                    continue
                key = repo.lower()
                if key not in records:
                    records[key] = {"repo_id": repo, "name": repo.split("/")[-1], "author": repo.split("/")[0],
                        "category": family["category"], "source": "upstream_documentation", "metadata_level": "documentation",
                        "source_urls": [], "families": [], "researched_at": self.research.get("researched_at"),
                        "metadata_checked_at": None, "downloads": None, "likes": None, "license": None, "gated": None}
                records[key]["source_urls"].append(family["source_url"])
                records[key]["families"].append(family["family"])
        for definition in self.catalog.get_all_definitions():
            if definition.get("validation_state") == "candidate":
                continue
            for i, artifact in enumerate(definition.get("artifacts", [])):
                repo = artifact.get("repo_id")
                if not repo:
                    continue
                key = repo.lower()
                if key not in records:
                    records[key] = {"repo_id": repo, "name": repo.split("/")[-1], "author": repo.split("/")[0],
                        "source": "studio_catalog", "metadata_level": "catalog", "metadata_checked_at": None,
                        "source_urls": [], "downloads": None, "likes": None, "license": definition.get("license"), "gated": None}
                if i == 0:
                    records[key]["category"] = definition["category"]
                else:
                    records[key].setdefault("category", "lora" if "lora" in str(artifact.get("role")) else definition["category"])
                # A repository may contain many named adapters (e.g. Kijai/WanVideo_comfy).
                # Searching only its repository name makes a known file-backed recipe invisible.
                aliases = records[key].setdefault("catalog_search_aliases", [])
                for value in (definition.get("id"), definition.get("name"), artifact.get("filename")):
                    if value and value not in aliases:
                        aliases.append(value)
        return records

    def _known_definitions(self, repo_id: str) -> tuple[List[Dict[str, Any]], List[str]]:
        primary, components = [], []
        for definition in self.catalog.get_all_definitions():
            if definition.get("validation_state") == "candidate" or (definition.get("is_custom") and definition.get("validation_state") != "reviewed_recipe"):
                continue
            artifacts = definition.get("artifacts", [])
            if artifacts and str(artifacts[0].get("repo_id", "")).lower() == repo_id.lower():
                primary.append(definition)
            elif any(str(a.get("repo_id", "")).lower() == repo_id.lower() for a in artifacts):
                components.append(definition["id"])
        return primary, components

    def decorate(self, record: Dict[str, Any], context: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        item = copy.deepcopy(record)
        definitions, component_of = self._known_definitions(item["repo_id"])
        definitions = [d for d in definitions if d["category"] == item.get("category")]
        # If the query names a particular file/recipe within a shared repository,
        # do not report an unrelated adapter as the matching recipe.
        if query:
            compact = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
            def recipe_matches(definition: Dict[str, Any]) -> bool:
                parts = [item.get("repo_id", ""), *item.get("families", []), definition.get("id", ""),
                         definition.get("name", ""), *(a.get("filename", "") or "" for a in definition.get("artifacts", []))]
                text = compact(" ".join(parts))
                return all(compact(term) in text for term in query.split() if compact(term))
            narrowed = [d for d in definitions if recipe_matches(d)]
            if narrowed:
                definitions = narrowed
        recipes = [{"model_id": d["id"], "engine": d.get("engine"), "video_modes": d.get("video_modes"),
                    "capabilities": d.get("capabilities"), "assessment": assess_definition(d, context, catalog=self.catalog)} for d in definitions]
        recipes.sort(key=lambda row: (not row["assessment"]["compatible"], not self.catalog.is_downloaded(row["model_id"]), row["assessment"]["tier_rank"]))
        tags = {str(t).lower() for t in item.get("tags", [])}
        files = item.get("files", [])
        filenames = [f["name"].lower() for f in files]
        unsupported = []
        if item.get("requires_remote_code"):
            unsupported.append("El repositorio declara código remoto/custom; Studio no lo ejecuta.")
        if ("gguf" in tags or any(f.endswith(".gguf") for f in filenames)) and not item.get("has_model_index"):
            unsupported.append("GGUF/ComfyUI sin pipeline completo: requiere una receta/backend de componentes diferente.")
        if tags.intersection({"mlx", "onnx", "coreml"}) and "diffusers" not in tags and not recipes:
            unsupported.append("Formato/backend externo sin cargador revisado en Studio.")
        manifest_class = item.get("pipeline_class")
        allowed_manifest_aliases = {
            "sdxl_turbo": {"StableDiffusionXLPipeline"},
            "ltx_video": {"LTXPipeline"}, "ltx_video_distilled": {"LTXPipeline"},
            "ltx2_av": {"LTX2Pipeline"},
            "ltx2_distilled_av": {"LTX2Pipeline"},
        }
        if recipes and item.get("quantization") and not any(ENGINE_CONTRACTS.get(r["engine"], {}).get("quantization") for r in recipes):
            unsupported.append("El repositorio declara una cuantización no incluida en la receta de precisión completa; requiere revisión.")
        if recipes and manifest_class:
            permitted = set()
            for recipe in recipes:
                permitted.update(ENGINE_CONTRACTS.get(recipe["engine"], {}).get("classes", []))
                permitted.update(allowed_manifest_aliases.get(recipe["engine"], set()))
            if manifest_class not in permitted:
                unsupported.append("La clase actual del manifiesto no coincide con la receta revisada; requiere nueva revisión.")
        matching_engines = [engine for engine, contract in ENGINE_CONTRACTS.items() if item.get("pipeline_class") in contract.get("classes", [])]
        support = "unsupported" if unsupported else "reviewed_recipe" if recipes else "candidate"
        if recipes and not unsupported:
            comp = recipes[0]["assessment"]
            definition = self.catalog.get_definition(recipes[0]["model_id"]) or {}
            item.update(estimated_gb=definition.get("estimated_gb"), min_vram_gb=comp["resources"]["vram"]["required_gb"],
                        recommended_vram_gb=definition.get("recommended_vram_gb"))
        else:
            label = "⛔ Backend/archivo no soportado" if unsupported else "🧪 Candidato · falta validar receta"
            comp = {"compatible": False, "runnable_now": False, "support": support, "status": support,
                    "tier": "incompatible" if unsupported else "candidate", "tier_rank": 3 if unsupported else 2,
                    "sticker": label, "label": label, "is_simulated": bool(context["target"].get("is_simulated")),
                    "resources": {axis: {"status": "unknown", "required_gb": None} for axis in ("vram", "ram", "disk")},
                    "reasons": unsupported or ["Metadatos/documentación no bastan para validar API, componentes, precisión y picos de memoria."],
                    "warnings": ["No inferir VRAM a partir del nombre, número de parámetros ni tamaño de un LoRA."], "confidence": "unknown"}
            item.update(estimated_gb=None, min_vram_gb=None, recommended_vram_gb=None)
        gated = item.get("gated")
        access = "requires_acceptance" if gated and gated != "false" else "public_metadata" if gated is False else "unknown"
        if access == "requires_acceptance":
            comp = copy.deepcopy(comp)
            comp["runnable_now"] = False
            comp.setdefault("warnings", []).append("Aceptar licencia y comprobar acceso a los pesos; tener un token no acredita acceso.")
        if unsupported:
            recipes = []
        item.update(support={"status": support, "potential_engines": matching_engines,
                            "component_of": component_of, "notes": unsupported},
                    compatibility=comp, recipes=recipes, catalog_model_ids=[r["model_id"] for r in recipes],
                    already_in_catalog=bool(recipes), access={"status": access, "weights_access_verified": False,
                        "token_present": bool(self.catalog.get_hf_token())}, gated_locked=access == "requires_acceptance",
                    import_policy="metadata_only_candidate" if not recipes else "use_existing_recipe",
                    can_auto_load=False,
                    weight_size_note="Inventario/tensores del repositorio; puede incluir variantes duplicadas o excluir encoders/VAE externos. No es un presupuesto de VRAM.")
        return item

    @staticmethod
    def _matches(record: Dict[str, Any], query: str, category: str) -> bool:
        if category != "all" and record.get("category") != category:
            return False
        compact = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
        text = compact(record.get("repo_id", "") + " " + " ".join(record.get("families", [])) + " "
                       + " ".join(record.get("catalog_search_aliases", [])))
        return all(compact(term) in text for term in query.split() if compact(term))

    def _remember_error(self, exc: MetadataError) -> None:
        self._state["failures"] = min(8, self._state.get("failures", 0) + 1)
        if exc.code in {"offline", "rate_limited", "invalid_response", "metadata_too_large"}:
            delay = exc.retry_after or min(900, 15 * 2 ** self._state["failures"])
            self._state["cooldown_until"] = self.clock() + delay
        self._state["last_error"] = {"code": exc.code, "message": str(exc), "at": _iso(self.clock())}

    def search(self, query: str = "", category: str = "video", gpu_profile: str = "auto",
               only_compatible: bool = False, limit: int = 12, offset: int = 0, refresh: bool = False,
               sort: str = "downloads", context: Optional[Dict[str, Any]] = None,
               detected_gpu: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query, category, sort = _query(query, category, sort)
        context = context or assessment_context(gpu_profile=gpu_profile, detected_gpu=detected_gpu)
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 50 or not 0 <= offset <= 5000:
            raise ValueError("Límite: 1–50; desplazamiento: 0–5000.")
        limit, offset = int(limit), int(offset)
        key = hashlib.sha256(json.dumps([query.lower(), category, sort]).encode()).hexdigest()[:24]
        now = self.clock()
        mode, network_attempted = "cache", False
        with self._lock:
            entry = copy.deepcopy(self._state["queries"].get(key, {}))
            stale = not entry.get("success_ts") or now - entry["success_ts"] >= CACHE_TTL_SECONDS
            visible_count = len(entry.get("display_ids") or entry.get("ids", []))
            more_needed = bool(offset + limit > visible_count and any((entry.get("cursors") or {}).values()) and len(entry.get("ids", [])) < MAX_QUERY_ITEMS and not only_compatible)
            busy = key in self._inflight or len(self._inflight) >= MAX_NETWORK_QUERIES
            should_fetch = (refresh or stale or more_needed) and not busy and now >= self._state["cooldown_until"]
            if should_fetch:
                self._inflight.add(key)
                self._state["last_attempt_at"] = _iso(now)
        if should_fetch:
            network_attempted = True
            try:
                next_page = entry.get("cursors") if more_needed and not refresh and not stale else None
                page = self.provider.search_page(query, category, sort, next_page)
                normalized = []
                if not isinstance(page, dict) or not isinstance(page.get("items"), list) or not isinstance(page.get("cursors", {}), dict):
                    raise MetadataError("invalid_response", "Respuesta de búsqueda no válida; se conserva el índice anterior.")
                for raw in page.get("items", [])[:150]:
                    if not isinstance(raw, dict) or raw.get("private"):
                        continue
                    try:
                        normalized.append(normalize_metadata(raw, category, now=_iso(now)))
                    except (ValueError, TypeError):
                        continue
                if page["items"] and not normalized:
                    raise MetadataError("invalid_response", "Ningún registro válido en la respuesta pública; se conserva la caché anterior.")
                with self._lock:
                    ids = list(entry.get("ids", [])) if next_page else []
                    for record in normalized:
                        rid = record["repo_id"].lower()
                        previous = self._state["records"].get(rid, {})
                        if previous.get("metadata_level") == "inspected" and previous.get("revision") == record.get("revision"):
                            record = {**previous, **{k: record[k] for k in ("downloads", "likes", "last_modified", "metadata_checked_at")}}
                        self._state["records"][rid] = record
                        if rid not in ids:
                            ids.append(rid)
                    entry = {"query": query, "category": category, "sort": sort, "ids": ids[:MAX_QUERY_ITEMS],
                             "cursors": page.get("cursors", {}), "success_ts": now, "attempt_ts": now,
                             "display_ids": entry.get("display_ids", []) if next_page else [], "error": None}
                    self._state["queries"][key] = entry
                    self._state.update(last_success_at=_iso(now), last_error=None, failures=0, cooldown_until=0)
                    self._save()
                mode = "network"
            except MetadataError as exc:
                with self._lock:
                    self._remember_error(exc)
                    entry.update(query=query, category=category, sort=sort, attempt_ts=now,
                                 error={"code": exc.code, "message": str(exc), "at": _iso(now)})
                    self._state["queries"][key] = entry
                    self._save()
                mode = "offline"
            finally:
                with self._lock:
                    self._inflight.discard(key)
        elif stale:
            mode = "refreshing" if busy else "offline"
        context = context or assessment_context(gpu_profile=gpu_profile, detected_gpu=detected_gpu)
        with self._lock:
            indexed = copy.deepcopy(self._state["records"])
            provider_error = copy.deepcopy(entry.get("error") or (self._state["last_error"] if mode == "offline" else None))
        records = self._seed_records()
        for rid, record in indexed.items():
            seed = records.get(rid, {})
            records[rid] = {**seed, **record, "source_urls": list(dict.fromkeys(seed.get("source_urls", []) + record.get("source_urls", [])))}
        remote_ids = set(entry.get("ids", []))
        matched = {rid: r for rid, r in records.items() if
                   (rid in remote_ids or self._matches(r, query, category))}
        # Preserve the order of already served pages within this cached query.
        # New provider pages append rather than shifting rows the user has seen.
        # Refresh/TTL starts a new snapshot. Unknown popularity stays unknown.
        existing_order = [rid for rid in entry.get("display_ids", []) if rid in matched]
        existing_set = set(existing_order)
        additions = [rid for rid in matched if rid not in existing_set]
        additions.sort()
        if sort == "lastModified":
            additions.sort(key=lambda rid: matched[rid].get("last_modified") or "", reverse=True)
        else:
            additions.sort(key=lambda rid: matched[rid].get("downloads") or 0, reverse=True)
        order = existing_order + additions
        if entry:
            with self._lock:
                if key not in self._inflight and key in self._state["queries"] and self._state["queries"][key].get("attempt_ts") == entry.get("attempt_ts"):
                    if self._state["queries"][key].get("display_ids") != order:
                        self._state["queries"][key]["display_ids"] = order
                        self._save()
        items = [self.decorate(matched[rid], context, query=query) for rid in order]
        if only_compatible:
            items = [i for i in items if i["compatibility"]["compatible"]]
        page_items = items[offset:offset + limit]
        success_ts = entry.get("success_ts")
        stale = not success_ts or now - success_ts >= CACHE_TTL_SECONDS
        return {"ok": True, "online": mode == "network", "source_mode": mode, "network_attempted": network_attempted,
            "query": query, "category": category, "sort": sort, "gpu_target": context["target"]["gpu_name"],
            "active_gpu_profile": context["target"], "items": page_items, "total_local_matches": len(items),
            "offset": offset, "next_offset": offset + len(page_items) if offset + limit < len(items) or (not only_compatible and any((entry.get("cursors") or {}).values()) and len(entry.get("ids", [])) < MAX_QUERY_ITEMS) else None,
            "remote_has_more": bool(any((entry.get("cursors") or {}).values())),
            "query_limit_reached": len(entry.get("ids", [])) >= MAX_QUERY_ITEMS,
            "metadata_checked_at": _iso(success_ts) if success_ts else None, "stale": stale,
            "provider_error": provider_error, "discovery_status": self.status(),
            "warning": (provider_error or {}).get("message") if mode in ("offline", "refreshing") else None,
            "pagination_note": "Instantánea acotada por consulta: las páginas remotas se añaden sin desplazar las ya mostradas. Refrescar/TTL reinicia el orden; concretar la búsqueda al alcanzar el límite.",
            "coverage": "Búsqueda paginada y acotada de metadatos públicos más recetas/referencias locales; no enumera todos los modelos existentes ni valida inferencia."}

    def inspect(self, repo_id: str, *, gpu_profile: str = "auto", refresh: bool = False,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        repo_id = clean_repo_id(repo_id)
        context = context or assessment_context(gpu_profile=gpu_profile)
        rid, now = repo_id.lower(), self.clock()
        with self._lock:
            record = copy.deepcopy(self._state["records"].get(rid))
            fresh = bool(record and record.get("metadata_level") == "inspected" and record.get("inspection_ts") and now - record["inspection_ts"] < CACHE_TTL_SECONDS)
            key = "inspect:" + rid
            fetch = (refresh or not fresh) and now >= self._state["cooldown_until"] and key not in self._inflight and len(self._inflight) < MAX_NETWORK_QUERIES
            if fetch:
                self._inflight.add(key)
                self._state["last_attempt_at"] = _iso(now)
        mode = "cache" if fresh else "refreshing" if key in self._inflight and not fetch else "offline"
        inspection_error = None
        if fetch:
            try:
                raw = self.provider.inspect(repo_id)
                category = (record or {}).get("category", self._seed_records().get(rid, {}).get("category", "unknown"))
                record = normalize_metadata(raw, category, now=_iso(now), inspected=True)
                record["inspection_ts"] = now
                with self._lock:
                    self._state["records"][rid] = record
                    self._state.update(last_success_at=_iso(now), last_error=None, failures=0, cooldown_until=0)
                    self._save()
                mode = "network"
            except MetadataError as exc:
                inspection_error = {"code": exc.code, "message": str(exc), "at": _iso(now)}
                mode = "offline"
                with self._lock:
                    self._remember_error(exc)
                    self._save()
            finally:
                with self._lock:
                    self._inflight.discard(key)
        record = record or self._seed_records().get(rid) or {"repo_id": repo_id, "category": "unknown", "source": "user_candidate", "metadata_level": "unavailable", "metadata_checked_at": None}
        context = context or assessment_context(gpu_profile=gpu_profile)
        return {"ok": True, "source_mode": mode, "online": mode == "network", "stale": not fresh and mode != "network",
                "provider_error": inspection_error, "item": self.decorate(record, context),
                "discovery_status": self.status(), "metadata_only": True,
                "next_step": "Usar únicamente catalog_model_ids con su receta exacta. Sin receta: guardar como candidato; no descargar ni ejecutar código remoto."}

    def refresh(self, *, category: str = "all", query: str = "", sort: str = "downloads", job: Any = None) -> Dict[str, Any]:
        query, category, sort = _query(query, category, sort)
        with self._lock:
            if self._refresh_running:
                return {"ok": True, "already_running": True, "status": self.status()}
            self._refresh_running = True
            recent = sorted(self._state["queries"].values(), key=lambda q: q.get("attempt_ts", 0), reverse=True)[:2]
        targets = [(query, category, sort)] if category != "all" or query else [("", c, "lastModified") for c in ("image", "video", "lora", "tts", "audio", "vision")]
        if category == "all" and not query:
            targets.extend((q["query"], q["category"], q.get("sort", "downloads")) for q in recent if q.get("query"))
        targets = list(dict.fromkeys(targets))[:8]
        outcomes = []
        try:
            # One shared hardware snapshot, not one GPU probe per search/model.
            context = assessment_context()
            for index, (text, cat, order) in enumerate(targets):
                if job:
                    job.check_cancelled()
                    job.log(f"Actualizando solo metadatos: {cat} / {text or 'recientes'}", progress=index * 100 / len(targets), step="Índice de modelos (sin pesos)")
                result = self.search(text, cat, sort=order, refresh=True, limit=1, context=context)
                outcomes.append({"query": text, "category": cat, "source_mode": result["source_mode"]})
                if result["source_mode"] == "offline":
                    break  # Shared outage/rate limit: don't hammer each category.
            status = self.status()
            status["refresh_running"] = False
            return {"ok": True, "updated": any(o["source_mode"] == "network" for o in outcomes), "outcomes": outcomes, "status": status, "metadata_only": True}
        finally:
            with self._lock:
                self._refresh_running = False

    def queue_refresh(self, *, category: str = "all", query: str = "", sort: str = "downloads") -> Dict[str, Any]:
        _query(query, category, sort)
        from studio.jobs import job_manager
        job = job_manager.submit(kind="model_discovery", title="Actualizar índice de modelos (solo metadatos)",
            fn=lambda j: self.refresh(category=category, query=query, sort=sort, job=j), dedup_key="model-discovery-refresh")
        return {"ok": True, "job": job.to_dict(), "metadata_only": True}

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._wake.clear()
            self._next_refresh_ts = self.clock() + 60
            self._thread = threading.Thread(target=self._scheduler, name="studio-model-metadata-updater", daemon=True)
            self._thread.start()

    def _scheduler(self) -> None:
        # Startup stays fast/offline-safe; opt-out via UI or STUDIO_MODEL_AUTO_REFRESH=0.
        delay = 60.0
        while not self._stop.is_set():
            self._wake.wait(delay)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                enabled = self._state["settings"]["auto_refresh"]
                interval = self._state["settings"]["interval_minutes"] * 60
            if enabled:
                self.queue_refresh()
            delay = float(interval)
            self._next_refresh_ts = self.clock() + delay if enabled else None

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread:
            thread.join(timeout=1.0)


model_discovery = ModelDiscovery()
