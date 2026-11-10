#!/usr/bin/env python3
"""Test de generación real contra una instancia Studio IA, vía MCP.

Ejecuta el caso de aceptación declarado —dos vecinos andando por un parque,
parodia de la obra infinita, 30 s— de punta a punta y guarda los resultados
en `./salidas_studio` (o `--out`).

    python3 mcp_generation_test.py --url https://xxx.trycloudflare.com --token st_mcp_...
    python3 mcp_generation_test.py --url "https://xxx.trycloudflare.com/mcp?token=st_mcp_..." --fast

Sin dependencias externas: solo librería estándar, igual que studio_mcp.py.

Fases (se pueden saltar con `--from`): prepare, keyframes, audit, clips,
voices, lipsync, montage. Cada llamada MCP se espera con `studio_wait_for_job`
usando el MISMO job_id, como exige el manual. El token no se imprime nunca.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCENE = (
    "Dos vecinos, un hombre de unos 45 años con cazadora azul y barba corta y una mujer de unos 40 "
    "con abrigo verde y gafas, andando lado a lado por un parque urbano español con árboles, setos, "
    "un banco de madera y una fuente al fondo"
)
CHARS = (
    "Personaje izquierda: hombre de 45 años, pelo castaño corto, barba de tres días, cazadora vaquera azul "
    "y zapatillas. Personaje derecha: mujer de 40 años, melena castaña recogida, gafas de montura fina, "
    "abrigo verde y bufanda granate."
)
TURNS = [
    ("leftmost", "es-ES-AlvaroNeural",
     "Mari, que está el parque hecho un desastre, tres años con la obra y no avanzan, tres años, Mari."),
    ("rightmost", "es-ES-ElviraNeural",
     "Pues yo creo que en realidad están trabajando, lo que pasa es que trabajan despacio, que no es lo mismo."),
    ("leftmost", "es-ES-AlvaroNeural",
     "¡Mujer! Que el cartel pone fase dos, y el cartel es del año pasado, que lo leí yo, que estaba ahí escrito."),
    ("rightmost", "es-ES-ElviraNeural",
     "Bueno, en fase dos se incluye la reflexión, que es la parte más larga del proyecto, hombre ya."),
    ("leftmost", "es-ES-AlvaroNeural",
     "¡Anda ya! ¿Y el banco? Que llevo dos años sin bancos para sentarme y no puedo sentarme en ningún sitio."),
    ("rightmost", "es-ES-ElviraNeural",
     "Es que el banco está previsto, lo que pasa es que primero están los adoquines, que es lo primero, ¿eh?"),
]


class Client:
    def __init__(self, base: str, token: str, timeout: float = 60.0, out_dir: str = "salidas_studio") -> None:
        self.base = base.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.out_dir = Path(out_dir)
        self.session_id: Optional[str] = None
        self._id = 0
        self.steps: List[Dict[str, Any]] = []
        self.state_path = self.out_dir / "estado_test.json"

    # ---------------------------------------------------------------- JSON-RPC
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.base + "/mcp",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "studio-generation-test/1.0",
            },
        )
        if self.session_id:
            req.add_header("Mcp-Session-Id", self.session_id)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self.session_id = resp.headers.get("Mcp-Session-Id") or self.session_id
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise SystemExit(f"HTTP {exc.code} en /mcp: {detail}") from None
        text = body.decode("utf-8", "replace").strip()
        if text.startswith("data:"):
            text = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")][-1]
        return json.loads(text) if text else {}

    def handshake(self) -> Dict[str, Any]:
        self._id += 1
        init = self._post({
            "jsonrpc": "2.0", "id": self._id, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                       "clientInfo": {"name": "studio-generation-test", "version": "1.0"}},
        })
        return (init.get("result") or {})

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._id += 1
        resp = self._post({
            "jsonrpc": "2.0", "id": self._id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments, "_meta": {"progressToken": "gen-test"}},
        })
        if "error" in resp:
            return {"ok": False, "error_code": "RPC_ERROR", "message": resp["error"]}
        result = resp.get("result") or {}
        texts = [c.get("text", "") for c in (result.get("content") or []) if c.get("type") == "text"]
        raw = "\n".join(t for t in texts if t)
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        if not isinstance(payload, dict):
            payload = {"raw": raw}
        if result.get("isError"):
            payload.setdefault("ok", False)
        return payload

    def save_state(self, payload: Dict[str, Any]) -> None:
        """Guarda lo ya generado: un corte de Colab no debe costar la GPU gastada."""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def load_state(self) -> Dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ---------------------------------------------------------------- helpers
    def wait(self, job_id: str, label: str, budget_sec: float = 3600.0) -> Dict[str, Any]:
        """Espera el MISMO job_id; un timeout de red no es un fallo del motor."""
        started = time.time()
        last_pct = None
        while True:
            out = self.call("studio_wait_for_job", {"job_id": job_id, "wait_seconds": 80})
            status = out.get("status")
            pct = out.get("progress_pct")
            if pct is not None and pct != last_pct:
                print(f"    {label}: {pct}% · {out.get('current_step') or ''}", flush=True)
                last_pct = pct
            if status == "completed":
                return out
            if status in ("failed", "cancelled"):
                return out
            if time.time() - started > budget_sec:
                return {"ok": False, "error_code": "CLIENT_TIMEOUT", "status": status,
                        "message": f"Presupuesto de espera agotado para {label}"}

    def tool_and_wait(self, name: str, args: Dict[str, Any], label: str,
                      budget_sec: float = 3600.0) -> Dict[str, Any]:
        out = self.call(name, {**args, "wait": False})
        if out.get("error_code"):
            self.steps.append({"step": label, "tool": name, "ok": False, "error": out.get("error_code")})
            return out
        job_id = out.get("job_id") or (out.get("job") or {}).get("id")
        if not job_id:
            self.steps.append({"step": label, "tool": name, "ok": True, "job_id": None})
            return out
        print(f"  · {label}: job {job_id}", flush=True)
        done = self.wait(job_id, label, budget_sec=budget_sec)
        ok = done.get("status") == "completed"
        self.steps.append({"step": label, "tool": name, "job_id": job_id, "ok": ok,
                           "elapsed": done.get("elapsed_seconds"),
                           "error": None if ok else done.get("error") or done.get("error_code")})
        return done

    def asset(self, done: Dict[str, Any]) -> Optional[str]:
        if done.get("status") != "completed":
            return None
        res = done.get("result") or {}
        return res.get("rel_path") or res.get("media_rel") or None

    def fetch(self, rel: str) -> Optional[Path]:
        if not rel:
            return None
        target = self.out_dir / Path(rel).name
        try:
            req = urllib.request.Request(f"{self.base}/media/{rel}")
            req.add_header("Authorization", f"Bearer {self.token}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(resp.read())
            return target
        except Exception as exc:  # descarga opcional: no invalida la generación
            print(f"    (no se pudo descargar {rel}: {type(exc).__name__})", flush=True)
            return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Test de generación real (parodia del parque, 30 s).")
    ap.add_argument("--url", required=True, help="URL base del túnel o el enlace completo /mcp?token=...")
    ap.add_argument("--token", default=os.environ.get("STUDIO_MCP_TOKEN", ""))
    ap.add_argument("--out", default="salidas_studio")
    ap.add_argument("--fast", action="store_true", help="Solo 2 clips y 3 keyframes (test rápido)")
    ap.add_argument("--from", dest="start_at", default="prepare",
                    choices=["prepare", "keyframes", "audit", "clips", "voices", "lipsync", "montage"])
    ap.add_argument("--resume", action="store_true",
                    help="Reutiliza los assets ya generados en --out (estado_test.json)")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    url, token = args.url, args.token
    if "?" in url:
        base, _, query = url.partition("?")
        for part in query.split("&"):
            k, _, v = part.partition("=")
            if k.lower() == "token" and not token:
                token = v
        url = base
    if url.endswith("/mcp"):
        url = url[: -len("/mcp")]
    if not token:
        print("Falta el token (--token o STUDIO_MCP_TOKEN).", file=sys.stderr)
        return 2

    cli = Client(url.rstrip("/"), token.strip(), timeout=args.timeout, out_dir=args.out)
    info = cli.handshake()
    server = (info.get("serverInfo") or {}).get("name", "?")
    print(f"Conectado a {cli.base} ({server})", flush=True)

    state = cli.call("studio_get_state_and_manual", {"include_recent_assets": False})
    build = (state.get("build") or {}).get("commit_short")
    gpu = (state.get("hardware") or {}).get("gpu") or {}
    print(f"Build {build} · {gpu.get('gpu_name')} · {gpu.get('vram_free_gb')} GB VRAM libres", flush=True)

    presets = [q.get("key") for q in ((cli.call("studio_get_workflow_preset", {
        "preset_id": "coherencia_personajes_entorno_avance",
        "scene_description": "dos vecinos andando por un parque, 30 segundos",
    }).get("production_preferences") or {}).get("pending_questions") or [])]
    # El usuario ya decidió: LoRAs rápidos sí, clips de 5 s, 16:9, Álvaro/Elvira.
    for key, value in (("use_fast_loras", True), ("clip_duration_sec", 5.0), ("aspect_ratio", "16:9"),
                       ("voice_pair", ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"])):
        if key in presets:
            cli.call("studio_answer_preference", {"key": key, "value": value})
    print(f"Preferencias: {presets or 'ya estaban guardadas'}", flush=True)

    n_clips = 2 if args.fast else 6
    plan = cli.call("studio_get_workflow_preset", {
        "preset_id": "coherencia_personajes_entorno_avance",
        "scene_description": f"{SCENE}, el vídeo debe durar {n_clips * 5} segundos",
        "characters_description": CHARS,
        "num_clips": n_clips,
        "dialogue_lines": [t[2] for t in TURNS[:n_clips]],
    })
    if not plan.get("ok", True):
        print(json.dumps(plan, ensure_ascii=False)[:800], file=sys.stderr)
        return 1
    steps = plan.get("execution_plan") or []
    print(f"Plan: {json.dumps(plan.get('planned_duration_summary'), ensure_ascii=False)}", flush=True)
    stack = ((plan.get("detected_gpu") or {}).get("selected_video_stack") or {})
    print(f"Stack vídeo: {stack.get('model_id')} + {stack.get('video_lora_id')} + {stack.get('extra_video_lora_ids')}", flush=True)

    order = ["prepare", "keyframes", "audit", "clips", "voices", "lipsync", "montage"]
    active = set(order[order.index(args.start_at):])
    frames: List[str] = []
    clips: List[str] = []
    voices: List[str] = []
    ambience: Optional[str] = None
    master: Optional[str] = None
    if args.resume:
        prev = cli.load_state()
        frames = [f for f in (prev.get("frames") or []) if f]
        clips = [c for c in (prev.get("clips") or []) if c]
        voices = [v for v in (prev.get("voices") or []) if v]
        ambience = prev.get("ambience")
        master = prev.get("master")
        if frames or clips:
            print(f"Reanudando: {len(frames)} keyframes, {len(clips)} clips, {len(voices)} voces", flush=True)

    def _persist() -> None:
        cli.save_state({"frames": frames, "clips": clips, "voices": voices,
                        "ambience": ambience, "master": master,
                        "build": build, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})

    try:
        for step in steps:
            tool = step.get("tool")
            phase = str(step.get("phase") or "")
            args_ = dict(step.get("arguments") or {})

            if "prepare" in active and tool == "studio_prepare_model":
                done = cli.tool_and_wait("studio_prepare_model", args_, phase, budget_sec=5400)
                if done.get("status") != "completed":
                    print(f"  ✗ prepare falló: {done.get('error') or done.get('error_code')}", flush=True)
                    return 1
                continue

            if "keyframes" in active and tool == "studio_generate_or_edit_image":
                import re as _re

                _m = _re.search(r"FRAME_(\d+)_REL", str(step.get("save_output_as") or ""))
                if _m and len(frames) >= int(_m.group(1)):
                    continue  # ya generado en una ejecución anterior
                filled = _fill(args_, {"<FRAME_1_REL>": frames[0] if frames else "", **{
                    f"<FRAME_{i}_REL>": (frames[i - 1] if len(frames) >= i else "") for i in range(1, 9)}})
                done = cli.tool_and_wait(tool, filled, phase, budget_sec=1800)
                rel = cli.asset(done)
                if not rel:
                    print(f"  ✗ keyframe falló: {done.get('error') or done.get('error_code')}", flush=True)
                    return 1
                frames.append(rel)
                _persist()
                print(f"  ✓ {rel}", flush=True)
                cli.fetch(rel)
                continue

            if "audit" in active and tool == "studio_analyze_coherence":
                cands = [f for f in frames[1:]]
                if not cands:
                    continue
                filled = _fill(args_, {"<FRAME_1_REL>": frames[0]})
                filled["candidate_image_rels"] = cands
                done = cli.tool_and_wait(tool, filled, phase, budget_sec=1200)
                res = done.get("result") or {}
                print(f"  auditoría: all_consistent={res.get('all_consistent')} "
                      f"vlm={res.get('vlm_semantic_used')} veredictos="
                      f"{[v.get('verdict') for v in (res.get('per_candidate') or [])][:8]}", flush=True)
                if not res.get("all_consistent"):
                    print("  ⚠ auditoría no conforme: se continúa para ver el resultado, pero consta en el informe", flush=True)
                continue

            if "clips" in active and tool == "studio_generate_video":
                if len(clips) >= len(frames) - 1 and clips:
                    continue
                filled = _fill(args_, {**{f"<FRAME_{i}_REL>": (frames[i - 1] if len(frames) >= i else "")
                                         for i in range(1, 9)}})
                done = cli.tool_and_wait(tool, filled, phase, budget_sec=5400)
                rel = cli.asset(done)
                if not rel:
                    print(f"  ✗ clip falló: {done.get('error') or done.get('error_code')}", flush=True)
                    return 1
                clips.append(rel)
                _persist()
                print(f"  ✓ {rel} ({done.get('elapsed_seconds')}s)", flush=True)
                cli.fetch(rel)
                continue

            if "voices" in active and phase == "voices_and_street_ambience":
                for call in step.get("parallel_or_sequential_calls") or []:
                    c_args = dict(call.get("arguments") or {})
                    if call.get("tool") == "studio_generate_speech":
                        text = str(c_args.get("text") or "")
                        idx = len(voices)
                        if idx < len(TURNS):
                            c_args["text"], c_args["voice_id"] = TURNS[idx][2], TURNS[idx][1]
                    done = cli.tool_and_wait(call.get("tool"), c_args, call.get("tool"), budget_sec=900)
                    rel = cli.asset(done)
                    if not rel:
                        print(f"  ✗ {call.get('tool')} falló: {done.get('error') or done.get('error_code')}"
                              " (se continúa: el montaje saldrá sin esa pista)", flush=True)
                        continue
                    if call.get("tool") == "studio_generate_music_or_sfx":
                        ambience = rel
                    else:
                        voices.append(rel)
                    _persist()
                    print(f"  ✓ {rel}", flush=True)
                    cli.fetch(rel)
                continue

            if "lipsync" in active and phase == "lipsync_and_final_master":
                for call in step.get("recommended_calls") or []:
                    if call.get("tool") == "studio_prepare_model":
                        cli.tool_and_wait("studio_prepare_model", dict(call.get("arguments") or {}),
                                          "prepare lipsync", budget_sec=1800)
                        continue
                    if call.get("tool") == "studio_lipsync_character":
                        c_args = dict(call.get("arguments") or {})
                        turn = _match_turn(c_args.get("dialogue_turns"), voices)
                        c_args["dialogue_turns"] = turn
                        start = _video_index(c_args.get("media_rel"), clips)
                        if start is None:
                            continue
                        c_args["media_rel"] = clips[start]
                        done = cli.tool_and_wait("studio_lipsync_character", c_args, "lipsync", budget_sec=3600)
                        rel = cli.asset(done)
                        if rel:
                            clips[start] = rel
                            print(f"  ✓ lipsync {rel}", flush=True)
                            cli.fetch(rel)
                        else:
                            print(f"  ✗ lipsync falló: {done.get('error') or done.get('error_code')}", flush=True)
                        continue
                    if call.get("tool") == "studio_render_montage" and "montage" in active:
                        c_args = dict(call.get("arguments") or {})
                        c_args["items"] = [{"media_rel": c} for c in clips]
                        if ambience:
                            c_args["global_audio_rel"] = ambience
                        c_args["fade_seconds"] = 0.0
                        done = cli.tool_and_wait("studio_render_montage", c_args, "montaje", budget_sec=1800)
                        master = cli.asset(done)
                        if master:
                            _persist()
                            print(f"  ✓ MASTER {master}", flush=True)
                            cli.fetch(master)
                        else:
                            print(f"  ✗ montaje falló: {done.get('error') or done.get('error_code')}", flush=True)
    finally:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        report = Path(args.out) / f"informe_generacion_{stamp}.md"
        report.write_text(_report(cli, plan, build, gpu, frames, clips, voices, ambience, master), encoding="utf-8")
        print(f"\nInforme: {report}", flush=True)

    ok = bool(master or clips)
    return 0 if ok else 1


def _fill(obj: Any, mapping: Dict[str, str]) -> Any:
    """Sustituye los placeholders `<FRAME_i_REL>` por los rel_path reales."""
    if isinstance(obj, str):
        for key, value in mapping.items():
            if key in obj and value:
                obj = obj.replace(key, value)
        return obj
    if isinstance(obj, dict):
        return {k: _fill(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, mapping) for v in obj]
    return obj


def _match_turn(turns: Any, voices: List[str]) -> List[Dict[str, Any]]:
    """El plan sugiere un turno por clip; se empareja con la voz ya generada."""
    if not turns:
        return []
    out = []
    for turn in turns:
        idx = len(out)
        if idx < len(voices):
            out.append({**turn, "voice_audio_rel": voices[idx], "start_sec": 0.0})
    return out


def _video_index(rel: Any, clips: List[str]) -> Optional[int]:
    """`RAW_VIDEO_CLIP_4_REL` → índice del clip (el plan lo nombra, no lo conoce)."""
    import re

    m = re.search(r"CLIP_(\d+)_REL", str(rel or ""))
    if m:
        idx = int(m.group(1)) - 1
        return idx if 0 <= idx < len(clips) else None
    return None


def _report(cli: Client, plan: Dict[str, Any], build: Optional[str], gpu: Dict[str, Any],
            frames: List[str], clips: List[str], voices: List[str], ambience: Optional[str],
            master: Optional[str]) -> str:
    lines = [
        "# Test de generación — parodia del parque",
        "",
        f"- **Instancia:** `{cli.base}`",
        f"- **Build:** `{build}`",
        f"- **GPU:** {gpu.get('gpu_name')} · {gpu.get('vram_total_gb')} GB",
        f"- **Plan:** {json.dumps(plan.get('planned_duration_summary'), ensure_ascii=False)}",
        f"- **Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Pasos ejecutados",
        "",
        "| Paso | Herramienta | Estado | job | segundos | error |",
        "|---|---|---|---|---|---|",
    ]
    for s in cli.steps:
        lines.append(
            f"| {s.get('step')} | {s.get('tool')} | {'✅' if s.get('ok') else '❌'} | "
            f"{s.get('job_id') or '-'} | {s.get('elapsed') or '-'} | {s.get('error') or ''} |"
        )
    lines += ["", "## Salidas", "", f"- Keyframes: {len(frames)}", f"- Clips: {len(clips)}",
              f"- Voces: {len(voices)}", f"- Ambiente: {ambience or '-'}", f"- **Master: {master or 'no generado'}**", ""]
    lines += ["## Diálogo", ""]
    for i, (_, _, text) in enumerate(TURNS, 1):
        lines.append(f"{i}. {text}")
    lines += ["", "## Límites", "",
              "- Los tiempos y la calidad son los que reporte la instancia; este informe no los estima.",
              "- Si algún paso falló, el error está en la tabla, no se reintenta a ciegas ni se descargan alternativas.",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
