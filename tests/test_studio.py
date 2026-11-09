import base64
import io
import os
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


def _make_png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (640, 360), label: str = "A") -> bytes:
    im = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(im)
    draw.rectangle([40, 40, size[0] - 40, size[1] - 40], outline=(255, 255, 255), width=6)
    draw.text((80, 80), f"Plano {label}", fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _wait_job(client: TestClient, job_id: str, timeout: float = 45.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        last = r.json()
        if last["status"] in ("completed", "failed", "cancelled"):
            return last
        time.sleep(0.25)
    raise TimeoutError(f"El trabajo {job_id} no terminó a tiempo: {last}")


def test_index_status_and_default_qwen_catalog(temp_studio_env):
    client, _ = temp_studio_env

    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "STUDIO IA" in r_home.text

    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    st = r_status.json()
    assert "hardware" in st
    assert "credits" in st["hardware"]

    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    models = r_cat.json()["models"]
    ids = {m["id"]: m for m in models}

    assert "qwen-image-edit-2511-pack" in ids
    assert ids["qwen-image-edit-2511-pack"]["is_default"] is True
    assert ids["qwen-image-edit-2511-pack"]["category"] == "image"

    assert "wan22-ti2v-5b" in ids
    assert "ltx-video-2b" in ids
    assert "sana-video-2b-480p" in ids
    assert "ltx-2-19b-av" in ids
    assert "edge-tts-cloud" in ids


def test_disk_space_warning_and_model_delete(temp_studio_env):
    client, _ = temp_studio_env
    from studio.catalog import catalog_manager

    # Simular que wan22-ti2v-5b ocupa archivos en disco
    wan_dir = catalog_manager.model_dir("wan22-ti2v-5b") / "base"
    wan_dir.mkdir(parents=True, exist_ok=True)
    dummy_weight = wan_dir / "dummy.safetensors"
    dummy_weight.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

    # Comprobar aviso cuando solo quedan 2 GB libres pero Qwen pide ~12.2 GB + margen
    check = catalog_manager.check_disk_space(
        "qwen-image-edit-2511-pack",
        free_bytes_override=2 * (1024**3),
    )
    assert check["ok"] is False
    assert check["missing_gb"] > 8.0
    assert any(s["id"] == "wan22-ti2v-5b" for s in check["suggestions"])
    assert "Espacio insuficiente" in check["message"]

    # Borrar el modelo simulado mediante la API y comprobar limpieza
    r_del = client.delete("/api/catalog/wan22-ti2v-5b")
    assert r_del.status_code == 200
    assert not wan_dir.exists()


def test_custom_hf_model_and_credits_and_memory_unload(temp_studio_env):
    client, _ = temp_studio_env
    from studio.memory_manager import memory_manager

    r_custom = client.post(
        "/api/catalog/custom",
        json={
            "name": "LoRA Cine Test",
            "repo_id": "usuario/lora-cine-test",
            "category": "lora",
            "filename": "cine.safetensors",
            "estimated_gb": 0.4,
        },
    )
    assert r_custom.status_code == 200
    new_id = r_custom.json()["model"]["id"]
    cat_ids = [m["id"] for m in client.get("/api/catalog").json()["models"]]
    assert new_id in cat_ids

    # Configurar contador de créditos
    r_cred = client.post(
        "/api/credits/config",
        json={"initial_credits": 85.5, "hourly_rate": 4.82, "idle_alert_minutes": 12},
    )
    assert r_cred.status_code == 200
    cr = r_cred.json()["hardware"]["credits"]
    assert cr["initial_credits"] == 85.5
    assert cr["hourly_rate"] == 4.82

    # Registrar un objeto simulado en memoria RAM/VRAM y descargarlo sin tocar disco
    memory_manager.register_loaded("image", "qwen-image-edit-2511-pack", "Qwen Pack Test", {"dummy": [1, 2, 3]})
    assert "qwen-image-edit-2511-pack" in memory_manager.loaded_model_ids()

    r_un = client.post("/api/memory/unload", json={"target": "image"})
    assert r_un.status_code == 200
    assert memory_manager.loaded_model_ids() == []


def test_hang_detector_over_30_seconds():
    from studio.jobs import JobRecord

    job = JobRecord(kind="image", title="Proceso lento simulado")
    job.status = "running"
    job.started_ts = time.time() - 45.0
    job.last_heartbeat_ts = time.time() - 35.0

    snap = job.to_dict()
    assert snap["possibly_hung"] is True
    assert snap["seconds_since_update"] >= 34.0
    assert "Posible cuelgue" in snap["hang_warning"]


def test_inpaint_mask_composite_preserves_unmasked_pixels():
    from studio.engines.ffmpeg_utils import apply_inpaint_composite

    orig = Image.new("RGB", (200, 200), color=(10, 20, 30))
    edited = Image.new("RGB", (200, 200), color=(240, 50, 60))
    mask = Image.new("L", (200, 200), color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([80, 80, 140, 140], fill=255)

    comp = apply_inpaint_composite(orig, edited, mask, feather_px=0)
    # Esquina fuera de la máscara debe quedar idéntica al píxel original
    assert comp.getpixel((10, 10)) == (10, 20, 30)
    # Centro de la máscara debe tener el píxel editado
    assert comp.getpixel((110, 110)) == (240, 50, 60)


def test_full_upload_bt709_video_clip_montage_and_zip_export(temp_studio_env):
    client, _ = temp_studio_env
    from studio.config import MONTAGES_OUT_DIR
    from studio.engines.ffmpeg_utils import probe_media_info

    # 1. Subir 2 imágenes reales
    img1_bytes = _make_png_bytes((35, 90, 180), label="1")
    img2_bytes = _make_png_bytes((160, 55, 95), label="2")

    r_up1 = client.post("/api/assets/upload", files={"file": ("plano1.png", img1_bytes, "image/png")})
    r_up2 = client.post("/api/assets/upload", files={"file": ("plano2.png", img2_bytes, "image/png")})
    assert r_up1.status_code == 200
    assert r_up2.status_code == 200
    rel1 = r_up1.json()["asset"]["rel_path"]
    rel2 = r_up2.json()["asset"]["rel_path"]

    # 2. Subir máscara base64 del lienzo de Inpainting
    mask_im = Image.new("L", (640, 360), color=0)
    ImageDraw.Draw(mask_im).ellipse([200, 100, 420, 260], fill=255)
    mbuf = io.BytesIO()
    mask_im.save(mbuf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(mbuf.getvalue()).decode("ascii")
    r_mask = client.post("/api/assets/upload-mask", json={"data_url": data_url})
    assert r_mask.status_code == 200

    # 3. Ejecutar normalización FFmpeg BT.709 real sobre plano1
    r_bt = client.post(
        "/api/image/generate",
        json={"base_image_rel": rel1, "width": 640, "height": 360, "cpu_bt709_only": True},
    )
    assert r_bt.status_code == 200
    j_bt = _wait_job(client, r_bt.json()["job"]["id"])
    assert j_bt["status"] == "completed"
    bt_rel = f"outputs/images/{j_bt['result']['filename']}"

    # 4. Generar clip de vídeo animado con FFmpeg a partir de la imagen BT.709
    r_vid = client.post(
        "/api/video/generate",
        json={
            "mode": "ffmpeg_motion",
            "image_rel": bt_rel,
            "width": 640,
            "height": 360,
            "duration_sec": 1.5,
            "fps": 24,
            "motion_style": "zoom_in",
        },
    )
    assert r_vid.status_code == 200
    j_vid = _wait_job(client, r_vid.json()["job"]["id"])
    assert j_vid["status"] == "completed"
    vid_rel = f"outputs/videos/{j_vid['result']['filename']}"

    # 5. Renderizar montaje final MP4 combinando el clip animado + la segunda imagen
    r_mon = client.post(
        "/api/montage/render",
        json={
            "title": "demo_final",
            "width": 640,
            "height": 360,
            "fps": 24,
            "fade_seconds": 0.15,
            "items": [
                {"media_rel": vid_rel, "duration": 1.5, "motion": "none"},
                {"media_rel": rel2, "duration": 1.5, "motion": "pan_right"},
            ],
        },
    )
    assert r_mon.status_code == 200
    j_mon = _wait_job(client, r_mon.json()["job"]["id"])
    assert j_mon["status"] == "completed"

    final_file = MONTAGES_OUT_DIR / j_mon["result"]["filename"]
    info = probe_media_info(final_file)
    assert info["has_video"] is True
    assert info["has_audio"] is True
    assert info["duration_seconds"] >= 2.5

    # 6. Verificar exportación .zip completa antes de apagar sesión
    r_zip = client.get("/api/export/zip")
    assert r_zip.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r_zip.content))
    names = zf.namelist()
    assert any(n.startswith("imagenes/") and n.endswith(".jpg") for n in names)
    assert any(n.startswith("videos/") and n.endswith(".mp4") for n in names)
    assert any(n.startswith("montajes_mp4/") and n.endswith(".mp4") for n in names)


def test_l4_rate_is_192_and_realtime_colab_stdout_logs(capsys):
    from studio.hardware import GPU_RATES_DEFAULT
    from studio.jobs import JobRecord
    from studio.memory_manager import run_with_live_memory_progress

    assert GPU_RATES_DEFAULT["L4"] == 1.92

    job = JobRecord(kind="load_model", title="Preparar Qwen + Lightning")
    job.log("Descargando archivo 1/5...", progress=25.0, step="[1/5] model.safetensors")
    job.heartbeat(progress=32.5, step="[1/5] model.safetensors (3.9 / 12.2 GB)", speed_text="120.4 MB/s")

    out_val = run_with_live_memory_progress(
        job=job,
        start_pct=70.0,
        end_pct=90.0,
        step_label="Cargando tensores 4-bit",
        fn=lambda: (time.sleep(1.7) or "ok"),
        expected_seconds=10.0,
    )
    assert out_val == "ok"
    assert job.progress > 70.0

    captured = capsys.readouterr().out
    assert "25.0%" in captured
    assert "32.5%" in captured
    assert "120.4 MB/s" in captured


def test_monotonic_progress_and_colab_torchao_cleanup():
    import launch_colab
    from studio.jobs import JobRecord

    rec = JobRecord(kind="model_download", title="Test Monotonic")
    rec.status = "running"
    rec.log("Inicio", progress=3.8)
    rec.log("Archivo pequeño", progress=1.0)
    assert rec.progress == 3.8
    rec.heartbeat(progress=2.0)
    assert rec.progress == 3.8
    rec.log("Avance real", progress=45.5)
    assert rec.progress == 45.5

    launch_colab._remove_incompatible_preinstalled_packages()


def test_gpu_filter_stickers_artifact_patterns_and_hf_search(temp_studio_env, monkeypatch):
    client, _ = temp_studio_env
    from studio.catalog import (
        DEFAULT_CATALOG,
        _DIFFUSERS_VIDEO_ALLOW,
        _matches_artifact_filters,
        evaluate_gpu_tier,
        resolve_gpu_profile,
    )

    video_defs = [m for m in DEFAULT_CATALOG if m["category"] == "video"]
    assert len(video_defs) >= 10
    assert len(DEFAULT_CATALOG) >= 19

    # Verificar filtro de archivos que evita bajar 280 GB de checkpoints raíz duplicados en LTX-Video / LTX-2
    assert _matches_artifact_filters(
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        _DIFFUSERS_VIDEO_ALLOW,
        None,
        True,
    )
    assert _matches_artifact_filters("model_index.json", _DIFFUSERS_VIDEO_ALLOW, None, True)
    assert not _matches_artifact_filters("ltx-video-2b-v0.9.5.safetensors", _DIFFUSERS_VIDEO_ALLOW, None, True)
    assert not _matches_artifact_filters("media/demo.gif", _DIFFUSERS_VIDEO_ALLOW, None, True)

    # Verificar pegatinas en NVIDIA L4 (22.5 GB) vs T4 (15 GB) vs A100 80 GB
    prof_l4 = resolve_gpu_profile("L4")
    prof_t4 = resolve_gpu_profile("T4")
    prof_a100_80 = resolve_gpu_profile("A100_80")

    ltx_2b = next(m for m in video_defs if m["id"] == "ltx-video-2b")
    wan_27b = next(m for m in video_defs if m["id"] == "wan22-i2v-a14b")
    cog_5b = next(m for m in video_defs if m["id"] == "cogvideox-5b-i2v")

    c_ltx_l4 = evaluate_gpu_tier(ltx_2b, prof_l4)
    assert c_ltx_l4["compatible"] is True
    assert c_ltx_l4["tier"] == "excellent"
    assert "Excelente" in c_ltx_l4["sticker"]

    c_cog_t4 = evaluate_gpu_tier(cog_5b, prof_t4)
    assert c_cog_t4["compatible"] is True
    assert c_cog_t4["tier"] in ("normal", "good", "slow")

    c_wan27_l4 = evaluate_gpu_tier(wan_27b, prof_l4)
    assert c_wan27_l4["compatible"] is False
    assert c_wan27_l4["tier"] == "incompatible"
    assert "No cabe" in c_wan27_l4["sticker"]

    c_wan27_a100 = evaluate_gpu_tier(wan_27b, prof_a100_80)
    assert c_wan27_a100["compatible"] is True
    assert c_wan27_a100["tier"] == "excellent"

    # This regression isolates GPU filtering. A GPU simulation deliberately keeps
    # real RAM/disk constraints, so supply an explicit complete hardware snapshot
    # instead of depending on whichever small CPU CI machine runs the suite.
    from studio.hardware import hardware_tracker
    from studio.model_discovery import MetadataError, model_discovery
    snapshot = hardware_tracker.get_snapshot()
    snapshot["ram"].update(total_gb=192.0, available_gb=180.0)
    snapshot["disk"].update(free_gb=512.0)
    monkeypatch.setattr(hardware_tracker, "get_snapshot", lambda **_kw: snapshot)

    def offline_provider(*_a, **_kw):
        raise MetadataError("offline", "Proveedor desconectado para esta prueba determinista.")

    monkeypatch.setattr(model_discovery.provider, "search_page", offline_provider)
    r_l4 = client.get("/api/catalog?gpu_profile=L4&only_compatible=true")
    assert r_l4.status_code == 200
    data_l4 = r_l4.json()
    assert data_l4["active_gpu_profile"]["id"] == "L4"
    l4_ids = {m["id"] for m in data_l4["models"]}
    assert "ltx-video-2b" in l4_ids
    assert "wan22-ti2v-5b" in l4_ids
    assert "sana-video-2b-480p" in l4_ids
    assert "wan22-i2v-a14b" not in l4_ids
    assert all(m["compatibility"]["compatible"] for m in data_l4["models"])
    assert all(len(m["gpu_matrix"]) == 4 for m in data_l4["models"])

    r_hf = client.get("/api/catalog/hf-search?query=Wan&category=video&gpu_profile=L4&only_compatible=true")
    assert r_hf.status_code == 200
    hf_json = r_hf.json()
    assert hf_json["ok"] is True
    assert len(hf_json["items"]) >= 1
    assert all("sticker" in it["compatibility"] for it in hf_json["items"])

    # Verificar gestión opcional de Token HF en memoria de sesión (sin persistencia pública)
    r_auth0 = client.get("/api/hf/auth")
    assert r_auth0.status_code == 200
    assert r_auth0.json()["has_token"] is False

    r_bad = client.post("/api/hf/token", json={"token": "abc"})
    assert r_bad.status_code == 400

    r_set = client.post("/api/hf/token", json={"token": "hf_testsessiontoken123456"})
    assert r_set.status_code == 200
    assert r_set.json()["hf_auth"]["has_token"] is True
    assert "hf_te...3456" == r_set.json()["hf_auth"]["masked_token"]

    r_del = client.delete("/api/hf/token")
    assert r_del.status_code == 200
    assert r_del.json()["hf_auth"]["has_token"] is False


def test_wan_pipeline_switching_strips_image_encoder_and_preserves_expand_timesteps():
    from studio.engines.video_engine import _switch_pipeline_class

    class DummyVae:
        def __init__(self):
            self.slicing = False
            self.tiling = False

        def enable_slicing(self):
            self.slicing = True

        def enable_tiling(self):
            self.tiling = True

    class MockWanImageToVideoPipeline:
        def __init__(
            self,
            tokenizer,
            text_encoder,
            vae,
            scheduler,
            image_processor=None,
            image_encoder=None,
            transformer=None,
            transformer_2=None,
            boundary_ratio=None,
            expand_timesteps=False,
        ):
            self.vae = vae
            self.offloaded = False
            self.hooks_removed = False
            self.components = {
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
                "vae": vae,
                "scheduler": scheduler,
                "image_processor": image_processor,
                "image_encoder": image_encoder,
                "transformer": transformer,
                "transformer_2": transformer_2,
            }
            self.config = {
                "_class_name": "WanImageToVideoPipeline",
                "boundary_ratio": boundary_ratio,
                "expand_timesteps": expand_timesteps,
            }

        def remove_all_hooks(self):
            self.hooks_removed = True

        def enable_model_cpu_offload(self):
            self.offloaded = True

    class MockWanPipeline:
        # Firma idéntica a diffusers==0.42.1 WanPipeline.__init__ (NO acepta image_encoder ni image_processor)
        def __init__(
            self,
            tokenizer,
            text_encoder,
            vae,
            scheduler,
            transformer=None,
            transformer_2=None,
            boundary_ratio=None,
            expand_timesteps=False,
        ):
            self.vae = vae
            self.offloaded = False
            self.hooks_removed = False
            self.components = {
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
                "vae": vae,
                "scheduler": scheduler,
                "transformer": transformer,
                "transformer_2": transformer_2,
            }
            self.config = {
                "_class_name": "WanPipeline",
                "boundary_ratio": boundary_ratio,
                "expand_timesteps": expand_timesteps,
            }

        def remove_all_hooks(self):
            self.hooks_removed = True

        def enable_model_cpu_offload(self):
            self.offloaded = True

    vae = DummyVae()
    pipe_i2v = MockWanImageToVideoPipeline(
        tokenizer="tok",
        text_encoder="te",
        vae=vae,
        scheduler="sched",
        image_processor="clip_proc",
        image_encoder="clip_enc",
        transformer="dit_5b",
        expand_timesteps=True,
    )
    handle = {
        "pipe": pipe_i2v,
        "all_components": dict(pipe_i2v.components),
        "extra_init_kwargs": {"expand_timesteps": True, "boundary_ratio": None},
        "min_vram_gb": 13.5,
    }

    # 1) Conmutar de I2V -> T2V sin fallar por 'image_encoder' y preservando expand_timesteps=True
    pipe_t2v = _switch_pipeline_class(handle, MockWanPipeline, vram_total_gb=22.5)
    assert isinstance(pipe_t2v, MockWanPipeline)
    assert pipe_i2v.hooks_removed is True
    assert pipe_t2v.offloaded is True
    assert pipe_t2v.vae.slicing is True
    assert pipe_t2v.vae.tiling is True
    assert pipe_t2v.config["expand_timesteps"] is True
    assert pipe_t2v.components["transformer"] == "dit_5b"

    # 2) Conmutar de vuelta T2V -> I2V recuperando intactos image_encoder e image_processor
    pipe_back = _switch_pipeline_class(handle, MockWanImageToVideoPipeline, vram_total_gb=22.5)
    assert isinstance(pipe_back, MockWanImageToVideoPipeline)
    assert pipe_back.components["image_encoder"] == "clip_enc"
    assert pipe_back.components["image_processor"] == "clip_proc"
    assert pipe_back.config["expand_timesteps"] is True
    assert pipe_back.offloaded is True


def test_video_three_modes_t2v_i2v_and_frame_to_frame(temp_studio_env):
    from PIL import Image
    from studio.catalog import DEFAULT_CATALOG

    client, _ = temp_studio_env

    # Verificar que el catálogo ofrece modelos con soporte explícito de t2v, i2v y flf2v (Frame a Frame)
    vdefs = {m["id"]: m for m in DEFAULT_CATALOG if m["category"] == "video"}
    assert set(vdefs["ltx-video-2b"]["video_modes"]) >= {"t2v", "i2v", "flf2v"}
    assert set(vdefs["wan21-vace-1.3b"]["video_modes"]) >= {"t2v", "i2v", "flf2v"}
    assert set(vdefs["skyreels-v2-i2v-1.3b"]["video_modes"]) >= {"i2v", "flf2v"}
    assert "wan21-flf2v-14b-720p" in vdefs
    assert "flf2v" in vdefs["wan21-flf2v-14b-720p"]["video_modes"]

    # Subir Fotograma Inicial (azul) y Fotograma Final (naranja)
    def _png_bytes(rgb):
        buf = io.BytesIO()
        Image.new("RGB", (640, 360), rgb).save(buf, format="PNG")
        return buf.getvalue()

    up1 = client.post(
        "/api/assets/upload",
        files={"file": ("frame_inicio.png", _png_bytes((30, 90, 210)), "image/png")},
    ).json()["asset"]
    up2 = client.post(
        "/api/assets/upload",
        files={"file": ("frame_final.png", _png_bytes((220, 110, 35)), "image/png")},
    ).json()["asset"]

    # Validación 400 si falta imagen en i2v o falta fotograma final en flf2v
    r_err_i2v = client.post("/api/video/generate", json={"mode": "ai", "video_submode": "i2v", "prompt": "hola"})
    assert r_err_i2v.status_code == 400
    r_err_flf = client.post(
        "/api/video/generate",
        json={"mode": "ai", "video_submode": "flf2v", "image_rel": up1["rel_path"], "prompt": "transición"},
    )
    assert r_err_flf.status_code == 400

    # Ejecución real de Frame a Frame (Inicio -> Fin) en CPU con FFmpeg xfade
    r_ok = client.post(
        "/api/video/generate",
        json={
            "mode": "ffmpeg_motion",
            "video_submode": "flf2v",
            "image_rel": up1["rel_path"],
            "end_image_rel": up2["rel_path"],
            "width": 640,
            "height": 360,
            "duration_sec": 2.0,
            "fps": 24,
            "motion_style": "pan_right",
        },
    )
    assert r_ok.status_code == 200
    job_id = r_ok.json()["job"]["id"]

    deadline = time.time() + 25.0
    final_job = None
    while time.time() < deadline:
        st = client.get("/api/status").json()
        found = next((j for j in st["jobs"] if j["id"] == job_id), None)
        if found and found["status"] in ("completed", "failed"):
            final_job = found
            break
        time.sleep(0.25)

    assert final_job is not None, "El trabajo Frame a Frame no terminó a tiempo"
    assert final_job["status"] == "completed", final_job.get("error")
    assert final_job["result"]["video_submode"] == "flf2v"
    assert final_job["result"]["duration_seconds"] >= 1.5


def test_official_model_specs_prevent_hallucination_and_incompatible_flf2v(temp_studio_env):
    from PIL import Image
    from studio.catalog import DEFAULT_CATALOG
    from studio.engines.video_engine import _fit_and_crop_rgb, compute_official_video_params

    client, _ = temp_studio_env
    vdefs = {m["id"]: m for m in DEFAULT_CATALOG if m["category"] == "video"}

    # 1) Wan 2.2 TI2V-5B exige 720P (704×1280 / 1280×704) y 121 frames @ 5s/24fps (4k+1), NUNCA 480×832
    wan22 = vdefs["wan22-ti2v-5b"]
    p_wan22 = compute_official_video_params(
        defn=wan22,
        req_width=480,
        req_height=832,
        req_duration_sec=5.0,
        req_fps=24,
        req_steps=45,
        req_cfg=None,
        img_size=(1080, 1920),
    )
    assert (p_wan22["width"], p_wan22["height"]) == (704, 1280)
    assert p_wan22["num_frames"] == 121
    assert p_wan22["guidance_scale"] == 5.0

    # También en T2V sin imagen, si llegan 480×832 sube a 704×1280 por min_area
    p_wan22_t2v = compute_official_video_params(
        defn=wan22,
        req_width=480,
        req_height=832,
        req_duration_sec=5.0,
        req_fps=24,
        req_steps=50,
        req_cfg=None,
        img_size=None,
    )
    assert (p_wan22_t2v["width"], p_wan22_t2v["height"]) == (704, 1280)

    # 2) Wan 2.1 VACE 1.3B usa 480P (480×832 / 832×480), 81 frames @ 5s/16fps, CFG 5.0
    vace = vdefs["wan21-vace-1.3b"]
    p_vace = compute_official_video_params(
        defn=vace,
        req_width=480,
        req_height=832,
        req_duration_sec=5.0,
        req_fps=16,
        req_steps=40,
        req_cfg=None,
        img_size=(900, 1600),
    )
    assert (p_vace["width"], p_vace["height"]) == (480, 832)
    assert p_vace["num_frames"] == 81

    # 3) SkyReels V2 1.3B 540P usa 544×960 / 960×544, 97 frames @ 4s/24fps
    sky = vdefs["skyreels-v2-i2v-1.3b"]
    p_sky = compute_official_video_params(
        defn=sky,
        req_width=832,
        req_height=480,
        req_duration_sec=4.0,
        req_fps=24,
        req_steps=45,
        req_cfg=None,
        img_size=(1080, 1920),
    )
    assert (p_sky["width"], p_sky["height"]) == (544, 960)
    assert p_sky["num_frames"] == 97

    # 4) LTX-Video 0.9.5 usa CFG 3.0 por defecto y regla 8k+1 (97 frames @ 4s/24fps)
    ltx = vdefs["ltx-video-2b"]
    p_ltx = compute_official_video_params(
        defn=ltx,
        req_width=768,
        req_height=512,
        req_duration_sec=4.0,
        req_fps=24,
        req_steps=40,
        req_cfg=None,
        img_size=None,
    )
    assert p_ltx["guidance_scale"] == 3.0
    assert p_ltx["num_frames"] == 97

    # 5) CogVideoX-5B-I2V fuerza estrictamente 720×480 @ 8 fps y 49 frames
    cog = vdefs["cogvideox-5b-i2v"]
    p_cog = compute_official_video_params(
        defn=cog,
        req_width=480,
        req_height=832,
        req_duration_sec=4.0,
        req_fps=24,
        req_steps=50,
        req_cfg=None,
        img_size=(800, 1200),
    )
    assert (p_cog["width"], p_cog["height"]) == (720, 480)
    assert p_cog["fps"] == 8
    assert p_cog["num_frames"] == 49

    # 6) Center-crop conserva el tamaño exacto sin aplastar imagen 3:4
    cropped = _fit_and_crop_rgb(Image.new("RGB", (900, 1200), (80, 90, 100)), 704, 1280)
    assert cropped.size == (704, 1280)

    # 7) Wan 2.2 TI2V-5B es especialista honesto I2V/T2V; Frame a Frame usa Wan 2.1 VACE 1.3B / Wan 2.1 Fun InP / SkyReels V2
    assert "flf2v" not in wan22["video_modes"]
    vace = vdefs["wan21-vace-1.3b"]
    fun_inp = vdefs["wan21-fun-1.3b-inp"]
    assert "flf2v" in vace["video_modes"] and vace["default_steps"] == 30 and vace["default_cfg"] == 5.0
    assert "flf2v" in fun_inp["video_modes"]
    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (20, 40, 60)).save(buf, format="PNG")
    raw = buf.getvalue()
    u1 = client.post("/api/assets/upload", files={"file": ("a.png", raw, "image/png")}).json()["asset"]
    u2 = client.post("/api/assets/upload", files={"file": ("b.png", raw, "image/png")}).json()["asset"]
    for non_flf_id in ("cogvideox-5b-i2v", "wan22-ti2v-5b"):
        r_bad_flf = client.post(
            "/api/video/generate",
            json={
                "mode": "ai",
                "video_submode": "flf2v",
                "model_id": non_flf_id,
                "image_rel": u1["rel_path"],
                "end_image_rel": u2["rel_path"],
                "prompt": "transición",
            },
        )
        assert r_bad_flf.status_code == 400
        assert "Wan 2.1 VACE 1.3B" in r_bad_flf.json()["detail"]


def test_video_speed_loras_and_steps_reset_controls(temp_studio_env, monkeypatch):
    import numpy as np
    from studio.catalog import DEFAULT_CATALOG, catalog_manager
    from studio.engines.video_engine import _prepare_wan_lora_state_dict, _sync_video_lora_on_handle

    client, _ = temp_studio_env

    # 1) Controles UI de pasos seguros, slider, botón resetear y selector de LoRA de vídeo presentes en index.html
    html = client.get("/").text
    assert 'id="vid-steps-slider"' in html
    assert 'id="btn-reset-vid-steps"' in html
    assert 'id="btn-steps-fast-safe"' in html
    assert 'id="vid-lora-select"' in html

    # 2) Verificar LoRAs aceleradores reales para Wan 2.2 TI2V-5B y Wan 2.1 1.3B en el catálogo
    loras = {m["id"]: m for m in DEFAULT_CATALOG if m["category"] == "lora" and m.get("lora_target") == "video"}
    assert "lora-wan22-ti2v-5b-turbo" in loras
    assert "lora-wan22-ti2v-5b-fastwan" in loras
    assert "lora-wan21-1.3b-rcm-turbo" in loras
    turbo5b = loras["lora-wan22-ti2v-5b-turbo"]
    assert "wan22-ti2v-5b" in turbo5b["compatible_models"]
    assert turbo5b["recommended_steps"] == 6
    assert turbo5b["recommended_cfg"] == 1.0
    assert (
        turbo5b["artifacts"][0]["filename"]
        == "LoRAs/Wan22-Turbo/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
    )

    # 3) _prepare_wan_lora_state_dict normaliza prefijos desnudos y aplica .alpha fuera de blocks.*
    fake_down = np.ones((4, 16), dtype=np.float32)
    raw_sd = {
        "blocks.0.self_attn.q.lora_down.weight": fake_down,
        "blocks.0.self_attn.q.lora_up.weight": np.ones((16, 4), dtype=np.float32),
        "text_embedding.0.lora_down.weight": fake_down.copy(),
        "text_embedding.0.lora_up.weight": np.ones((16, 4), dtype=np.float32),
        "text_embedding.0.alpha": np.float32(2.0),  # rank=4 -> escala 0.5
        "vace_blocks.0.proj.lora_down.weight": fake_down,
    }
    clean_sd = _prepare_wan_lora_state_dict(raw_sd)
    assert "diffusion_model.blocks.0.self_attn.q.lora_down.weight" in clean_sd
    assert "diffusion_model.text_embedding.0.alpha" not in clean_sd
    assert "diffusion_model.vace_blocks.0.proj.lora_down.weight" not in clean_sd
    assert np.allclose(clean_sd["diffusion_model.text_embedding.0.lora_down.weight"], 0.5)

    # 4) _sync_video_lora_on_handle activa, ajusta escala y desactiva limpiamente un LoRA con CPU Offload
    lora_id = "lora-wan22-ti2v-5b-turbo"
    m_dir = catalog_manager.model_dir(lora_id)
    art = turbo5b["artifacts"][0]
    target_sf = m_dir / art["subdir"] / art["filename"]
    target_sf.parent.mkdir(parents=True, exist_ok=True)
    target_sf.write_bytes(b"fake_safetensors")
    catalog_manager.marker_file(lora_id).write_text("{}", encoding="utf-8")
    assert catalog_manager.is_downloaded(lora_id) is True

    class MockLoraPipe:
        def __init__(self):
            self.hooks_removed = 0
            self.offloaded = 0
            self.loaded_sd = None
            self.adapter_weights = None
            self.unloaded = 0
            self.vae = None

        def remove_all_hooks(self):
            self.hooks_removed += 1

        def enable_model_cpu_offload(self):
            self.offloaded += 1

        def load_lora_weights(self, sd_or_dir, **kwargs):
            self.loaded_sd = sd_or_dir

        def set_adapters(self, names, adapter_weights=None):
            self.adapter_weights = (names, adapter_weights)

        def unload_lora_weights(self):
            self.unloaded += 1

    import types, sys

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.load_file = lambda _p, device="cpu": raw_sd
    fake_st.torch = fake_st_torch
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    mpipe = MockLoraPipe()
    handle = {
        "engine": "wan_ti2v",
        "min_vram_gb": 13.5,
        "defn": catalog_manager.get_definition("wan22-ti2v-5b"),
    }

    res_l = _sync_video_lora_on_handle(handle, mpipe, lora_id, video_lora_scale=0.85)
    assert res_l is not None and res_l["id"] == lora_id
    assert handle["active_lora_id"] == lora_id
    assert mpipe.adapter_weights == (["vid_speed_lora"], [0.85])
    assert mpipe.offloaded == 1

    # Desactivar LoRA restaura pesos base y vuelve a activar CPU Offload
    _sync_video_lora_on_handle(handle, mpipe, None)
    assert handle["active_lora_id"] is None
    assert mpipe.unloaded == 1
    assert mpipe.offloaded == 2


def test_wan22_ti2v_native_flf2v_prepare_latents_and_fast_mp4(temp_studio_env, monkeypatch, tmp_path):
    import sys, types
    import numpy as np
    from PIL import Image
    from studio.engines.ffmpeg_utils import probe_media_info
    from studio.engines.video_engine import _attach_wan22_flf2v_support, _frames_to_mp4

    client, _ = temp_studio_env

    # 1) Controles UX anti-error de 2 fotos y píldoras de calidad LoRA en index.html
    html = client.get("/").text
    assert 'id="input-vid-quick-end-img"' in html
    assert 'id="vid-lora-quality-pills"' in html
    assert "lora-q-pill" in html

    # 2) Codificación rápida _frames_to_mp4 con lista PIL, np.ndarray 4D float32 [0,1] (Wan Diffusers) y NCHW [-1,1]
    frames = [Image.new("RGB", (320, 240), (20 * i, 80, 160)) for i in range(8)]
    mp4_out = tmp_path / "fast_pipe.mp4"
    info = _frames_to_mp4(frames=frames, output_mp4=mp4_out, fps=8)
    assert mp4_out.exists()
    assert info["has_video"] is True
    assert (info["width"], info["height"]) == (320, 240)

    # Caso exacto de Diffusers WanImageToVideoPipeline (`output.frames[0]` = np.ndarray 4D float32)
    np_frames_nhwc = np.linspace(0.0, 1.0, 8 * 240 * 320 * 3, dtype=np.float32).reshape((8, 240, 320, 3))
    mp4_np = tmp_path / "fast_wan_np4d.mp4"
    info_np = _frames_to_mp4(frames=np_frames_nhwc, output_mp4=mp4_np, fps=24)
    assert mp4_np.exists()
    assert info_np["has_video"] is True
    assert (info_np["width"], info_np["height"]) == (320, 240)

    np_frames_nchw = np.linspace(-1.0, 1.0, 8 * 3 * 240 * 320, dtype=np.float32).reshape((8, 3, 240, 320))
    mp4_nchw = tmp_path / "fast_nchw.mp4"
    info_nchw = _frames_to_mp4(frames=np_frames_nchw, output_mp4=mp4_nchw, fps=24)
    assert mp4_nchw.exists()
    assert (info_nchw["width"], info_nchw["height"]) == (320, 240)

    # Verificar mosaicos calibrados por VRAM + salvavidas automático anti-OOM en _decode
    from studio.engines.video_engine import _apply_best_offload

    class OutOfMemoryError(RuntimeError):
        pass

    class MockTileVAE:
        def __init__(self):
            self.tile_kwargs = None
            self.sliced = False
            self.cleared = 0
            self.fail_first = True

        def enable_slicing(self):
            self.sliced = True

        def clear_cache(self):
            self.cleared += 1

        def enable_tiling(
            self,
            tile_sample_min_height=None,
            tile_sample_min_width=None,
            tile_sample_stride_height=None,
            tile_sample_stride_width=None,
        ):
            self.tile_kwargs = (
                tile_sample_min_height,
                tile_sample_min_width,
                tile_sample_stride_height,
                tile_sample_stride_width,
            )

        def _decode(self, z):
            if self.fail_first:
                self.fail_first = False
                raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.51 GiB.")
            return ("decoded_ok", self.tile_kwargs)

    class MockOffloadPipe:
        def __init__(self):
            self.vae = MockTileVAE()

        def enable_model_cpu_offload(self):
            pass

    mop_a100 = MockOffloadPipe()
    _apply_best_offload(mop_a100, vram_total_gb=40.0, min_vram_gb=13.5)
    assert mop_a100.vae.tile_kwargs == (1280, 1280, 960, 960)

    mop_l4 = MockOffloadPipe()
    _apply_best_offload(mop_l4, vram_total_gb=22.0, min_vram_gb=13.5)
    assert mop_l4.vae.tile_kwargs == (448, 448, 320, 320)
    # Simular pico OOM inesperado: debe limpiar caché, bajar a 256×256 y completar _decode sin fallar
    res_dec = mop_l4.vae._decode("latents")
    assert res_dec == ("decoded_ok", (256, 256, 192, 192))
    assert mop_l4.vae.cleared == 1

    mop_t4 = MockOffloadPipe()
    _apply_best_offload(mop_t4, vram_total_gb=15.0, min_vram_gb=13.5)
    assert mop_t4.vae.tile_kwargs == (384, 384, 256, 256)

    # 3) Verificar condicionamiento nativo Inicio → Fin (`_attach_wan22_flf2v_support`) en Wan 2.2 TI2V-5B
    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=np.float32)

        @property
        def shape(self):
            return self.arr.shape

        def view(self, *s):
            return FakeTensor(self.arr.reshape(*s))

        def to(self, **_kw):
            return self

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.arr, axis=axis))

        def repeat(self, *reps):
            return FakeTensor(np.tile(self.arr, reps))

        def __sub__(self, other):
            return FakeTensor(self.arr - (other.arr if isinstance(other, FakeTensor) else other))

        def __mul__(self, other):
            return FakeTensor(self.arr * (other.arr if isinstance(other, FakeTensor) else other))

        def __rtruediv__(self, other):
            return FakeTensor(other / self.arr)

        def __getitem__(self, idx):
            return FakeTensor(self.arr[idx])

        def __setitem__(self, idx, val):
            self.arr[idx] = val.arr if isinstance(val, FakeTensor) else val

    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda x: FakeTensor(x)
    fake_torch.zeros = lambda shape, **_kw: FakeTensor(np.zeros(shape, dtype=np.float32))
    fake_torch.ones = lambda *shape, **_kw: FakeTensor(
        np.ones(shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape, dtype=np.float32)
    )
    fake_torch.randn = lambda shape, **_kw: FakeTensor(np.full(shape, 0.25, dtype=np.float32))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeDist:
        def __init__(self, val):
            self._val = val

        def mode(self):
            return self._val

    class FakeVAE:
        dtype = "float32"
        config = types.SimpleNamespace(z_dim=4, latents_mean=[0.0] * 4, latents_std=[1.0] * 4)

        def __init__(self):
            self.calls = []

        def encode(self, x):
            self.calls.append(x.shape)
            b, _c, nf, h, w = x.shape
            t_lat = 1 + (nf - 1) // 4
            # Codificar el valor medio del canal 0 para distinguir Inicio (10.0) vs Final (90.0)
            val = float(x.arr.mean())
            out = np.full((b, 4, t_lat, h // 16, w // 16), val, dtype=np.float32)
            # Simular diferencia entre frame 0 (causal inicial) y frame 1 (bloque causal estacionario 4-frame)
            if t_lat >= 2:
                out[:, :, 0, :, :] = val - 1.0
                out[:, :, 1, :, :] = val
            return types.SimpleNamespace(latent_dist=FakeDist(FakeTensor(out)))

    class FakeWan22Pipe:
        config = types.SimpleNamespace(expand_timesteps=True)
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 16

        def __init__(self):
            self.vae = FakeVAE()
            self.orig_called = 0

        def prepare_latents(self, *args, **kwargs):
            self.orig_called += 1
            return ("orig_i2v",)

    pipe = FakeWan22Pipe()
    _attach_wan22_flf2v_support(pipe)

    start_img = FakeTensor(np.full((1, 3, 64, 64), 10.0, dtype=np.float32))
    end_img = FakeTensor(np.full((1, 3, 64, 64), 90.0, dtype=np.float32))

    # Modo I2V normal (last_image=None) delega sin tocar nada
    res_i2v = pipe.prepare_latents(start_img, 1, 4, 64, 64, 21, None, "cpu", None, None, None)
    assert res_i2v == ("orig_i2v",)
    assert pipe.orig_called == 1

    # Modo Frame a Frame (last_image presente): ancla t=0 (10.0) y t=-1 (90.0 estacionario) con máscara 0 en ambos extremos
    latents, cond, mask = pipe.prepare_latents(start_img, 1, 4, 64, 64, 21, None, "cpu", None, None, end_img)
    assert latents.shape == (1, 4, 6, 4, 4)
    assert cond.shape == (1, 4, 6, 4, 4)
    assert mask.shape == (1, 1, 6, 4, 4)
    assert pipe.vae.calls == [(1, 3, 1, 64, 64), (1, 3, 5, 64, 64)]
    assert np.allclose(cond.arr[:, :, 0], 10.0)
    assert np.allclose(cond.arr[:, :, -1], 90.0)
    assert np.allclose(cond.arr[:, :, 1:-1], 0.0)
    assert np.allclose(mask.arr[:, :, 0], 0.0)
    assert np.allclose(mask.arr[:, :, -1], 0.0)
    assert np.allclose(mask.arr[:, :, 1:-1], 1.0)


def test_flf2v_dual_clip_fix_and_causal_lora_guardrails():
    import types
    import numpy as np
    from studio.catalog import DEFAULT_CATALOG
    from studio.engines.video_engine import _attach_dual_image_flf2v_fix

    # 1) Los LoRAs causales (Self-Forcing) siguen bloqueados en Frame a Frame, mientras los nuevos LoRAs bidireccionales 14B/1.3B sí permiten flf2v
    loras = {m["id"]: m for m in DEFAULT_CATALOG if m["category"] == "lora" and m.get("lora_target") == "video"}
    assert "flf2v" in loras["lora-wan22-ti2v-5b-turbo"]["incompatible_submodes"]
    assert loras["lora-wan22-ti2v-5b-turbo"]["causal_attention"] is True
    assert "flf2v" in loras["lora-wan22-ti2v-5b-fastwan"]["incompatible_submodes"]

    for bidir_id in (
        "lora-wan21-14b-fusionx-phantom",
        "lora-wan21-14b-fusionx-i2v",
        "lora-wan21-14b-lightx2v-i2v",
        "lora-wan21-14b-lightx2v-v2",
        "lora-wan21-1.3b-bidirect2",
        "lora-wan21-1.3b-rcm-turbo",
    ):
        assert bidir_id in loras
        assert loras[bidir_id]["causal_attention"] is False
        assert "flf2v" not in loras[bidir_id]["incompatible_submodes"]

    # 2) _attach_dual_image_flf2v_fix concatena (2, 257, 1280) -> (1, 514, 1280) cuando pos_embed_seq_len es None
    class FakeDualPipe:
        def __init__(self, pos_embed_seq_len=None):
            self.transformer = types.SimpleNamespace(
                config=types.SimpleNamespace(pos_embed_seq_len=pos_embed_seq_len)
            )

        def encode_image(self, image, device=None):
            n = len(image) if isinstance(image, (list, tuple)) else 1
            return np.ones((n, 257, 1280), dtype=np.float32)

    p_null = FakeDualPipe(pos_embed_seq_len=None)
    _attach_dual_image_flf2v_fix(p_null)
    assert p_null.encode_image("img1").shape == (1, 257, 1280)
    assert p_null.encode_image(["img1", "img2"]).shape == (1, 514, 1280)

    # Si el modelo ya tiene pos_embed_seq_len=514 (ej. Wan2.1-FLF2V-14B), mantiene (2, 257, 1280) para su view nativo
    p_514 = FakeDualPipe(pos_embed_seq_len=514)
    _attach_dual_image_flf2v_fix(p_514)
    assert p_514.encode_image(["img1", "img2"]).shape == (2, 257, 1280)


def test_quantized_14b_distilled_13b_models_and_fp8_loader(monkeypatch, tmp_path):
    import sys
    import types
    import numpy as np
    from studio.catalog import (
        DEFAULT_CATALOG,
        _DIFFUSERS_VIDEO_ALLOW,
        _matches_artifact_filters,
        evaluate_gpu_tier,
        resolve_gpu_profile,
    )
    from studio.engines.video_engine import (
        _load_wan_fp8_single_file_transformer,
        compute_official_video_params,
    )

    by_id = {m["id"]: m for m in DEFAULT_CATALOG}
    prof_l4 = resolve_gpu_profile("L4")
    prof_t4 = resolve_gpu_profile("T4")

    # 1) Los 4 nuevos modelos grandes cuantizados/destilados (13B–14B) caben con sticker Excelente en GPU L4 (22.5 GB) y soportan flf2v
    for mid in (
        "wan21-vace-14b-fp8",
        "wan21-flf2v-14b-fp8",
        "wan21-fun-inp-14b-fp8",
        "ltx-video-13b-distilled",
    ):
        assert mid in by_id
        defn = by_id[mid]
        assert "flf2v" in defn["video_modes"]
        tier_l4 = evaluate_gpu_tier(defn, prof_l4)
        assert tier_l4["compatible"] is True
        assert tier_l4["tier"] == "excellent"

    # En T4 (15 GB), los 14B FP8 marcan incompatible para que el selector salte automáticamente a VACE 1.3B
    assert evaluate_gpu_tier(by_id["wan21-vace-14b-fp8"], prof_t4)["compatible"] is False
    assert evaluate_gpu_tier(by_id["wan21-vace-1.3b"], prof_t4)["compatible"] is True

    # 2) Todos los compatible_models de cada LoRA de vídeo apuntan a IDs reales del catálogo
    for m in DEFAULT_CATALOG:
        if m["category"] == "lora" and m.get("lora_target") == "video":
            for target_id in m.get("compatible_models") or []:
                assert target_id in by_id, f"{m['id']} referencia modelo inexistente {target_id}"

    # 3) Filtros de descarga excluyen duplicados anidados de LTX-Video 0.9.8 13B (`vae/transformer/*`) y shards F32/BF16 en 14B FP8
    ltx13_ignore = by_id["ltx-video-13b-distilled"]["artifacts"][0]["ignore_patterns"]
    assert not _matches_artifact_filters(
        "vae/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
        _DIFFUSERS_VIDEO_ALLOW,
        ltx13_ignore,
        True,
    )
    assert _matches_artifact_filters(
        "transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
        _DIFFUSERS_VIDEO_ALLOW,
        ltx13_ignore,
        True,
    )

    flf14_ignore = by_id["wan21-flf2v-14b-fp8"]["artifacts"][0]["ignore_patterns"]
    assert not _matches_artifact_filters(
        "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
        _DIFFUSERS_VIDEO_ALLOW,
        flf14_ignore,
        True,
    )
    assert _matches_artifact_filters(
        "transformer/config.json",
        _DIFFUSERS_VIDEO_ALLOW,
        flf14_ignore,
        True,
    )
    assert _matches_artifact_filters(
        "image_encoder/model.safetensors",
        _DIFFUSERS_VIDEO_ALLOW,
        flf14_ignore,
        True,
    )

    # 4) Parámetros oficiales de LTX 13B Distilled (8 pasos @ CFG 1.0, regla 8k+1)
    eff_ltx13 = compute_official_video_params(
        defn=by_id["ltx-video-13b-distilled"],
        req_width=896,
        req_height=512,
        req_duration_sec=4.0,
        req_fps=24,
        req_steps=8,
        req_cfg=None,
    )
    assert eff_ltx13["steps"] == 8
    assert eff_ltx13["guidance_scale"] == 1.0
    assert eff_ltx13["num_frames"] == 97

    # 5) Verificar que `_load_wan_fp8_single_file_transformer` inyecta `img_emb.emb_pos` (514 tokens FLF2V-14B),
    #    desescala `.scale_weight` y convierte buffers excluidos del hook FP8 a bfloat16
    class FakeParam:
        def __init__(self, arr, dtype="float8_e4m3fn"):
            self.arr = np.array(arr, dtype=np.float32)
            self.dtype = dtype
            self.device = "cpu"
            self.shape = self.arr.shape

        @property
        def data(self):
            return self

        @data.setter
        def data(self, other):
            self.arr = other.arr.copy()
            self.dtype = other.dtype

        def to(self, dtype=None, device=None):
            target_dt = dtype if dtype is not None else self.dtype
            return FakeParam(self.arr.copy(), dtype=target_dt)

        def copy_(self, other):
            self.arr = other.arr.copy()
            self.dtype = other.dtype

        def __mul__(self, other):
            return FakeParam(self.arr * other.arr, dtype=self.dtype)

    fake_torch = types.ModuleType("torch")
    fake_torch.float8_e4m3fn = "float8_e4m3fn"
    fake_torch.float8_e5m2 = "float8_e5m2"
    fake_torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    saved_captured = {}

    def _fake_load_file(_path, device="cpu"):
        return {
            "blocks.0.self_attn.q.weight": FakeParam([[2.0, 4.0]], dtype="float8_e4m3fn"),
            "blocks.0.self_attn.q.scale_weight": FakeParam([0.5], dtype="bfloat16"),
            "img_emb.emb_pos": FakeParam(np.full((1, 514, 8), 7.5, dtype=np.float32), dtype="float8_e4m3fn"),
        }

    fake_st_torch.load_file = _fake_load_file
    fake_st.torch = fake_st_torch
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    class FakeSubMod:
        def __init__(self, hooked, params):
            self._params = params
            self._diffusers_hook = (
                types.SimpleNamespace(get_hook=lambda name: object() if (hooked and name == "layerwise_casting") else None)
                if hooked
                else None
            )

        def parameters(self, recurse=False):
            return self._params

        def buffers(self, recurse=False):
            return []

    class FakeTransformerCls:
        enable_layerwise_casting = lambda *args, **kwargs: None

        @classmethod
        def from_single_file(cls, sd, config=None, subfolder=None, torch_dtype=None, local_files_only=False,
                             low_cpu_mem_usage=False, device=None):
            assert torch_dtype == fake_torch.float8_e4m3fn
            assert low_cpu_mem_usage is True and device == "cpu" and local_files_only is True
            saved_captured["sd"] = dict(sd)
            saved_captured["subfolder"] = subfolder
            pos_p = FakeParam(np.zeros((1, 514, 8), dtype=np.float32), dtype="float8_e4m3fn")
            hooked_lin = FakeSubMod(True, [FakeParam([[1.0]], dtype="float8_e4m3fn")])
            unhooked_norm = FakeSubMod(False, [FakeParam([1.0], dtype="float8_e4m3fn")])

            class _Inst:
                def __init__(self):
                    self.lc_enabled = False
                    self.condition_embedder = types.SimpleNamespace(
                        image_embedder=types.SimpleNamespace(pos_embed=pos_p)
                    )
                    self._mods = [self, hooked_lin, unhooked_norm]

                def enable_layerwise_casting(self, storage_dtype=None, compute_dtype=None):
                    self.lc_enabled = (storage_dtype, compute_dtype)

                def modules(self):
                    return self._mods

                def parameters(self, recurse=False):
                    return [pos_p]

                def buffers(self, recurse=False):
                    return []

            return _Inst()

    sf_dummy = tmp_path / "wan_fp8.safetensors"
    sf_dummy.write_bytes(b"fp8")
    tr = _load_wan_fp8_single_file_transformer(FakeTransformerCls, sf_dummy, tmp_path)

    assert tr.lc_enabled == ("float8_e4m3fn", "bfloat16")
    # `.scale_weight` multiplicado: [[2.0, 4.0]] * 0.5 == [[1.0, 2.0]]
    assert np.allclose(saved_captured["sd"]["blocks.0.self_attn.q.weight"].arr, [[1.0, 2.0]])
    assert "img_emb.emb_pos" not in saved_captured["sd"]
    # `pos_embed` recibió los 514 tokens de `img_emb.emb_pos` en bfloat16
    assert np.allclose(tr.condition_embedder.image_embedder.pos_embed.arr, 7.5)
    assert tr.condition_embedder.image_embedder.pos_embed.dtype == "bfloat16"
    # Módulo CON hook mantiene float8_e4m3fn; módulo SIN hook sube a bfloat16
    assert tr._mods[1]._params[0].dtype == "float8_e4m3fn"
    assert tr._mods[2]._params[0].dtype == "bfloat16"


def test_mcp_server_auth_manuals_guardrails_and_colab_logs(temp_studio_env, monkeypatch, capsys, tmp_path):
    import base64
    import io
    import json
    from PIL import Image

    client, _ = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "st_mcp_secrettest123")

    # 1) Sin token devuelve 401; con Bearer token pasa
    r_unauth = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert r_unauth.status_code == 401

    hdrs = {"Authorization": "Bearer st_mcp_secrettest123"}

    # 2) Handshake MCP `initialize` + notificación `notifications/initialized` (202 Accepted)
    r_init = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "clientInfo": {"name": "claude-code-test", "version": "1.0.0"},
            },
        },
    )
    assert r_init.status_code == 200
    assert "Mcp-Session-Id" in r_init.headers
    init_body = r_init.json()["result"]
    assert init_body["protocolVersion"] == "2025-03-26"
    assert "studio_get_state_and_manual" in init_body["instructions"]
    assert "wan21-vace-14b-fp8" in init_body["instructions"]

    r_notif = client.post("/mcp", headers=hdrs, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r_notif.status_code == 202

    # 3) Manuales consultables (`resources/list` + `resources/read`) y plantillas (`prompts/list` + `prompts/get`)
    r_res = client.post("/mcp", headers=hdrs, json={"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
    uris = {x["uri"] for x in r_res.json()["result"]["resources"]}
    assert "studio://manual/quickstart" in uris
    assert "studio://manual/video-recipes" in uris
    assert "studio://state/live" in uris

    r_read = client.post(
        "/mcp",
        headers=hdrs,
        json={"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": "studio://manual/video-recipes"}},
    )
    assert "lora-wan21-14b-fusionx-phantom" in r_read.json()["result"]["contents"][0]["text"]

    r_prm = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/get",
            "params": {
                "name": "flf2v_production_workflow",
                "arguments": {"start_image": "a.png", "end_image": "b.png", "action_prompt": "salto mortal"},
            },
        },
    )
    assert "salto mortal" in r_prm.json()["result"]["messages"][0]["content"]["text"]

    # 4) Esquemas cerrados con `enum` y guardias anti-alucinación con instrucciones de corrección exacta
    r_tools = client.post("/mcp", headers=hdrs, json={"jsonrpc": "2.0", "id": 5, "method": "tools/list"})
    tools = {t["name"]: t for t in r_tools.json()["result"]["tools"]}
    assert len(tools) >= 13
    vid_schema = tools["studio_generate_video"]["inputSchema"]["properties"]
    assert vid_schema["video_submode"]["enum"] == ["flf2v", "i2v", "t2v"]
    assert "wan21-vace-14b-fp8" in vid_schema["model_id"]["enum"]
    assert "lora-wan21-14b-fusionx-phantom" in vid_schema["video_lora_id"]["enum"]

    # Subir dos fotogramas por MCP y probar guardia contra modelo incompatible en flf2v (`wan22-ti2v-5b`)
    def _png_b64(color):
        buf = io.BytesIO()
        Image.new("RGB", (320, 240), color).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    up1 = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "studio_upload_file_base64",
                "arguments": {"filename": "inicio.png", "data_base64": _png_b64((30, 90, 210))},
            },
        },
    ).json()["result"]
    assert up1["isError"] is False
    rel1 = json.loads(up1["content"][0]["text"])["asset"]["rel_path"]

    up2 = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "studio_upload_file_base64",
                "arguments": {"filename": "final.png", "data_base64": _png_b64((220, 80, 40))},
            },
        },
    ).json()["result"]
    rel2 = json.loads(up2["content"][0]["text"])["asset"]["rel_path"]

    # Guardia 1: modelo solo I2V/T2V (`wan22-ti2v-5b`) rechazado en `flf2v` con lista exacta de alternativas
    bad_mod = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "studio_generate_video",
                "arguments": {
                    "video_submode": "flf2v",
                    "mode": "ai",
                    "model_id": "wan22-ti2v-5b",
                    "prompt": "transición",
                    "image_rel": rel1,
                    "end_image_rel": rel2,
                },
            },
        },
    ).json()["result"]
    assert bad_mod["isError"] is True
    bad_mod_json = json.loads(bad_mod["content"][0]["text"])
    assert bad_mod_json["error_code"] == "SUBMODE_NOT_SUPPORTED_BY_MODEL"
    assert "wan21-vace-14b-fp8" in bad_mod_json["compatible_model_ids"]

    # 5) Ejecución real Frame a Frame FFmpeg vía MCP + verificación de logs en tiempo real hacia la celda de Colab
    ok_vid = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "studio_generate_video",
                "arguments": {
                    "video_submode": "flf2v",
                    "mode": "ffmpeg_motion",
                    "prompt": "transición suave",
                    "image_rel": rel1,
                    "end_image_rel": rel2,
                    "duration_sec": 2.0,
                    "wait": True,
                },
            },
        },
    ).json()["result"]
    assert ok_vid["isError"] is False
    ok_vid_data = json.loads(ok_vid["content"][0]["text"])
    assert ok_vid_data["status"] == "completed"
    assert ok_vid_data["result"]["rel_path"].startswith("outputs/videos/")

    captured_stdout = capsys.readouterr().out
    assert "[MCP] Agente conectado desde terminal: «claude-code-test 1.0.0»" in captured_stdout
    assert "[MCP → Studio IA] Agente ejecutó «studio_generate_video»" in captured_stdout

    # 6) Endpoints auxiliares `/mcp/bridge.py` y `/api/mcp/info`
    r_bridge = client.get("/mcp/bridge.py")
    assert r_bridge.status_code == 200
    assert "class StudioColabBridge" in r_bridge.text

    r_info = client.get("/api/mcp/info")
    assert r_info.status_code == 200
    assert r_info.json()["token_required"] is True
    assert "claude mcp add" in r_info.json()["commands"]["claude_code_stdio"]


def test_studio_mcp_local_bridge_upload_poll_and_download(temp_studio_env, monkeypatch, tmp_path):
    import io
    import json
    from PIL import Image
    from studio_mcp import StudioColabBridge

    client, _ = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "st_mcp_bridgetoken")

    # Conectar StudioColabBridge en memoria contra el TestClient FastAPI para probar subida local + polling + bajada a PC
    bridge = StudioColabBridge(
        base_url="http://testserver",
        token="st_mcp_bridgetoken",
        out_dir=str(tmp_path / "auto_out"),
    )

    def _mock_http_json(method, path, payload=None, timeout=65.0):
        resp = client.request(method, path, json=payload, headers=bridge._headers())
        assert resp.status_code < 400, resp.text
        return resp.json() if resp.text.strip() else {}

    def _mock_upload(local_path_str):
        p = tmp_path / local_path_str if not str(local_path_str).startswith("/") else Image.open(local_path_str)
        from pathlib import Path as _P

        fp = _P(local_path_str)
        with open(fp, "rb") as fh:
            resp = client.post(
                "/api/assets/upload",
                files={"file": (fp.name, fh, "image/png")},
                headers={"X-MCP-Client": "studio-mcp-local-bridge"},
            )
        assert resp.status_code == 200
        return resp.json()["asset"]

    def _mock_download(rel_path=None, save_to_local_path=None, export_all_zip=False):
        from pathlib import Path as _P

        remote = "/api/export/zip" if export_all_zip else f"/media/{rel_path.lstrip('/')}"
        resp = client.get(remote, headers=bridge._headers())
        assert resp.status_code == 200
        dest = _P(save_to_local_path) if save_to_local_path else (bridge.out_dir / _P(rel_path).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return str(dest)

    monkeypatch.setattr(bridge, "_http_json", _mock_http_json)
    monkeypatch.setattr(bridge, "upload_local_file", _mock_upload)
    monkeypatch.setattr(bridge, "download_to_local", _mock_download)

    # 1) `tools/list` en el puente añade propiedades locales (`local_image_path`, `local_end_image_path`, `save_to_local_path`) y `studio_download_asset_to_pc`
    tlist_resp = bridge.handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    b_tools = {t["name"]: t for t in tlist_resp["result"]["tools"]}
    assert "studio_download_asset_to_pc" in b_tools
    v_props = b_tools["studio_generate_video"]["inputSchema"]["properties"]
    assert "local_image_path" in v_props
    assert "local_end_image_path" in v_props
    assert "save_to_local_path" in v_props

    # 2) Crear dos imágenes en el "PC local", lanzar `studio_generate_video` por el puente y verificar el MP4 descargado en PC
    p_start = tmp_path / "pc_pose_1.png"
    p_end = tmp_path / "pc_pose_2.png"
    Image.new("RGB", (320, 240), (40, 180, 90)).save(p_start)
    Image.new("RGB", (320, 240), (190, 50, 140)).save(p_end)
    p_dest_mp4 = tmp_path / "descargas_pc" / "mi_clip_final.mp4"

    call_resp = bridge.handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "studio_generate_video",
                "arguments": {
                    "video_submode": "flf2v",
                    "mode": "ffmpeg_motion",
                    "prompt": "transición desde PC",
                    "local_image_path": str(p_start),
                    "local_end_image_path": str(p_end),
                    "duration_sec": 2.0,
                    "save_to_local_path": str(p_dest_mp4),
                },
            },
        }
    )
    assert call_resp["result"]["isError"] is False
    done_data = json.loads(call_resp["result"]["content"][0]["text"])
    assert done_data["ok"] is True
    assert done_data["result"]["saved_local_path"] == str(p_dest_mp4)
    assert p_dest_mp4.exists() and p_dest_mp4.stat().st_size > 1000


def test_termux_agent_guide_link_and_preconfigured_bridge(temp_studio_env, monkeypatch):
    client, _ = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "st_mcp_termux777")

    # Sin token, `/mcp/guia` rechaza con 401; con `?token=...` devuelve el manual completo preconfigurado
    assert client.get("/mcp/guia").status_code == 401
    r_guide = client.get("/mcp/guia?token=st_mcp_termux777")
    assert r_guide.status_code == 200
    assert "st_mcp_termux777" in r_guide.text
    assert "--cli studio_generate_video" in r_guide.text
    assert "wan21-vace-14b-fp8" in r_guide.text

    # `/mcp/bridge.py?token=...` incrusta URL y token automáticamente para uso directo en Termux
    r_br = client.get("/mcp/bridge.py?token=st_mcp_termux777")
    assert r_br.status_code == 200
    assert 'EMBEDDED_DEFAULT_TOKEN = "st_mcp_termux777"' in r_br.text
    assert 'EMBEDDED_DEFAULT_URL = "http://testserver"' in r_br.text


def test_session_log_export_md_and_txt_and_mcp_tool(temp_studio_env, monkeypatch, tmp_path):
    import json
    from studio.hardware import colab_console_log

    client, _ = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "st_mcp_logtest")
    colab_console_log("Evento de prueba para exportación completa de log", icon="🧪")

    # 1) Botón «📄 Imprimir Log» de la Web (`/api/logs/export?format=md` y `?format=txt`)
    r_md = client.get("/api/logs/export?format=md")
    assert r_md.status_code == 200
    assert "attachment; filename=\"studio_ia_log_" in r_md.headers.get("content-disposition", "")
    assert "Registro Completo de Sesión — Studio IA" in r_md.text
    assert "Servidor Studio IA encendido" in r_md.text
    assert "Evento de prueba para exportación completa de log" in r_md.text

    r_txt = client.get("/api/logs/export?format=txt")
    assert r_txt.status_code == 200
    assert r_txt.headers.get("content-disposition", "").endswith('.txt"')
    assert "REGISTRO COMPLETO DE SESIÓN — STUDIO IA" in r_txt.text

    # 2) Herramienta MCP `studio_export_session_log`
    hdrs = {"Authorization": "Bearer st_mcp_logtest"}
    r_mcp = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "studio_export_session_log", "arguments": {"format": "md"}},
        },
    ).json()["result"]
    assert r_mcp["isError"] is False
    log_payload = json.loads(r_mcp["content"][0]["text"])
    assert log_payload["rel_path"].startswith("outputs/logs/studio_ia_log_")
    assert client.get(log_payload["media_url"]).status_code == 200


def test_coherence_engine_fast_cv_and_hybrid_vlm_and_mcp_repair_loop(temp_studio_env, monkeypatch, tmp_path):
    import json
    import cv2
    import numpy as np
    from studio.engines import coherence_engine
    from studio.engines.coherence_engine import analyze_pair_fast_cv

    client, _ = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "st_mcp_cohtest")
    hdrs = {"Authorization": "Bearer st_mcp_cohtest"}

    # Crear 3 fotogramas sintéticos con textura real (ancla cálida de restaurante, candidato coherente y candidato con deriva fuerte)
    def _make_restaurant_scene(path, shirt_bgr, bg_bgr, size=(640, 480)):
        w, h = size
        canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)
        # Mesa, platos y detalles con bordes nítidos (para keypoints ORB y nitidez Laplaciana)
        cv2.rectangle(canvas, (0, int(h * 0.65)), (w, h), (45, 70, 115), -1)
        for x in range(40, w - 40, 60):
            cv2.line(canvas, (x, 20), (x, int(h * 0.62)), (80, 110, 150), 2)
        # Dos comensales
        cv2.circle(canvas, (int(w * 0.35), int(h * 0.36)), 38, (150, 180, 220), -1)
        cv2.rectangle(canvas, (int(w * 0.24), int(h * 0.44)), (int(w * 0.46), int(h * 0.66)), shirt_bgr, -1)
        cv2.circle(canvas, (int(w * 0.68), int(h * 0.37)), 36, (145, 175, 215), -1)
        cv2.rectangle(canvas, (int(w * 0.57), int(h * 0.45)), (int(w * 0.79), int(h * 0.66)), (60, 60, 180), -1)
        cv2.ellipse(canvas, (int(w * 0.50), int(h * 0.75)), (75, 24), 0, 0, 360, (235, 235, 240), -1)
        cv2.imwrite(str(path), canvas)

    p_anchor = tmp_path / "rest_ancla.png"
    p_good = tmp_path / "rest_coherente.png"
    p_bad = tmp_path / "rest_deriva.png"
    _make_restaurant_scene(p_anchor, shirt_bgr=(180, 95, 40), bg_bgr=(50, 95, 145), size=(640, 480))
    _make_restaurant_scene(p_good, shirt_bgr=(178, 92, 42), bg_bgr=(52, 96, 143), size=(640, 480))
    _make_restaurant_scene(p_bad, shirt_bgr=(30, 220, 40), bg_bgr=(210, 190, 30), size=(480, 640))  # Aspecto invertido + color cambiado

    cv_ok = analyze_pair_fast_cv(p_anchor, p_good)
    cv_bad = analyze_pair_fast_cv(p_anchor, p_bad)
    assert cv_ok["cv_overall_score"] >= 82.0
    assert cv_ok["aspect_ratio_compatible"] is True
    assert cv_bad["cv_overall_score"] < 55.0
    assert cv_bad["aspect_ratio_compatible"] is False
    assert len(cv_bad["cv_warnings"]) >= 1

    # Subir los 3 fotogramas a Studio IA
    rels = []
    for fp in (p_anchor, p_good, p_bad):
        with open(fp, "rb") as fh:
            up = client.post("/api/assets/upload", files={"file": (fp.name, fh, "image/png")}).json()
            rels.append(up["asset"]["rel_path"])
    rel_anchor, rel_good, rel_bad = rels

    # 1) Probar `studio_analyze_coherence` en modo `fast_cv` vía MCP
    r_fast = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "studio_analyze_coherence",
                "arguments": {
                    "anchor_image_rel": rel_anchor,
                    "candidate_image_rels": [rel_good, rel_bad],
                    "mode": "fast_cv",
                    "strictness_threshold": 78.0,
                },
            },
        },
    ).json()["result"]
    assert r_fast["isError"] is False
    fast_res = json.loads(r_fast["content"][0]["text"])["result"]
    assert fast_res["all_consistent"] is False
    assert fast_res["comparisons"][0]["verdict"] == "PASS"
    assert fast_res["comparisons"][1]["verdict"] == "NEEDS_REPAIR"
    rep_call = fast_res["comparisons"][1]["recommended_repair_tool_call"]
    assert rep_call["tool"] == "studio_generate_or_edit_image"
    assert rep_call["arguments"]["model_id"] == "qwen-image-edit-2511-pack"
    assert rep_call["arguments"]["base_image_rel"] == rel_anchor

    # 2) Probar Etapa 2 semántica simulando Qwen2.5-VL-3B detectando cambio de rostro en una escena donde el fondo sí coincidía
    monkeypatch.setattr(coherence_engine, "load_coherence_vlm_into_memory", lambda **_kw: {"model_id": "qwen25-vl-3b-coherence"})
    monkeypatch.setattr(
        coherence_engine,
        "analyze_pair_qwen_vl",
        lambda **_kw: {
            "same_people_count": True,
            "character_identity_score": 48.0,  # ¡Cambió la cara aunque OpenCV vio el mismo fondo!
            "clothing_hair_score": 52.0,
            "environment_props_score": 94.0,
            "flf2v_pose_transition_score": 85.0,
            "vlm_overall_score": 61.8,
            "anchor_characters_summary": "Hombre con gafas y camisa azul junto a mujer pelirroja.",
            "drifts_detected": ["El hombre perdió las gafas y ahora parece otra persona con camiseta verde."],
            "repair_edit_prompt": "Mantén exactamente al hombre con gafas y camisa azul y a la mujer pelirroja de la imagen original; cambia solo a que brindan con sus copas.",
            "raw_vlm_response": "{}",
            "elapsed_seconds": 0.1,
        },
    )
    r_vlm = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "studio_analyze_coherence",
                "arguments": {
                    "anchor_image_rel": rel_anchor,
                    "candidate_image_rels": [rel_good],
                    "mode": "vlm_qwen",
                    "scene_goal": "dos amigos comiendo en un restaurante",
                    "strictness_threshold": 78.0,
                },
            },
        },
    ).json()["result"]
    assert r_vlm["isError"] is False
    vlm_res = json.loads(r_vlm["content"][0]["text"])["result"]
    assert vlm_res["vlm_semantic_used"] is True
    assert vlm_res["all_consistent"] is False
    assert vlm_res["comparisons"][0]["verdict"] == "NEEDS_REPAIR"
    assert "gafas y camisa azul" in vlm_res["comparisons"][0]["recommended_repair_tool_call"]["arguments"]["prompt"]


def test_ffmpeg_montage_no_truncation_pingpong_and_ducking(temp_studio_env, tmp_path):
    """Verifica corrección de bugs FFmpeg: sin corte por `-shortest`, extensión ping-pong sin congelar y Auto-Ducking sin perder -6 dB."""
    import wave
    import numpy as np
    from PIL import Image
    from studio.engines.ffmpeg_utils import (
        create_motion_clip_from_image,
        probe_media_info,
        render_montage_mp4,
    )

    def _make_sine_wav(fp, dur_s, freq=330.0, amp=0.55):
        sr = 24000
        t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
        pcm = (amp * np.sin(2.0 * np.pi * freq * t) * 32767.0).astype(np.int16)
        with wave.open(str(fp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    img_p = tmp_path / "shot.png"
    Image.new("RGB", (640, 360), (140, 90, 60)).save(img_p)
    short_voice = tmp_path / "short_voice.wav"
    _make_sine_wav(short_voice, 1.1, freq=440.0)

    # 1) Un plano de 3.6s con voz corta de 1.1s NO debe truncarse a 1.1s (antes `-shortest` lo cortaba)
    clip_1 = tmp_path / "clip_full.mp4"
    create_motion_clip_from_image(
        image_path=img_p,
        output_mp4=clip_1,
        width=640,
        height=360,
        fps=24,
        duration_sec=3.6,
        motion="zoom_in",
        audio_path=short_voice,
    )
    info_1 = probe_media_info(clip_1)
    assert info_1["duration_seconds"] >= 3.45
    assert info_1["has_audio"] is True

    # 2) Vídeo corto (1.4s) con locución larga (3.2s) en montaje: se extiende por ping-pong fluido a ~3.2s + mezcla global con Auto-Ducking
    short_vid = tmp_path / "short_vid.mp4"
    create_motion_clip_from_image(
        image_path=img_p,
        output_mp4=short_vid,
        width=640,
        height=360,
        fps=24,
        duration_sec=1.4,
        motion="pan_right",
    )
    long_voice = tmp_path / "long_voice.wav"
    _make_sine_wav(long_voice, 3.2, freq=380.0)
    bgm_wav = tmp_path / "bgm.wav"
    _make_sine_wav(bgm_wav, 4.0, freq=180.0, amp=0.35)

    montage_out = tmp_path / "montage_ducked.mp4"
    m_res = render_montage_mp4(
        items=[
            {"path": str(short_vid), "audio_path": str(long_voice), "fit_to_audio": True},
            {"path": str(clip_1), "duration": 2.0},
        ],
        output_mp4=montage_out,
        width=640,
        height=360,
        fps=24,
        global_audio_path=bgm_wav,
        global_audio_volume=0.25,
        fade_seconds=0.15,
    )
    assert montage_out.exists()
    assert m_res["duration_seconds"] >= 5.0


def test_audio_studio_ambience_and_multitrack_mix_mcp(temp_studio_env, monkeypatch, tmp_path):
    """Prueba `studio_generate_music_or_sfx` (ambiente restaurante 0 GB) y `studio_mix_audio_tracks` con múltiples voces + Auto-Ducking."""
    import json
    import wave
    import numpy as np
    from PIL import Image
    from studio.engines.ffmpeg_utils import create_motion_clip_from_image

    client, data_dir = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "tok_audio_test")
    hdrs = {"Authorization": "Bearer tok_audio_test"}

    # 1) Generar ambiente estéreo cálido de restaurante vía MCP (0 GB CPU)
    r_amb = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "studio_generate_music_or_sfx",
                "arguments": {
                    "model_id": "procedural-ambience-cpu",
                    "preset": "restaurant_warm_murmur",
                    "duration_seconds": 4.0,
                },
            },
        },
    ).json()["result"]
    assert r_amb["isError"] is False
    amb_data = json.loads(r_amb["content"][0]["text"])["result"]
    assert amb_data["preset"] == "restaurant_warm_murmur"
    assert amb_data["duration_seconds"] >= 3.8
    amb_rel = amb_data["rel_path"]

    # Generar también pad musical cálido
    r_mus = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "studio_generate_music_or_sfx",
                "arguments": {
                    "model_id": "procedural-ambience-cpu",
                    "preset": "cinematic_warm_pad",
                    "duration_seconds": 4.0,
                },
            },
        },
    ).json()["result"]
    mus_rel = json.loads(r_mus["content"][0]["text"])["result"]["rel_path"]

    # 2) Subir un vídeo corto de 2.0s y 2 voces que llegan hasta 3.6s → `studio_mix_audio_tracks` extiende vídeo + aplica Auto-Ducking
    vid_fp = tmp_path / "base_scene.mp4"
    img_fp = tmp_path / "base_scene.png"
    Image.new("RGB", (640, 360), (95, 70, 55)).save(img_fp)
    create_motion_clip_from_image(img_fp, vid_fp, width=640, height=360, fps=24, duration_sec=2.0, motion="zoom_in")
    up_vid = client.post("/api/assets/upload", files={"file": ("base_scene.mp4", vid_fp.read_bytes(), "video/mp4")}).json()["asset"]["rel_path"]

    def _upload_voice(name: str, dur: float, freq: float) -> str:
        p = tmp_path / name
        sr = 24000
        t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        pcm = (0.65 * np.sin(2.0 * np.pi * freq * t) * 32767.0).astype(np.int16)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return client.post("/api/assets/upload", files={"file": (name, p.read_bytes(), "audio/wav")}).json()["asset"]["rel_path"]

    v1_rel = _upload_voice("voz_persona_izq.wav", 1.2, 320.0)
    v2_rel = _upload_voice("voz_persona_der.wav", 1.3, 460.0)

    r_mix = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 23,
            "method": "tools/call",
            "params": {
                "name": "studio_mix_audio_tracks",
                "arguments": {
                    "video_rel": up_vid,
                    "voice_tracks": [
                        {"audio_rel": v1_rel, "start_sec": 0.2, "volume": 1.0},
                        {"audio_rel": v2_rel, "start_sec": 1.9, "volume": 1.0},
                    ],
                    "music_rel": mus_rel,
                    "music_volume": 0.22,
                    "ambient_rel": amb_rel,
                    "ambient_volume": 0.18,
                    "auto_duck_bg": True,
                    "extend_video_to_audio": True,
                },
            },
        },
    ).json()["result"]
    assert r_mix["isError"] is False
    mix_res = json.loads(r_mix["content"][0]["text"])["result"]
    assert mix_res["auto_ducking_applied"] is True
    assert mix_res["video_extended"] is True
    assert mix_res["voice_tracks_count"] == 2
    assert mix_res["duration_seconds"] >= 3.2
    assert (data_dir / mix_res["rel_path"]).exists()


def test_lipsync_engine_multi_turn_dialogue_and_mcp_tool(temp_studio_env, monkeypatch, tmp_path):
    """Prueba el motor LipSync HD multi-personaje (`leftmost` / `rightmost` por turnos + XSeg + GFPGAN + Auto-Ducking) vía MCP."""
    import json
    import wave
    import numpy as np
    from PIL import Image, ImageDraw
    import studio.engines.lipsync_engine as ls_mod
    from studio.catalog import catalog_manager

    client, data_dir = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "tok_lipsync_test")
    hdrs = {"Authorization": "Bearer tok_lipsync_test"}

    # Verificar que la guía `/mcp/guia` expone la sección de LipSync HD y Auto-Ducking
    guia_txt = client.get("/mcp/guia?token=tok_lipsync_test").text
    assert "studio_lipsync_character" in guia_txt
    assert "studio_generate_music_or_sfx" in guia_txt
    assert "studio_mix_audio_tracks" in guia_txt

    # Simular paquete ONNX descargado y sesiones ligeras verificando exactamente las dimensiones de entrada/salida
    ls_dir = catalog_manager.model_dir("edtalk-256-lipsync-hd") / "onnx"
    ls_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("yunet_2023_mar.onnx", "edtalk_256.onnx", "xseg_1.onnx", "gfpgan_1.4.onnx"):
        (ls_dir / fn).write_bytes(b"onnx_placeholder")
    catalog_manager.marker_file("edtalk-256-lipsync-hd").write_text('{"ok": true}', encoding="utf-8")

    class _FakeInput:
        def __init__(self, name: str):
            self.name = name

    class _FakeEdtalk:
        def run(self, _, feeds):
            assert feeds["source"].shape == (1, 1, 80, 16)
            assert feeds["target"].shape == (1, 3, 256, 256)
            assert feeds["weight"].shape == (1,)
            # Modificar sutilmente la zona de la boca para verificar integración real de paste_back
            out = feeds["target"].copy()
            out[:, :, 165:210, 90:166] = np.clip(out[:, :, 165:210, 90:166] * 0.85 + 0.12, 0.0, 1.0)
            return [out.astype(np.float32)]

    class _FakeXseg:
        def get_inputs(self):
            return [_FakeInput("input")]

        def run(self, _, feeds):
            inp = feeds["input"]
            assert inp.shape == (1, 256, 256, 3)
            return [np.ones((1, 256, 256, 1), dtype=np.float32)]

    class _FakeGfpgan:
        def get_inputs(self):
            return [_FakeInput("input")]

        def run(self, _, feeds):
            inp = feeds["input"]
            assert inp.shape == (1, 3, 512, 512)
            return [inp.astype(np.float32)]

    fake_sessions = {
        "yunet": object(),
        "edtalk": _FakeEdtalk(),
        "xseg": _FakeXseg(),
        "gfpgan": _FakeGfpgan(),
    }
    monkeypatch.setattr(ls_mod.lipsync_engine, "_ensure_loaded", lambda model_id, job=None: fake_sessions)

    # Devolver 2 rostros claros en la escena (persona izquierda x~180 y persona derecha x~460)
    def _fake_detect_two_diners(_sess, frame_bgr, det_size=640, score_thresh=0.45):
        base_tpl = ls_mod.FFHQ_512_TEMPLATE
        left_kps = base_tpl * 140.0 + np.array([110.0, 95.0], dtype=np.float32)
        right_kps = base_tpl * 140.0 + np.array([390.0, 95.0], dtype=np.float32)
        return [
            {
                "bbox": np.array([110.0, 95.0, 250.0, 245.0], dtype=np.float32),
                "score": 0.96,
                "landmarks_5": left_kps,
                "area": 140.0 * 150.0,
                "center_x": 180.0,
                "center_y": 170.0,
            },
            {
                "bbox": np.array([390.0, 95.0, 530.0, 245.0], dtype=np.float32),
                "score": 0.95,
                "landmarks_5": right_kps,
                "area": 140.0 * 150.0,
                "center_x": 460.0,
                "center_y": 170.0,
            },
        ]

    monkeypatch.setattr(ls_mod, "detect_faces_yunet_onnx", _fake_detect_two_diners)

    # Crear imagen de dos comensales + 2 pistas de voz + ambiente restaurante
    scene_img = Image.new("RGB", (640, 360), (70, 50, 42))
    drw = ImageDraw.Draw(scene_img)
    drw.ellipse([125, 100, 235, 235], fill=(225, 185, 155))
    drw.ellipse([405, 100, 515, 235], fill=(215, 172, 145))
    scene_fp = tmp_path / "dos_comensales.png"
    scene_img.save(scene_fp)
    scene_rel = client.post("/api/assets/upload", files={"file": ("dos_comensales.png", scene_fp.read_bytes(), "image/png")}).json()["asset"]["rel_path"]

    def _mk_voice(fn: str, dur: float, f0: float) -> str:
        p = tmp_path / fn
        sr = 16000
        t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        sig = 0.5 * np.sin(2.0 * np.pi * f0 * t) * (0.55 + 0.45 * np.sin(2.0 * np.pi * 4.5 * t))
        pcm = (sig * 32767.0).astype(np.int16)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return client.post("/api/assets/upload", files={"file": (fn, p.read_bytes(), "audio/wav")}).json()["asset"]["rel_path"]

    v_left = _mk_voice("comensal_izq.wav", 0.7, 210.0)
    v_right = _mk_voice("comensal_der.wav", 0.7, 310.0)

    r_amb = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {
                "name": "studio_generate_music_or_sfx",
                "arguments": {"model_id": "procedural-ambience-cpu", "preset": "restaurant_warm_murmur", "duration_seconds": 3.0},
            },
        },
    ).json()["result"]
    amb_rel = json.loads(r_amb["content"][0]["text"])["result"]["rel_path"]

    r_ls = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {
                "name": "studio_lipsync_character",
                "arguments": {
                    "media_rel": scene_rel,
                    "dialogue_turns": [
                        {"voice_audio_rel": v_left, "target_face": "leftmost", "start_sec": 0.1},
                        {"voice_audio_rel": v_right, "target_face": "rightmost", "start_sec": 0.95},
                    ],
                    "ambient_rel": amb_rel,
                    "ambient_volume": 0.18,
                },
            },
        },
    ).json()["result"]
    assert r_ls["isError"] is False
    ls_data = json.loads(r_ls["content"][0]["text"])["result"]
    assert ls_data["turns_count"] == 2
    assert ls_data["frames_lipsynced"] > 10
    assert ls_data["auto_ducking_applied"] is True
    assert (data_dir / ls_data["rel_path"]).exists()


def test_higgsfield_mode_local_cinema_motion_cast_clipper_and_mcp(temp_studio_env) -> None:
    """Verifica el Modo Higgsfield 100 % local/Colab (0 € APIs): catálogo óptico, Casting Vault,
    curvas Crash Zoom / Handheld, Smart Clipper 9:16 centrado en sujeto, Auto-Director y MCP."""
    client, data_dir = temp_studio_env
    import json
    import cv2
    import numpy as np
    from studio.engines.ffmpeg_utils import probe_media_info

    # 1. Catálogo óptico y presets Higgsfield + HTML con conmutador Modo Normal <-> Modo Higgsfield
    html = client.get("/").text
    assert "btn-mode-normal" in html
    assert "btn-mode-higgsfield" in html
    assert "tab-higgsfield" in html

    cat = client.get("/api/higgsfield/catalog").json()
    assert cat["ok"] is True
    assert "imax_70mm_film" in cat["cameras"]
    assert "classic_anamorphic" in cat["lenses"]
    assert "crash_zoom" in cat["motions"]
    assert "dolly_zoom_vertigo" in cat["motions"]

    # 2. Subir un fotograma panorámico 16:9 con un sujeto claro en la zona derecha (X ~ 75%)
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas[:, :] = (25, 30, 42)
    # Dibujar silueta/rostro detallado a la derecha para comprobar que OpenCV centra el 9:16 ahí
    cv2.circle(canvas, (480, 130), 46, (215, 190, 170), -1)
    cv2.circle(canvas, (465, 120), 6, (20, 20, 20), -1)
    cv2.circle(canvas, (495, 120), 6, (20, 20, 20), -1)
    cv2.ellipse(canvas, (480, 145), (16, 8), 0, 0, 180, (140, 60, 60), 3)
    ok_enc, png_buf = cv2.imencode(".png", canvas)
    assert ok_enc
    up_img = client.post(
        "/api/assets/upload",
        files={"file": ("actor_derecha.png", png_buf.tobytes(), "image/png")},
    ).json()["asset"]

    # 3. Guardar personaje en el Casting Vault y verificar persistencia
    r_char = client.post(
        "/api/higgsfield/characters",
        json={
            "name": "Detective Marco",
            "traits": {
                "gender_age": "hombre de 38 años",
                "hair": "pelo corto oscuro",
                "outfit": "gabardina gris cine noir",
            },
            "anchor_image_rel": up_img["rel_path"],
            "voice_id": "es-ES-AlvaroNeural",
            "default_face_side": "rightmost",
        },
    ).json()
    assert r_char["ok"] is True
    char_id = r_char["character"]["id"]
    assert "gabardina gris" in r_char["character"]["appearance_prompt"]

    # 4. Generar clip Higgsfield Motion (Crash Zoom óptico en CPU)
    r_mot = client.post(
        "/api/higgsfield/motion-clip",
        json={
            "image_rel": up_img["rel_path"],
            "motion_id": "crash_zoom",
            "engine_mode": "ffmpeg_instant",
            "aspect_ratio": "16:9",
            "duration_seconds": 2.2,
        },
    ).json()
    j_mot = _wait_job(client, r_mot["job"]["id"])
    assert j_mot["status"] == "completed", j_mot
    mot_rel = j_mot["result"]["rel_path"]
    mot_path = data_dir / mot_rel
    assert mot_path.exists()
    mot_info = probe_media_info(mot_path)
    assert mot_info["has_video"] is True and mot_info["has_audio"] is True

    # 5. Smart Clipper 9:16 Reels: debe centrarse hacia la derecha (>58%) donde está el personaje
    r_clip = client.post(
        "/api/higgsfield/smart-clip",
        json={
            "video_rel": mot_rel,
            "target_aspect": "9:16",
            "start_sec": 0.0,
            "end_sec": 1.8,
        },
    ).json()
    j_clip = _wait_job(client, r_clip["job"]["id"])
    assert j_clip["status"] == "completed", j_clip
    clip_res = j_clip["result"]
    assert clip_res["width"] == 720 and clip_res["height"] == 1280
    assert clip_res["subject_center_x_pct"] > 58.0

    # 6. MCP: `studio_higgsfield_presets_and_cast` + `studio_higgsfield_director` (Auto-Director con ambiente)
    hdrs = {"X-Studio-MCP-Key": client.get("/api/mcp/info").json()["token"]}
    mcp_cast = client.post(
        "/mcp",
        headers=hdrs,
        json={"jsonrpc": "2.0", "id": 91, "method": "tools/call", "params": {"name": "studio_higgsfield_presets_and_cast", "arguments": {}}},
    ).json()["result"]
    assert mcp_cast["isError"] is False
    cast_payload = json.loads(mcp_cast["content"][0]["text"])
    assert any(c["id"] == char_id for c in cast_payload["characters"])

    mcp_dir = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 92,
            "method": "tools/call",
            "params": {
                "name": "studio_higgsfield_director",
                "arguments": {
                    "action": "scene_director",
                    "prompt": "Detective observando pistas bajo luces cálidas",
                    "camera_id": "imax_70mm_film",
                    "lens_id": "classic_anamorphic",
                    "motion_id": "handheld_docu",
                    "video_engine_mode": "ffmpeg_instant",
                    "character_ids": [char_id],
                    "duration_seconds": 2.4,
                    "ambience_preset": "rain_cozy",
                    "apply_lipsync": False,
                },
            },
        },
    ).json()["result"]
    assert mcp_dir["isError"] is False
    dir_res = json.loads(mcp_dir["content"][0]["text"])["result"]
    assert dir_res["mode"] == "higgsfield_director"
    assert (data_dir / dir_res["rel_path"]).exists()

    # Limpieza de ficha de prueba
    r_del = client.delete(f"/api/higgsfield/characters/{char_id}").json()
    assert r_del["ok"] is True


def test_lora_catalog_multi_model_compatibility_and_stacking(temp_studio_env) -> None:
    """Verifica los 35 LoRAs de HF (personaje, entorno, calidad, edición, cámara, estilo y velocidad) y el apilamiento Multi-LoRA."""
    from studio.catalog import catalog_manager
    from studio.engines.image_engine import _is_image_lora_compatible
    from studio.engines.video_engine import _sync_video_lora_on_handle

    client, _ = temp_studio_env
    all_defs = catalog_manager.get_all_definitions()
    loras = [d for d in all_defs if d.get("category") == "lora" and d.get("validation_state") != "candidate"]
    candidates = [d for d in all_defs if d.get("category") == "lora" and d.get("validation_state") == "candidate"]
    assert all(not d.get("compatible_models") for d in candidates)
    assert all(d.get("engine") == "unvalidated" for d in candidates)
    assert len(loras) >= 35

    valid_groups = {"character", "environment", "quality", "camera", "style", "speed"}
    adapter_names = set()
    for l_defn in loras:
        assert l_defn.get("lora_target") in ("image", "video"), l_defn["id"]
        assert l_defn.get("lora_group") in valid_groups, l_defn["id"]
        compat = l_defn.get("compatible_models") or []
        assert len(compat) >= 1, f"{l_defn['id']} sin compatible_models"
        for mid in compat:
            assert catalog_manager.get_definition(mid) is not None, f"Modelo desconocido {mid} en {l_defn['id']}"
        arts = l_defn.get("artifacts") or []
        assert len(arts) >= 1 and arts[0].get("repo_id") and (arts[0].get("filename") or arts[0].get("verify_file")), l_defn["id"]
        if not l_defn.get("is_custom"):
            assert arts[0].get("filename"), f"LoRA integrado sin filename exacto: {l_defn['id']}"
        adp = l_defn.get("adapter_name") or l_defn["id"]
        assert adp not in adapter_names, f"adapter_name duplicado: {adp}"
        adapter_names.add(adp)

    # Cobertura en los 3 modelos de imagen y todas las familias de vídeo principales
    for img_mid in ("qwen-image-edit-2511-pack", "qwen-image-2512-pack", "sdxl-turbo-fp16"):
        compat_img = [
            d for d in loras if d.get("lora_target") == "image" and _is_image_lora_compatible(d, img_mid)
        ]
        # T2I has one exact reviewed adapter. A saved custom candidate must not
        # count as a second compatible Qwen adapter just to satisfy coverage.
        minimum_reviewed = 1 if img_mid == "qwen-image-2512-pack" else 2
        assert len(compat_img) >= minimum_reviewed, f"Faltan LoRAs revisados para {img_mid}"

    for vid_mid in (
        "wan21-vace-14b-fp8",
        "wan21-flf2v-14b-fp8",
        "wan21-fun-inp-14b-fp8",
        "wan21-vace-1.3b",
        "skyreels-v2-i2v-1.3b",
        "wan22-ti2v-5b",
        "wan22-i2v-a14b",
        "ltx-video-13b-distilled",
        "ltx-2-19b-av",
    ):
        compat_vid = [
            d for d in loras if d.get("lora_target") == "video" and vid_mid in (d.get("compatible_models") or [])
        ]
        assert len(compat_vid) >= 1, f"Faltan LoRAs compatibles para {vid_mid}"

    # Endpoint REST /api/loras filtrado por modelo y HTML del Rack Multi-LoRA
    r_all = client.get("/api/loras").json()
    assert r_all["ok"] is True
    assert len(r_all["image_loras"]) >= 16
    assert len(r_all["video_loras"]) >= 19

    r_sdxl = client.get("/api/loras?model_id=sdxl-turbo-fp16").json()
    sdxl_ids = {x["id"] for x in r_sdxl["image_loras"]}
    assert "sdxl-lora-high-detail" in sdxl_ids
    assert "qwen-lora-skin-realism" not in sdxl_ids

    html = client.get("/").text
    assert 'id="img-lora-rack-box"' in html
    assert 'id="vid-extra-lora-box"' in html
    assert 'id="select-image-model"' in html

    # Prueba unitaria del apilamiento Multi-LoRA de vídeo en caliente sobre un pipeline simulado
    for test_lid in ("lora-wan21-14b-fusionx-phantom", "lora-wan21-14b-unianimate-char", "lora-wan21-14b-detailz"):
        ldef_t = catalog_manager.get_definition(test_lid)
        art_t = ldef_t["artifacts"][0]
        fake_lora_p = catalog_manager.model_dir(test_lid) / art_t["subdir"] / art_t["filename"]
        fake_lora_p.parent.mkdir(parents=True, exist_ok=True)
        fake_lora_p.write_bytes(b"fake_safetensors_weights")
        catalog_manager.marker_file(test_lid).write_text("{}", encoding="utf-8")

    class DummyPipe:
        def __init__(self):
            self.loaded = []
            self.active = None
            self.unloaded = 0

        def load_lora_weights(self, folder, weight_name=None, adapter_name=None):
            self.loaded.append((str(folder), weight_name, adapter_name))

        def set_adapters(self, names, adapter_weights=None):
            self.active = (list(names), list(adapter_weights or []))

        def unload_lora_weights(self):
            self.unloaded += 1
            self.active = None

    dummy_pipe = DummyPipe()
    handle = {
        "model_id": "wan21-vace-14b-fp8",
        "pipe": dummy_pipe,
        "active_lora_id": None,
        "active_lora_ids": [],
    }
    primary_ret = _sync_video_lora_on_handle(
        handle=handle,
        pipe=dummy_pipe,
        video_lora_id="lora-wan21-14b-fusionx-phantom",
        video_lora_scale=0.95,
        extra_video_lora_ids=["lora-wan21-14b-unianimate-char", "lora-wan21-14b-detailz"],
        extra_video_lora_scales={"lora-wan21-14b-detailz": 0.50},
        job=None,
    )
    assert primary_ret is not None and primary_ret["id"] == "lora-wan21-14b-fusionx-phantom"
    assert len(dummy_pipe.loaded) == 3
    assert dummy_pipe.active == (
        ["vid_speed_lora", "wan21_unianimate_char", "wan21_detailz"],
        [0.95, 0.65, 0.50],
    )
    assert list(handle["active_lora_signature"]) == [
        "lora-wan21-14b-fusionx-phantom",
        "lora-wan21-14b-unianimate-char",
        "lora-wan21-14b-detailz",
    ]


def test_mcp_agent_workflow_presets_and_anti_static_coherence(temp_studio_env, tmp_path: Path) -> None:
    """Verifica el Preset MCP `coherencia_personajes_entorno_avance` (andando por la calle hablando), `city_street_walk` y el detector `TOO_STATIC_NEEDS_ADVANCE`."""
    import json
    import numpy as np

    from studio.agent_presets import compile_agent_workflow_preset
    from studio.engines.audio_studio_engine import synthesize_procedural_ambience_wav
    from studio.engines.coherence_engine import run_coherence_analysis

    client, data_dir = temp_studio_env
    tok = client.get("/api/mcp/info").json()["token"]
    hdrs = {"Authorization": f"Bearer {tok}"} if tok else {}

    # 1) Guía /mcp/guia y endpoints REST de presets
    guia_txt = client.get("/mcp/guia", headers=hdrs).text
    assert "studio_get_workflow_preset" in guia_txt
    assert "coherencia_personajes_entorno_avance" in guia_txt
    assert "expect_spatial_progression" in guia_txt

    r_presets = client.get("/api/mcp/presets").json()
    assert r_presets["ok"] is True
    assert len(r_presets["presets"]) >= 6

    # 2) Compilación sobre GPU L4 simulada (22.5 GB VRAM) y llamada vía MCP JSON-RPC
    plan_l4 = compile_agent_workflow_preset(
        preset_id="andando_por_la_calle",
        scene_description="dos amigos andando por la calle hablando al atardecer",
        num_clips=2,
        gpu_override={"gpu_name": "NVIDIA L4", "vram_total_gb": 22.5, "has_gpu": True},
    )
    assert plan_l4["preset_id"] == "coherencia_personajes_entorno_avance"
    vstack = plan_l4["detected_gpu"]["selected_video_stack"]
    assert vstack["model_id"] == "wan21-vace-14b-fp8"
    assert "lora-wan21-14b-unianimate-char" in vstack["extra_video_lora_ids"]
    assert "lora-wan21-14b-moviigen-cinema" in vstack["extra_video_lora_ids"]

    steps = plan_l4["execution_plan"]
    # Paso 1 = Ancla, Paso 2 = Frame 2 con Next Scene, Paso 3 = Frame 3 con [FRAME_1, FRAME_2]
    assert steps[0]["tool"] == "studio_generate_or_edit_image"
    assert "qwen-lora-next-scene" in steps[1]["arguments"]["custom_lora_ids"]
    assert steps[1]["arguments"]["prompt"].startswith("Next Scene:")
    assert steps[2]["arguments"]["base_image_rel"] == "<FRAME_2_REL>"
    assert steps[2]["arguments"]["reference_rels"] == ["<FRAME_1_REL>", "<FRAME_2_REL>"]
    assert steps[3]["tool"] == "studio_analyze_coherence"
    assert steps[3]["arguments"]["expect_spatial_progression"] is True
    assert steps[4]["tool"] == "studio_generate_video"
    assert steps[4]["arguments"]["video_submode"] == "flf2v"
    assert "Steadicam" in steps[4]["arguments"]["prompt"]

    mcp_resp = client.post(
        "/mcp",
        headers=hdrs,
        json={
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "studio_get_workflow_preset",
                "arguments": {
                    "preset_id": "coherencia_personajes_entorno_avance",
                    "scene_description": "dos personas andando por la calle hablando",
                    "num_clips": 2,
                },
            },
        },
    ).json()["result"]
    assert mcp_resp["isError"] is False
    parsed_mcp = json.loads(mcp_resp["content"][0]["text"])
    assert parsed_mcp["preset_id"] == "coherencia_personajes_entorno_avance"

    # 3) Ambiente urbano `city_street_walk` (0 GB VRAM)
    street_wav = tmp_path / "street.wav"
    amb_info = synthesize_procedural_ambience_wav(street_wav, preset="city_street_walk", duration_seconds=3.0)
    assert street_wav.exists() and street_wav.stat().st_size > 10_000
    assert amb_info["preset"] == "city_street_walk"

    # 4) Detector anti-estático `TOO_STATIC_NEEDS_ADVANCE` vs fotograma con avance real `PASS`
    up_dir = data_dir / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    # Crear un fotograma ancla con textura nítida de calle y personajes simulados
    arr_a = np.zeros((360, 640, 3), dtype=np.uint8)
    arr_a[:, :] = [160, 140, 120]
    arr_a[60:260, 100:220] = [50, 90, 170]  # edificio / tienda posición A
    arr_a[110:340, 250:330] = [40, 60, 95]  # personaje izq
    arr_a[115:340, 360:440] = [185, 165, 130]  # personaje dcha
    # Añadir detalle de alta frecuencia para superar el filtro Laplaciano de nitidez
    rng = np.random.default_rng(7)
    arr_a = np.clip(arr_a.astype(np.int16) + rng.integers(-18, 19, size=arr_a.shape), 0, 255).astype(np.uint8)
    p_anchor = up_dir / "walk_anchor.png"
    Image.fromarray(arr_a).save(p_anchor)

    # Fotograma congelado (idéntico al ancla -> debe saltar TOO_STATIC_NEEDS_ADVANCE)
    p_static = up_dir / "walk_static.png"
    Image.fromarray(arr_a).save(p_static)

    res_static = run_coherence_analysis(
        anchor_path=p_anchor,
        candidate_paths=[p_static],
        scene_goal="dos personas andando por la calle",
        mode="fast_cv",
        expect_spatial_progression=True,
    )
    assert res_static["all_consistent"] is False
    assert res_static["comparisons"][0]["verdict"] == "TOO_STATIC_NEEDS_ADVANCE"
    rep_args = res_static["comparisons"][0]["recommended_repair_tool_call"]["arguments"]
    assert "qwen-lora-next-scene" in rep_args["custom_lora_ids"]

    # Fotograma con avance real (la tienda de fondo se desplaza en perspectiva y cambia la zancada, misma paleta y luz)
    arr_adv = np.zeros((360, 640, 3), dtype=np.uint8)
    arr_adv[:, :] = [160, 140, 120]
    arr_adv[80:230, 20:115] = [50, 90, 170]  # tienda desplazada hacia atrás/izquierda
    arr_adv[70:250, 480:600] = [65, 115, 85]  # árbol nuevo entrando en plano
    arr_adv[105:345, 235:320] = [40, 60, 95]  # personaje izq avanzando
    arr_adv[110:345, 370:455] = [185, 165, 130]  # personaje dcha avanzando
    arr_adv = np.clip(arr_adv.astype(np.int16) + rng.integers(-18, 19, size=arr_adv.shape), 0, 255).astype(np.uint8)
    p_adv = up_dir / "walk_advanced.png"
    Image.fromarray(arr_adv).save(p_adv)

    res_adv = run_coherence_analysis(
        anchor_path=p_anchor,
        candidate_paths=[p_adv],
        scene_goal="dos personas andando por la calle",
        mode="fast_cv",
        expect_spatial_progression=True,
    )
    assert res_adv["all_consistent"] is True
    assert res_adv["comparisons"][0]["verdict"] == "PASS"

















# Regression coverage for the Colab L4 / Python 3.13 failures. These tests use
# small parameter/pipeline doubles: no GPU weights or network downloads needed.


def test_colab_parameter_metadata_patch_preserves_peft_and_bnb_contracts(monkeypatch):
    import inspect
    import sys
    import types
    from contextlib import contextmanager
    from studio.compat_patches import apply_runtime_compatibility_patches

    class Tensor:
        __slots__ = ("value", "device", "requires_grad", "__dict__")

        def __new__(cls, data=None, requires_grad=True):
            obj = object.__new__(cls)
            obj.value = data.value if isinstance(data, Tensor) else data
            obj.device = data.device if isinstance(data, Tensor) else "cpu"
            obj.requires_grad = requires_grad
            return obj

        def to(self, device):
            result = Tensor(self.value, self.requires_grad)
            result.device = device
            return result

    class Parameter(Tensor):
        def __new__(cls, data=None, requires_grad=True):
            return Tensor.__new__(cls, data, requires_grad)

    class Params4bit(Parameter):
        # Deliberately bypass Parameter.__new__, as bitsandbytes does.
        def __new__(cls, data=None, requires_grad=False, quant_state=None, blocksize=64, quant_type="nf4"):
            obj = Tensor.__new__(cls, data, requires_grad)
            obj.quant_state = quant_state
            obj.blocksize = blocksize
            obj.quant_type = quant_type
            return obj

    class Int8Params(Parameter):
        def __new__(cls, data=None, requires_grad=False, has_fp16_weights=False, CB=None, SCB=None):
            obj = Tensor.__new__(cls, data, requires_grad)
            obj.has_fp16_weights, obj.CB, obj.SCB = has_fp16_weights, CB, SCB
            return obj

    class Module:
        def __init__(self):
            self._parameters, self._buffers = {}, {}

        def register_parameter(self, name, param):
            self._parameters[name] = param

        def register_buffer(self, name, buffer, persistent=True):
            self._buffers[name] = buffer

    calls, skip = [], {"value": False}

    @contextmanager
    def init_empty_weights(include_buffers=None):
        calls.append(include_buffers)
        old_register, old_buffer = Module.register_parameter, Module.register_buffer

        def register_empty(module, name, param):
            old_register(module, name, param)
            if param is not None and not skip["value"]:
                kwargs = dict(param.__dict__, requires_grad=param.requires_grad)
                module._parameters[name] = type(param)(param.to("meta"), **kwargs)

        def register_buffer(module, name, buffer, persistent=True):
            old_buffer(module, name, buffer, persistent)
            if buffer is not None:
                module._buffers[name] = buffer.to("meta")

        Module.register_parameter = register_empty
        if include_buffers:
            Module.register_buffer = register_buffer
        try:
            yield "original-context"
        finally:
            Module.register_parameter, Module.register_buffer = old_register, old_buffer

    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Parameter=Parameter, Module=Module)
    bnb = types.ModuleType("bitsandbytes")
    bnb.nn = types.SimpleNamespace(Params4bit=Params4bit, Int8Params=Int8Params)
    peft, utils, integrations = (types.ModuleType(n) for n in ("peft", "peft.utils", "peft.utils.integrations"))
    peft.utils, utils.integrations = utils, integrations
    integrations.init_empty_weights = init_empty_weights
    for name, mod in (("torch", torch), ("bitsandbytes", bnb), ("peft", peft), ("peft.utils", utils), ("peft.utils.integrations", integrations)):
        monkeypatch.setitem(sys.modules, name, mod)

    # Reproduce the original error before applying compatibility.
    p = Parameter(Tensor(7), requires_grad=False)
    p._is_hf_initialized = True
    with pytest.raises(TypeError, match="_is_hf_initialized"):
        type(p)(p.to("meta"), **dict(p.__dict__, requires_grad=p.requires_grad))

    original_signature = inspect.signature(Params4bit.__new__)
    apply_runtime_compatibility_patches()
    patched_new, patched_context = Parameter.__new__, integrations.init_empty_weights
    apply_runtime_compatibility_patches()
    assert Parameter.__new__ is patched_new
    assert integrations.init_empty_weights is patched_context
    assert inspect.signature(Params4bit.__new__) == original_signature

    quant_state, cb, scb = object(), object(), object()
    p4 = Params4bit(Tensor(4), quant_state=quant_state, blocksize=128, _is_hf_initialized=True, _hf_private="kept")
    p8 = Int8Params(Tensor(8), has_fp16_weights=True, CB=cb, SCB=scb, _is_hf_initialized=True)
    layer = Module()
    original_register, original_buffer = Module.register_parameter, Module.register_buffer
    with pytest.raises(RuntimeError, match="restore hooks"):
        with integrations.init_empty_weights(include_buffers=True) as ctx:
            assert ctx == "original-context"
            for name, param in (("weight", p), ("q4", p4), ("q8", p8)):
                layer.register_parameter(name, param)
                rebuilt = layer._parameters[name]
                assert rebuilt.device == "meta"
                assert rebuilt.requires_grad is False
                assert rebuilt._is_hf_initialized is True
            assert layer._parameters["q4"].quant_state is quant_state
            assert layer._parameters["q4"].blocksize == 128
            assert layer._parameters["q4"]._hf_private == "kept"
            assert layer._parameters["q8"].SCB is scb
            assert layer._parameters["q8"].CB is cb
            assert layer._parameters["q8"].has_fp16_weights is True
            layer.register_buffer("running", Tensor(2), persistent=False)
            assert layer._buffers["running"].device == "meta"
            with integrations.init_empty_weights(include_buffers=False):
                skip["value"] = True
                layer.register_parameter("skip", p)
                skip["value"] = False
                assert layer._parameters["skip"] is p
            raise RuntimeError("restore hooks")
    assert calls == [True, False]
    assert Module.register_parameter is original_register
    assert Module.register_buffer is original_buffer
    # Real argument mistakes must still fail; only private metadata is tolerated.
    with pytest.raises(TypeError, match="requires_grads"):
        Parameter(Tensor(1), requires_grads=False)


def test_job_deduplication_is_atomic_and_does_not_conflate_downloads(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from studio.jobs import JobManager, model_prepare_job_key

    manager = JobManager(max_workers=2)
    started, release = threading.Event(), threading.Event()
    executed = []

    def work(job):
        executed.append(job.id)
        started.set()
        assert release.wait(5)
        return {"model_id": "qwen", "loaded": True}

    try:
        first = manager.submit("load_model", "prepare qwen", work, dedup_key="prepare:qwen")
        assert started.wait(2)
        with ThreadPoolExecutor(max_workers=8) as callers:
            same = list(callers.map(lambda _: manager.submit("load_model", "duplicate", work, dedup_key="prepare:qwen"), range(16)))
        assert {job.id for job in same} == {first.id}
        assert executed == [first.id]
        downloaded = manager.submit("download", "disk only", lambda _: {"downloaded": True}, dedup_key="download:qwen")
        assert downloaded.id != first.id
        assert first.to_dict()["dedup_key"] == "prepare:qwen"
        # Different adapter requirements are not silently swallowed by a base-only prepare.
        assert model_prepare_job_key("qwen", ["b", "a", "b"]) == model_prepare_job_key("qwen", ["a", "b"])
        assert model_prepare_job_key("qwen", ["a"]) != model_prepare_job_key("qwen")
        release.set()
        deadline = time.monotonic() + 2
        while first.status not in ("completed", "failed") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert first.status == "completed"
        after = manager.submit("load_model", "new prepare", lambda _: {"loaded": True}, dedup_key="prepare:qwen")
        assert after.id != first.id
    finally:
        release.set()
        manager._executor.shutdown(wait=True)


def test_cancelled_queued_deduplicated_job_does_not_execute():
    import threading
    from studio.jobs import JobManager

    manager = JobManager(max_workers=1)
    started, release = threading.Event(), threading.Event()
    called = []
    try:
        manager.submit("block", "block", lambda _: (started.set(), release.wait(5)) and {})
        assert started.wait(2)
        queued = manager.submit("load_model", "queued", lambda _: called.append(True) or {}, dedup_key="prepare:qwen")
        duplicate = manager.submit("load_model", "same", lambda _: {}, dedup_key="prepare:qwen")
        assert queued.id == duplicate.id
        manager.cancel(queued.id)
    finally:
        release.set()
        manager._executor.shutdown(wait=True)
    assert queued.status == "cancelled"
    assert called == []


def test_same_model_download_lock_prevents_hf_cache_deletion_race(temp_studio_env, monkeypatch, tmp_path):
    import threading
    import types
    from concurrent.futures import ThreadPoolExecutor
    import studio.catalog as catalog
    from studio.jobs import JobRecord

    manager = catalog.CatalogManager()
    mid = "download-race-regression"
    definition = {"id": mid, "name": "Race", "category": "lora", "estimated_gb": 0.001,
                  "artifacts": [{"repo_id": "test/race", "subdir": "lora", "filename": "weights.safetensors", "verify_file": "weights.safetensors"}]}
    monkeypatch.setattr(manager, "get_definition", lambda _mid: definition)
    monkeypatch.setattr(manager, "model_dir", lambda _mid: tmp_path / _mid)
    monkeypatch.setattr(manager, "get_hf_token", lambda: None)
    monkeypatch.setattr(manager, "check_disk_space", lambda *a, **k: {"ok": True})
    metadata_calls, downloads = [], []
    in_download, release, second_started = threading.Event(), threading.Event(), threading.Event()

    class API:
        def __init__(self, **kw):
            pass

        def model_info(self, **kw):
            metadata_calls.append(kw)
            return types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename="weights.safetensors", size=5)])

    def download(filename, local_dir, **kw):
        downloads.append(filename)
        cache = Path(local_dir) / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True)
        (cache / "weights.lock").write_text("locked")
        in_download.set()
        assert release.wait(5)
        assert cache.exists()
        target = Path(local_dir) / filename
        target.write_bytes(b"12345")
        return str(target)

    monkeypatch.setattr(catalog, "HfApi", API)
    monkeypatch.setattr(catalog, "hf_hub_download", download)
    jobs = [JobRecord("download", f"race {i}") for i in range(2)]

    def second():
        second_started.set()
        return manager.download_model(mid, jobs[1])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(manager.download_model, mid, jobs[0])
        try:
            assert in_download.wait(2)
            other = pool.submit(second)
            assert second_started.wait(2)
        finally:
            release.set()
        assert first.result(timeout=3)["model_id"] == mid
        assert other.result(timeout=3)["model_id"] == mid
    assert downloads == ["weights.safetensors"]
    assert len(metadata_calls) == 1
    assert manager.is_downloaded(mid)
    assert any("ya está verificado" in line["message"] for line in jobs[1].logs)
    # A marker is never enough if its verified file is empty.
    (manager.model_dir(mid) / "lora" / "weights.safetensors").write_bytes(b"")
    assert not manager.is_downloaded(mid)


@pytest.fixture
def mock_colab_image_runtime(temp_studio_env, monkeypatch, tmp_path):
    import sys
    import types
    from studio.engines import image_engine as engine
    from studio.memory_manager import RuntimeMemoryManager
    import studio.memory_manager as memory_module

    monkeypatch.setattr(memory_module, "flush_ram_and_vram", lambda: None)
    manager = RuntimeMemoryManager()
    monkeypatch.setattr(engine, "memory_manager", manager)
    monkeypatch.setattr(engine, "flush_ram_and_vram", lambda: None)
    monkeypatch.setattr(engine, "apply_runtime_compatibility_patches", lambda: None)
    monkeypatch.setattr(engine, "query_gpu_info", lambda: {"vram_used_gb": 1.0})
    monkeypatch.setattr(engine, "run_with_live_memory_progress", lambda fn, **kw: fn())
    monkeypatch.setattr(engine, "IMAGES_OUT_DIR", tmp_path / "images")
    events, pipes, downloaded = [], [], set()
    cat = engine.catalog_manager
    monkeypatch.setattr(cat, "model_dir", lambda mid: tmp_path / "models" / mid)
    monkeypatch.setattr(cat, "is_downloaded", lambda mid: mid in downloaded)

    def download(mid, job, **kw):
        events.append(("download", mid))
        downloaded.add(mid)
        for art in cat.get_definition(mid)["artifacts"]:
            fp = cat.model_dir(mid) / art["subdir"] / (art.get("filename") or art.get("verify_file"))
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"weights")
        return {"model_id": mid}

    monkeypatch.setattr(cat, "download_model", download)

    class Generator:
        def __init__(self, device):
            assert device == "cuda"

        def manual_seed(self, seed):
            return self

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.Generator = Generator
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))

    class Pipeline:
        def __init__(self):
            self.offloaded, self.enabled, self.weights = False, True, ([], [])
            self.loaded, self.prompts = [], []

        def load_lora_weights(self, folder, weight_name=None, adapter_name=None):
            if self.offloaded:
                raise NotImplementedError("Cannot copy out of meta tensor; no data!")
            self.loaded.append(adapter_name)
            events.append(("lora", adapter_name))

        def set_adapters(self, names, adapter_weights):
            self.weights = (list(names), list(adapter_weights))
            events.append(("activate", tuple(names)))

        def enable_model_cpu_offload(self):
            self.offloaded = True
            events.append(("offload", len(pipes)))

        def enable_lora(self):
            self.enabled = True

        def disable_lora(self):
            self.enabled = False

        def __call__(self, prompt, image=None, true_cfg_scale=1.0, negative_prompt=None, **kw):
            self.prompts.append(prompt)
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), "navy")])

    def instantiate(*a):
        events.append(("instantiate", len(pipes)))
        pipe = Pipeline()
        pipes.append(pipe)
        return pipe

    monkeypatch.setattr(engine, "_instantiate_image_pipeline", instantiate)
    return engine, manager, events, pipes, downloaded


def test_colab_image_autoloads_and_preloads_zero_scale_loras_before_offload(mock_colab_image_runtime):
    from studio.agent_presets import compile_agent_workflow_preset

    engine, manager, events, pipes, downloaded = mock_colab_image_runtime
    plan = compile_agent_workflow_preset(gpu_override={"has_gpu": True, "vram_total_gb": 22.0})
    first, second = (plan["execution_plan"][i]["arguments"] for i in (0, 1))
    assert manager.get_slot("image") is None
    result = engine.generate_or_edit_image(prompt=first["prompt"], model_id=first["model_id"],
        custom_lora_ids=first["custom_lora_ids"], lora_scales=first["lora_scales"], use_ffmpeg_bt709=False, seed=10)
    pipe = pipes[0]
    assert result["model_id"] == "qwen-image-edit-2511-pack"
    assert len(pipe.loaded) == 5  # Lightning + all four creative adapters.
    offload = next(i for i, event in enumerate(events) if event[0] == "offload")
    assert all(i < offload for i, event in enumerate(events) if event[0] == "lora")
    instantiate = next(i for i, event in enumerate(events) if event[0] == "instantiate")
    assert all(i < instantiate for i, event in enumerate(events) if event[0] == "download")
    assert len(pipe.weights[0]) == 3  # Lightning, skin, flatlog; inactive adapters add no trigger.
    assert "Next Scene" not in pipe.prompts[-1]
    assert "AnyLight" not in pipe.prompts[-1]
    count = len(events)
    engine.generate_or_edit_image(prompt=second["prompt"], custom_lora_ids=second["custom_lora_ids"],
        lora_scales=second["lora_scales"], use_ffmpeg_bt709=False, seed=11)
    assert len(pipes) == 1
    assert not any(event[0] in ("download", "lora", "offload", "instantiate") for event in events[count:])
    expected_names = {engine.catalog_manager.get_definition(mid)["adapter_name"] for mid in second["custom_lora_ids"]}
    assert set(pipe.weights[0]) == {"lightning"} | expected_names


def test_colab_new_image_adapter_reloads_once_and_zero_scale_can_be_reenabled(mock_colab_image_runtime):
    engine, manager, events, pipes, _ = mock_colab_image_runtime
    engine.generate_or_edit_image(prompt="two friends on a street", use_ffmpeg_bt709=False, seed=20)
    old_handle = manager.get_slot("image")["handle"]
    lid = "qwen-lora-next-scene"
    engine.generate_or_edit_image(prompt="two friends walking", custom_lora_ids=[lid], lora_scales={lid: 0.0},
        use_lightning_lora=False, use_ffmpeg_bt709=False, seed=21)
    assert len(pipes) == 2
    assert old_handle == {}  # No retained slot bundle while allocating a replacement.
    assert not pipes[-1].enabled
    count = len(events)
    engine.generate_or_edit_image(prompt="two friends walking further", custom_lora_ids=[lid], lora_scales={lid: 0.85},
        use_lightning_lora=False, use_ffmpeg_bt709=False, seed=22)
    assert len(pipes) == 2
    assert pipes[-1].enabled
    assert pipes[-1].weights[1] == [0.85]
    assert not any(event[0] in ("download", "lora", "offload") for event in events[count:])


def test_colab_image_respects_disabled_lora_downloads_and_model_selection(mock_colab_image_runtime):
    engine, manager, events, pipes, downloaded = mock_colab_image_runtime
    lid = "qwen-lora-next-scene"
    engine.generate_or_edit_image(prompt="a street", custom_lora_ids=[lid], auto_download_loras=False,
        use_ffmpeg_bt709=False, seed=30)
    assert lid not in downloaded
    engine.generate_or_edit_image(prompt="a forest", model_id="qwen-image-2512-pack", use_ffmpeg_bt709=False, seed=31)
    assert manager.get_slot("image")["model_id"] == "qwen-image-2512-pack"
    assert len(pipes) == 2


def test_gpu_model_jobs_and_unloads_do_not_overlap(temp_studio_env, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import studio.memory_manager as mm
    from studio.engines import image_engine, video_engine

    monkeypatch.setattr(mm, "flush_ram_and_vram", lambda: None)
    manager = mm.RuntimeMemoryManager()
    manager.register_loaded("image", "qwen", "Qwen", {"pipe": object()})
    active, release, attempted, video_entered = (threading.Event() for _ in range(4))
    order = []

    @mm.serialized_gpu_operation
    def infer():
        active.set()
        assert release.wait(5)
        assert manager.get_slot("image") is not None
        order.append("image_done")

    def prepare_video(**kw):
        video_entered.set()
        order.append("video_load")
        manager.unload_all()
        return {"model_id": "wan"}

    monkeypatch.setattr(video_engine, "_load_video_model_into_memory_locked", prepare_video)

    def competing_prepare():
        attempted.set()
        return video_engine.load_video_model_into_memory(model_id="wan")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(infer)
        try:
            assert active.wait(2)
            second = pool.submit(competing_prepare)
            assert attempted.wait(2)
            assert not video_entered.wait(0.05)
        finally:
            release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert order == ["image_done", "video_load"]
    assert manager.get_slot("image") is None


def test_mcp_and_rest_prepare_reuse_active_job_but_download_is_distinct(temp_studio_env, monkeypatch):
    import json
    import threading
    import app as app_module
    import studio.mcp_server as mcp
    from studio.jobs import JobManager

    client, _ = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "")
    manager = JobManager(max_workers=2)
    monkeypatch.setattr(app_module, "job_manager", manager)
    monkeypatch.setattr(mcp, "job_manager", manager)
    monkeypatch.setattr(mcp.catalog_manager, "is_downloaded", lambda _: True)
    started, release = threading.Event(), threading.Event()
    calls = []

    def load(**kw):
        calls.append(kw["model_id"])
        started.set()
        assert release.wait(5)
        return {"model_id": kw["model_id"], "loaded": True}

    monkeypatch.setattr(mcp, "load_image_model_into_memory", load)
    monkeypatch.setattr(app_module, "load_image_model_into_memory", load)
    monkeypatch.setattr(mcp.catalog_manager, "download_model", lambda mid, job: {"model_id": mid, "downloaded": True})
    mid = "qwen-image-edit-2511-pack"
    try:
        first = json.loads(mcp._execute_mcp_tool("studio_prepare_model", {"model_id": mid, "wait": False})["content"][0]["text"])
        assert started.wait(2)
        again = client.post("/api/memory/load", json={"model_id": mid, "auto_download": True}).json()
        assert first["job_id"] == again["job"]["id"]
        assert "wait_seconds=80" in first["how_to_wait"]
        download = json.loads(mcp._execute_mcp_tool("studio_prepare_model", {"model_id": mid, "load_into_vram": False, "wait": False})["content"][0]["text"])
        assert download["job_id"] != first["job_id"]
        assert calls == [mid]
    finally:
        release.set()
        manager._executor.shutdown(wait=True)


@pytest.mark.parametrize("duration,expected", [("un minuto", 12), ("60s", 12), ("30 segundos", 6), ("medio minuto", 6), ("0,5 minutos", 6), ("one minute", 12), ("20s", 4)])
def test_workflow_duration_inference_and_visual_prompt_cleanup(duration, expected):
    from studio.agent_presets import compile_agent_workflow_preset, _sanitize_visual_scene_description

    raw = f"Dos amigos con chaquetas rojas caminan por una avenida con coches. Crear los diálogos, que sea un clip de {duration}, utiliza LoRAs rápido, con lipsync."
    plan = compile_agent_workflow_preset(scene_description=raw, gpu_override={"has_gpu": True, "vram_total_gb": 22.0})
    summary = plan["planned_duration_summary"]
    assert summary["num_clips"] == expected
    assert summary["num_keyframes"] == expected + 1
    assert summary["estimated_total_seconds"] == expected * 5.0
    prompt = plan["execution_plan"][0]["arguments"]["prompt"].lower()
    assert "chaquetas rojas" in prompt and "avenida con coches" in prompt
    assert all(fragment not in prompt for fragment in ("crear los diálogos", "clip de", "loras", "lipsync", duration.lower()))
    assert _sanitize_visual_scene_description("Crear los diálogos, utiliza LoRAs rápido, con lipsync") == ""
    # Non-default explicit clip counts are not overwritten by a duration in the scene.
    one = compile_agent_workflow_preset(scene_description=raw, num_clips=1, gpu_override={"has_gpu": False})
    assert one["planned_duration_summary"]["num_clips"] == 1
    voices = one["execution_plan"][-2]["parallel_or_sequential_calls"]
    assert len([v for v in voices if v["tool"] == "studio_generate_speech"]) == 2


def test_full_minute_plan_has_strict_phase_barriers_valid_tools_and_all_dialogue(temp_studio_env):
    from studio.agent_presets import compile_agent_workflow_preset
    from studio.mcp_server import _build_mcp_tools_schema

    dialogue = [f"Réplica número {i}." for i in range(15)]
    plan = compile_agent_workflow_preset(scene_description="Dos personas andando por la calle hablando; un clip de un minuto.",
        dialogue_lines=dialogue, gpu_override={"has_gpu": True, "vram_total_gb": 22.0})
    steps = plan["execution_plan"]
    assert all(step.get("tool") == "studio_generate_or_edit_image" for step in steps[:13])
    assert steps[13]["tool"] == "studio_analyze_coherence"
    assert steps[13]["required_result"] == {"all_consistent": True, "vlm_semantic_used": True}
    assert steps[13]["save_output_as"] == "COHERENCE_AUDIT_RESULT"
    video = steps[14:26]
    assert len(video) == 12
    assert {step["arguments"]["model_id"] for step in video} == {"wan21-vace-14b-fp8"}
    assert all(step["batch"] == "3_all_video_clips" and "COHERENCE_AUDIT_RESULT" in step["requires_outputs"] for step in video)
    assert plan["execution_policy"]["allow_automatic_fallback_downloads"] is False
    assert plan["execution_policy"]["gpu_calls_parallel"] is False
    assert "POSITION M" in steps[12]["arguments"]["prompt"]
    assert "48 meters beyond" in steps[12]["arguments"]["prompt"]
    first = steps[0]["arguments"]
    assert first["lora_scales"]["qwen-lora-next-scene"] == 0.0
    assert first["lora_scales"]["qwen-lora-anylight-env"] == 0.0
    assert len(first["custom_lora_ids"]) == 4
    voices = steps[-2]["parallel_or_sequential_calls"]
    assert [v["arguments"]["text"] for v in voices if v["tool"] == "studio_generate_speech"] == dialogue
    post = steps[-1]["recommended_calls"]
    turns = [turn for call in post if call["tool"] == "studio_lipsync_character" for turn in call["arguments"]["dialogue_turns"]]
    assert len(turns) == len(dialogue)
    assert {t["voice_audio_rel"] for t in turns} == {f"<VOICE_{i}_REL>" for i in range(1, 16)}
    montage = post[-1]["arguments"]
    assert montage["global_audio_rel"] == "<STREET_AMBIENCE_REL>"
    assert montage["fade_seconds"] == 0.0
    assert len(montage["items"]) == 12

    schemas = {tool["name"]: tool["inputSchema"] for tool in _build_mcp_tools_schema()}
    calls = []
    for step in steps:
        if "tool" in step:
            calls.append(step)
        calls.extend(step.get("parallel_or_sequential_calls", []))
        calls.extend(step.get("recommended_calls", []))
    for call in calls:
        schema, args = schemas[call["tool"]], call["arguments"]
        assert set(args) <= set(schema["properties"]), call
        assert set(schema.get("required", [])) <= set(args), call


def test_preset_duration_inference_is_consistent_in_rest_and_mcp(temp_studio_env, monkeypatch):
    import json
    import studio.mcp_server as mcp

    client, _ = temp_studio_env
    monkeypatch.setenv("STUDIO_MCP_TOKEN", "")
    description = "Dos amigos caminan por la avenida; que sea un clip de un minuto, crear los diálogos, utiliza LoRAs rápido."
    rest = client.post("/api/mcp/presets/compile", json={"scene_description": description}).json()
    remote = json.loads(mcp._execute_mcp_tool("studio_get_workflow_preset", {"scene_description": description})["content"][0]["text"])
    assert rest["planned_duration_summary"] == remote["planned_duration_summary"]
    assert rest["planned_duration_summary"]["num_clips"] == 12
    assert "clip de" not in rest["execution_plan"][0]["arguments"]["prompt"]


def test_mcp_failed_job_instructs_stop_without_panic_downloading(monkeypatch):
    import studio.mcp_server as mcp
    from studio.jobs import JobRecord
    from types import SimpleNamespace

    failed = JobRecord("load_model", "Qwen")
    failed.status = "failed"
    failed.error = "Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'"
    monkeypatch.setattr(mcp, "job_manager", SimpleNamespace(get=lambda _: failed))
    result = mcp._wait_for_job_bounded(failed.id)
    assert result["ok"] is False
    assert result["error_code"] == "JOB_FAILED"
    assert "_is_hf_initialized" in result["error"]
    assert "PROHIBIDO" in result["anti_panic_rule"]
    assert "NO se arregla" in result["anti_panic_rule"]


def test_mcp_montage_preserves_full_video_duration_and_zero_fade(temp_studio_env, monkeypatch):
    import json
    import studio.mcp_server as mcp
    from studio.config import UPLOADS_DIR
    from studio.jobs import JobManager

    manager = JobManager()
    monkeypatch.setattr(mcp, "job_manager", manager)
    clip = UPLOADS_DIR / "mock_six_second_clip.mp4"
    clip.write_bytes(b"mock video")
    captured = {}

    def render(**kw):
        captured.update(kw)
        return {"duration_seconds": 6.0, "width": 640, "height": 360, "size_bytes": 100}

    monkeypatch.setattr(mcp, "render_montage_mp4", render)
    try:
        result = json.loads(mcp._execute_mcp_tool("studio_render_montage", {"items": [{"media_rel": f"uploads/{clip.name}"}],
            "fade_seconds": 0.0, "global_audio_volume": 0.0})["content"][0]["text"])
    finally:
        manager._executor.shutdown(wait=True)
    assert result["status"] == "completed"
    assert captured["items"][0]["duration"] == 0.0  # FFmpeg probes native duration; not the former hard-coded 4s.
    assert captured["fade_seconds"] == 0.0
    assert captured["global_audio_volume"] == 0.0


def test_video_batch_releases_image_and_vision_slots_before_loading(mock_colab_image_runtime, monkeypatch):
    from studio.engines import video_engine

    image, manager, events, pipes, downloaded = mock_colab_image_runtime
    image_handle = {"pipe": object()}
    vision_handle = {"model": object(), "processor": object()}
    manager.register_loaded("image", "qwen-image-edit-2511-pack", "Qwen", image_handle)
    manager.register_loaded("vision", "qwen25-vl-3b-coherence", "Inspector", vision_handle)
    monkeypatch.setattr(video_engine, "memory_manager", manager)
    monkeypatch.setattr(video_engine, "apply_runtime_compatibility_patches", lambda: None)
    monkeypatch.setattr(video_engine, "flush_ram_and_vram", lambda: None)
    monkeypatch.setattr(video_engine, "query_gpu_info", lambda: {"vram_used_gb": 1.0, "vram_total_gb": 22.0})
    monkeypatch.setattr(video_engine, "_apply_best_offload", lambda *a, **k: None)
    # This is a simulated high-RAM Colab, not the sandbox's 4 GiB host.
    import studio.memory_manager as mm
    from types import SimpleNamespace
    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: SimpleNamespace(total=64 * 1024**3, available=52 * 1024**3, used=12 * 1024**3))

    def instantiate(*a, **k):
        assert manager.get_slot("image") is None
        assert manager.get_slot("vision") is None
        assert image_handle == {} and vision_handle == {}
        return {"pipe": object()}

    monkeypatch.setattr(video_engine, "_instantiate_video_pipeline", instantiate)
    result = video_engine.load_video_model_into_memory("wan21-vace-1.3b", auto_download=True)
    assert result["model_id"] == "wan21-vace-1.3b"
    assert manager.get_slot("video")["model_id"] == result["model_id"]
    assert "wan21-vace-1.3b" in downloaded


def test_unload_pipeline_bundles_releases_objects_not_shared_catalogue(monkeypatch):
    import gc
    import weakref
    import studio.memory_manager as mm

    class Pipe:
        def maybe_free_model_hooks(self):
            pass

    pipe = Pipe()
    ref = weakref.ref(pipe)
    definition = {"id": "qwen", "name": "Do not clear the shared catalogue"}
    handle = {"pipe": pipe, "all_components": {"model": pipe}, "defn": definition}
    manager = mm.RuntimeMemoryManager()
    monkeypatch.setattr(mm, "flush_ram_and_vram", lambda: None)
    manager.register_loaded("image", "qwen", "Qwen", handle)
    del pipe
    manager.unload_by_model_id("qwen")
    gc.collect()
    assert ref() is None
    assert handle == {}
    assert definition["id"] == "qwen"


def test_mcp_sanitizes_raw_image_and_video_prompts_before_inference(temp_studio_env, monkeypatch):
    import json
    import studio.mcp_server as mcp
    from studio.jobs import JobManager

    manager = JobManager()
    monkeypatch.setattr(mcp, "job_manager", manager)
    prompts = []

    def generate(**kw):
        prompts.append(kw["prompt"])
        return {"model_id": kw["model_id"]}

    monkeypatch.setattr(mcp, "generate_or_edit_image", generate)
    monkeypatch.setattr(mcp, "generate_video_clip", generate)
    monkeypatch.setattr(mcp.catalog_manager, "check_disk_space", lambda *a, **k: {"ok": True})
    # This test replaces inference to inspect prompts; give that fake runtime a
    # complete fitting device instead of bypassing the real resource guard.
    from studio.model_compatibility import assessment_context
    snapshot = {"gpu": {"has_gpu": True, "gpu_tier": "L4", "gpu_name": "L4 (test)",
                         "vram_total_gb": 22.0, "vram_free_gb": 21.0, "bf16_supported": True},
                "ram": {"total_gb": 64.0, "available_gb": 60.0}, "disk": {"free_gb": 256.0}}
    monkeypatch.setattr(mcp, "assessment_context", lambda **_kw: assessment_context(snapshot=snapshot))
    raw = "Two friends walk down a tree-lined avenue. Crear los diálogos, un clip de 60s, utiliza LoRAs rápido, con lipsync."
    try:
        for tool, args in (
            ("studio_generate_or_edit_image", {"model_id": "qwen-image-edit-2511-pack", "prompt": raw}),
            ("studio_generate_video", {"model_id": "wan21-vace-1.3b", "video_submode": "t2v", "prompt": raw}),
        ):
            result = json.loads(mcp._execute_mcp_tool(tool, args)["content"][0]["text"])
            assert result["status"] == "completed", result
    finally:
        manager._executor.shutdown(wait=True)
    assert len(prompts) == 2
    assert all("tree-lined avenue" in prompt for prompt in prompts)
    assert all(not any(meta in prompt.lower() for meta in ("diálogos", "clip de", "60s", "loras", "lipsync")) for prompt in prompts)
