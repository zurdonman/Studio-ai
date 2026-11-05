import base64
import io
import os
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


@pytest.fixture(scope="module")
def temp_studio_env(tmp_path_factory: pytest.TempPathFactory):
    tmp_root = tmp_path_factory.mktemp("studio_test_data")
    os.environ["STUDIO_DATA_DIR"] = str(tmp_root)
    os.environ["STUDIO_FORCE_LOCAL"] = "1"
    from app import app

    client = TestClient(app)
    return client, tmp_root


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


def test_gpu_filter_stickers_artifact_patterns_and_hf_search(temp_studio_env):
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

    # 7) Wan 2.2 TI2V-5B ahora soporta nativamente flf2v; modelos exclusivos como cogvideox-5b-i2v devuelven HTTP 400
    assert "flf2v" in wan22["video_modes"]
    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (20, 40, 60)).save(buf, format="PNG")
    raw = buf.getvalue()
    u1 = client.post("/api/assets/upload", files={"file": ("a.png", raw, "image/png")}).json()["asset"]
    u2 = client.post("/api/assets/upload", files={"file": ("b.png", raw, "image/png")}).json()["asset"]
    r_bad_flf = client.post(
        "/api/video/generate",
        json={
            "mode": "ai",
            "video_submode": "flf2v",
            "model_id": "cogvideox-5b-i2v",
            "image_rel": u1["rel_path"],
            "end_image_rel": u2["rel_path"],
            "prompt": "transición",
        },
    )
    assert r_bad_flf.status_code == 400
    assert "Wan 2.2 TI2V-5B" in r_bad_flf.json()["detail"]


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

    # 2) Codificación rápida _frames_to_mp4 por pipe RGB24 directo
    frames = [Image.new("RGB", (320, 240), (20 * i, 80, 160)) for i in range(8)]
    mp4_out = tmp_path / "fast_pipe.mp4"
    info = _frames_to_mp4(frames=frames, output_mp4=mp4_out, fps=8)
    assert mp4_out.exists()
    assert info["has_video"] is True
    assert (info["width"], info["height"]) == (320, 240)

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








