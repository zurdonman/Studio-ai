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



