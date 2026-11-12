"""Deterministic metadata/recipe/oracle tests: no Hub weights, CUDA or real network."""
import copy
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolated_studio(temp_studio_env):
    # Internal modules are imported below, after the shared data root is set.
    return temp_studio_env


class CatalogStub:
    def __init__(self, definitions, downloaded=()):
        self.definitions = copy.deepcopy(definitions)
        self.downloaded = set(downloaded)

    def get_all_definitions(self):
        return self.definitions

    def get_definition(self, model_id):
        return next((d for d in self.definitions if d["id"] == model_id), None)

    def is_downloaded(self, model_id):
        definition = self.get_definition(model_id)
        return bool(definition and (definition.get("builtin_no_download") or model_id in self.downloaded))

    def get_hf_token(self):
        return "hf_not_for_public_metadata"

    def download_model(self, *_args, **_kwargs):
        raise AssertionError("Discovery/oracle must NEVER download weights")


@pytest.fixture
def catalog():
    from studio.catalog import DEFAULT_CATALOG
    return CatalogStub(DEFAULT_CATALOG)


def context(*, gpu="L4", ram=53, ram_free=48, disk=108, vram_free=21, profile="auto"):
    from studio.model_compatibility import assessment_context
    from studio.model_recipes import ENGINE_CONTRACTS
    packages = {"torch", "diffusers", "transformers", "accelerate", "safetensors", "peft", "bitsandbytes"}
    for contract in ENGINE_CONTRACTS.values():
        packages.update(contract.get("packages", []))
    versions = {package: "999.0" for package in packages}
    runtime = {"versions": versions, "diffusers_classes": sorted({c for e in ENGINE_CONTRACTS.values() for c in e.get("classes", [])}),
               "ffmpeg": True, "ffprobe": False, "executables": {"espeak-ng": True}}
    vram = {"CPU": 0, "T4": 15, "L4": 22.03, "A100": 80}[gpu]
    snapshot = {"gpu": {"has_gpu": gpu != "CPU", "gpu_name": "NVIDIA " + gpu, "gpu_tier": gpu,
                        "vram_total_gb": vram, "vram_free_gb": 0 if gpu == "CPU" else vram_free,
                        "bf16_supported": gpu in ("L4", "A100"), "gpu_count": 1},
                "ram": {"total_gb": ram, "available_gb": ram_free}, "disk": {"free_gb": disk}}
    return assessment_context(snapshot=snapshot, gpu_profile=profile, runtime=runtime)


def raw(repo="example/model", *, pipeline="text-to-image", downloads=10, revision="a" * 40):
    return {"id": repo, "sha": revision, "pipeline_tag": pipeline, "downloads": downloads, "likes": None,
            "gated": False, "cardData": {"license": "apache-2.0"}, "tags": [], "siblings": [], "library_name": "diffusers"}


class ProviderStub:
    def __init__(self, items=None):
        self.items = items or []
        self.search_calls = []
        self.inspect_calls = []
        self.error = None
        self.inspection = None

    def search_page(self, query, category, sort, cursors=None):
        self.search_calls.append((query, category, sort, cursors))
        if self.error:
            raise self.error
        return {"items": copy.deepcopy(self.items), "cursors": {}}

    def inspect(self, repo_id):
        self.inspect_calls.append(repo_id)
        if self.error:
            raise self.error
        return copy.deepcopy(self.inspection or raw(repo_id))


def store(tmp_path, catalog, provider=None, clock=None, *, research=False):
    from studio.model_discovery import ModelDiscovery
    service = ModelDiscovery(path=tmp_path / "index.json", catalog=catalog, provider=provider or ProviderStub(), clock=clock or (lambda: 1800000000.0))
    if not research:
        service.research = {"families": []}
    return service


def test_research_sources_are_pinned_and_not_claimed_runnable():
    from studio.model_discovery import RESEARCH_FILE, clean_repo_id
    from studio.model_recipes import EXTRA_IMAGE_MODELS
    data = json.loads(RESEARCH_FILE.read_text())
    families = data["families"]
    refs = {repo for family in families for repo in family["repo_ids"]}
    assert len(families) == 73 and len(refs) == 243
    for family in families:
        assert family["source_url"].startswith("https://github.com/")
        assert "/blob/v0.42.1/" in family["source_url"] or re.search(r"/blob/[a-f0-9]{40}/", family["source_url"])
    for repo in refs:
        assert clean_repo_id(repo) == repo
    assert len(EXTRA_IMAGE_MODELS) == 8
    assert all(d["gpu_tested"] is False for d in EXTRA_IMAGE_MODELS)
    assert all(d["id"] != "lumina-image-2" for d in EXTRA_IMAGE_MODELS)
    assert "Lumina" in (Path(__file__).parents[1] / "docs/MODEL_RESEARCH.md").read_text()


@pytest.mark.parametrize("model_id", ["sdxl-base-1.0", "flux2-klein-4b", "z-image-turbo", "pixart-sigma-xl-1024", "sana-600m-1024", "sana-1600m-1024", "sana-sprint-600m"])
def test_capacity_fit_is_not_downloaded_or_gpu_validation(catalog, model_id):
    from studio.model_compatibility import assess_definition
    assessment = assess_definition(catalog.get_definition(model_id), context(), catalog=catalog)
    assert assessment["compatible"] is True
    assert assessment["runnable_now"] is False
    assert assessment["weights_prepared"] is False
    assert assessment["gpu_tested"] is False
    assert assessment["resources"]["disk"]["download_budget_gb"] > catalog.get_definition(model_id)["estimated_gb"]


def test_precision_simulation_ram_disk_pressure_and_runtime_axes(catalog):
    from studio.model_compatibility import assess_definition
    from studio.catalog import resolve_gpu_profile
    d = catalog.get_definition("sana-600m-1024")
    catalog.downloaded.add(d["id"])
    assert assess_definition(d, context(), catalog=catalog)["runnable_now"] is True
    assert assess_definition(d, context(gpu="T4"), catalog=catalog)["compatible"] is False
    sim = assess_definition(d, context(gpu="CPU", profile="L4"), catalog=catalog)
    assert sim["compatible"] is True and sim["runnable_now"] is False and sim["is_simulated"] is True
    assert resolve_gpu_profile("A100_80")["tier"] == "A100"
    assert assess_definition(d, context(gpu="CPU", profile="L4", ram=4, ram_free=3), catalog=catalog)["compatible"] is False
    assert assess_definition(d, context(disk=0.2), catalog=catalog)["resources"]["disk"]["status"] == "insufficient"
    pressured = assess_definition(d, context(vram_free=1, ram_free=1), catalog=catalog)
    assert pressured["compatible"] is True and pressured["runnable_now"] is False
    assert pressured["resources"]["vram"]["needs_unload"] is True
    assert pressured["resources"]["ram"]["needs_release"] is True
    unknown = assess_definition(d, context(vram_free=None), catalog=catalog)
    assert unknown["runnable_now"] is False
    ctx = context()
    ctx["runtime"]["diffusers_classes"].remove("SanaPipeline")
    runtime_bad = assess_definition(d, ctx, catalog=catalog)
    assert runtime_bad["compatible"] is True and runtime_bad["runnable_now"] is False
    assert "SanaPipeline" in runtime_bad["runtime"]["missing_classes"]
    big = assess_definition(catalog.get_definition("wan22-i2v-a14b"), context(gpu="A100", disk=512), catalog=catalog)
    assert big["resources"]["ram"]["required_gb"] == 128
    assert big["compatible"] is False


def test_lora_requires_both_weights_and_peft_and_uses_the_full_base_budget(catalog):
    from studio.model_compatibility import assess_definition
    adapter = catalog.get_definition("sdxl-lora-high-detail")
    base_id = adapter["parent_model"]
    catalog.downloaded.add(adapter["id"])
    result = assess_definition(adapter, context(), catalog=catalog)
    assert result["assessed_base_model_id"] == base_id
    assert result["compatible"] is True and result["weights_prepared"] is False and result["runnable_now"] is False
    assert result["resources"]["vram"]["required_gb"] > catalog.get_definition(base_id)["min_vram_gb"]
    assert result["resources"]["disk"]["download_budget_gb"] >= catalog.get_definition(base_id)["estimated_gb"]
    catalog.downloaded.add(base_id)
    ctx = context()
    ctx["runtime"]["versions"]["peft"] = None
    result = assess_definition(adapter, ctx, catalog=catalog)
    assert result["compatible"] is True and result["runnable_now"] is False
    assert "peft" in result["runtime"]["missing"]
    assert result["resources"]["disk"]["download_budget_gb"] == 0
    assert assess_definition(adapter, context(), catalog=catalog)["runnable_now"] is True
    a14b = assess_definition(catalog.get_definition("lora-wan22-a14b-svi-continuity"), context(gpu="A100", ram=192, ram_free=180, disk=512), catalog=catalog)
    assert a14b["resources"]["ram"]["required_gb"] > 128
    unknown_base = {**adapter, "compatible_models": [], "parent_model": None}
    assert assess_definition(unknown_base, context(), catalog=catalog)["compatible"] is False


def test_metadata_normalization_is_bounded_and_does_not_infer_vram(tmp_path, catalog):
    from studio.model_discovery import normalize_metadata
    metadata = raw("new/7B-fast-4bit-lora", downloads=None)
    metadata.update(tags=["lora"], safetensors={"parameters": {"F16": 7_000_000_000}, "total": 7_000_000_000},
                    siblings=[{"rfilename": f"folder/{i}.safetensors", "size": 1024} for i in range(300)] + [{"rfilename": "../unsafe.py"}])
    record = normalize_metadata(metadata, "image", now="2026-11-05")
    assert record["category"] == "lora" and len(record["files"]) == 256 and record["inventory_file_count"] == 301
    assert record["inventory_preview_truncated"] is True and record["file_sizes_complete"] is False
    result = store(tmp_path, catalog).decorate(record, context())
    assert result["support"]["status"] == "candidate"
    assert result["min_vram_gb"] is None and result["downloads"] is None
    assert result["compatibility"]["compatible"] is False and result["can_auto_load"] is False
    broken = raw()
    broken.update(tags="not-an-array", siblings=[{"rfilename": "model_index.json", "lfs": "bad", "size": 0}],
                  cardData="not-an-object", safetensors={"parameters": []}, _small_configs=["bad"], downloads=float("nan"))
    normalized = normalize_metadata(broken, "image", now="date")
    assert normalized["tags"] == [] and normalized["downloads"] is None and normalized["license"] is None
    assert normalized["files"][0]["size"] == 0


@pytest.mark.parametrize("mutation", ["custom_code", "wrong_class", "new_quantization"])
def test_inspected_incompatible_manifest_overrides_reviewed_recipe(tmp_path, catalog, mutation):
    from studio.model_discovery import normalize_metadata
    metadata = raw("black-forest-labs/FLUX.2-klein-4B")
    manifest = {"_class_name": "Flux2KleinPipeline", "transformer": ["diffusers", "Flux2Transformer2DModel"]}
    configs = {"model_index.json": manifest}
    if mutation == "custom_code":
        manifest["transformer"] = ["some_remote_repo", "InjectedModel"]
    elif mutation == "wrong_class":
        manifest["_class_name"] = "FluxPipeline"
    else:
        configs["transformer/config.json"] = {"quantization_config": {"quant_method": "unknown_backend"}}
    metadata["_small_configs"] = configs
    result = store(tmp_path, catalog).decorate(normalize_metadata(metadata, "image", now="now", inspected=True), context())
    assert result["support"]["status"] == "unsupported"
    assert result["recipes"] == [] and result["compatibility"]["compatible"] is False
    assert result["can_auto_load"] is False


def test_known_recipe_gate_and_companions_have_distinct_semantics(tmp_path, catalog):
    from studio.model_discovery import normalize_metadata
    service = store(tmp_path, catalog)
    known = raw("black-forest-labs/FLUX.2-klein-4B")
    known["gated"] = "manual"
    known["_small_configs"] = {"model_index.json": {"_class_name": "Flux2KleinPipeline"}}
    result = service.decorate(normalize_metadata(known, "image", now="now", inspected=True), context())
    assert result["support"]["status"] == "reviewed_recipe" and result["compatibility"]["compatible"] is True
    assert result["access"]["weights_access_verified"] is False and result["gated_locked"] is True
    assert result["access"]["token_present"] is True and result["compatibility"]["runnable_now"] is False
    # A shared encoder may be a component, but never a stand-alone full pipeline.
    definition = catalog.get_definition("wan21-vace-14b-fp8")
    companion = definition["artifacts"][-1]["repo_id"]
    component = service.decorate(normalize_metadata(raw(companion, pipeline="text-encoding"), "video", now="now"), context())
    assert definition["id"] in component["support"]["component_of"] or component["recipes"]


def test_search_cache_ttl_offline_retention_freshness_and_no_token_persistence(tmp_path, catalog):
    from studio.model_discovery import MetadataError
    clock = [1800000000.0]
    provider = ProviderStub([raw("new/starlight", downloads=20)])
    service = store(tmp_path, catalog, provider, lambda: clock[0])
    first = service.search("starlight", "image", context=context())
    assert first["source_mode"] == "network" and first["items"][0]["downloads"] == 20
    assert service.search("starlight", "image", context=context())["source_mode"] == "cache"
    assert len(provider.search_calls) == 1
    clock[0] += 901
    provider.error = MetadataError("offline", "Red no disponible; se conserva la caché.")
    offline = service.search("starlight", "image", context=context())
    assert offline["source_mode"] == "offline" and offline["stale"] is True
    assert offline["items"][0]["downloads"] == 20 and offline["metadata_checked_at"] == first["metadata_checked_at"]
    assert service.search("other", "image", refresh=True, context=context())["network_attempted"] is False
    assert len(provider.search_calls) == 2
    cached_text = service.path.read_text()
    assert "hf_not_for_public_metadata" not in cached_text and "token_present" not in cached_text
    restored = store(tmp_path, catalog, provider, lambda: clock[0])
    assert restored.status()["indexed_repositories"] == 1
    assert restored.search("starlight", "image", context=context())["items"][0]["downloads"] == 20


def test_summary_refresh_does_not_make_an_old_inspection_fresh(tmp_path, catalog):
    clock = [1800000000.0]
    provider = ProviderStub([raw("new/one")])
    provider.inspection = {**raw("new/one"), "_small_configs": {"model_index.json": {"_class_name": "SomePipeline"}}}
    service = store(tmp_path, catalog, provider, lambda: clock[0])
    inspected = service.inspect("new/one", context=context())
    original_inspection_date = inspected["item"]["inspection_checked_at"]
    clock[0] += 901
    service.search("one", "image", refresh=True, context=context())
    assert service._state["records"]["new/one"]["inspection_checked_at"] == original_inspection_date
    service.inspect("new/one", context=context())
    assert len(provider.inspect_calls) == 2
    provider.items[0]["sha"] = "b" * 40
    service.search("one", "image", refresh=True, context=context())
    assert service._state["records"]["new/one"]["metadata_level"] == "summary"
    assert service._state["records"]["new/one"]["pipeline_class"] is None


def test_paginated_snapshot_appends_without_duplicate_or_shift(tmp_path):
    from studio.model_discovery import ModelDiscovery

    class Pages(ProviderStub):
        def search_page(self, query, category, sort, cursors=None):
            self.search_calls.append(cursors)
            return {"items": [raw("new/alpha", downloads=20), raw("new/beta", downloads=10), raw("new/gamma", downloads=5)] if not cursors else [raw("new/delta", downloads=1000)],
                    "cursors": {"text-to-image": "https://huggingface.co/api/models?cursor=next" if not cursors else None}}

    provider = Pages()
    service = ModelDiscovery(path=tmp_path / "pages.json", catalog=CatalogStub([]), provider=provider)
    service.research = {"families": []}
    first = service.search("tag-not-in-repo", "image", limit=2, context=context())
    assert [i["repo_id"] for i in first["items"]] == ["new/alpha", "new/beta"]
    second = service.search("tag-not-in-repo", "image", limit=2, offset=first["next_offset"], context=context())
    assert [i["repo_id"] for i in second["items"]] == ["new/gamma", "new/delta"]
    assert second["next_offset"] is None
    again = service.search("tag-not-in-repo", "image", limit=2, context=context())
    assert [i["repo_id"] for i in again["items"]] == ["new/alpha", "new/beta"]
    assert len(provider.search_calls) == 2


def test_concurrent_identical_search_is_deduplicated_and_network_is_bounded(tmp_path, catalog):
    entered = threading.Event()
    release = threading.Event()

    class Blocking(ProviderStub):
        def search_page(self, query, category, sort, cursors=None):
            self.search_calls.append(query)
            entered.set()
            assert release.wait(5)
            return {"items": [raw("new/" + query)], "cursors": {}}

    provider = Blocking()
    service = store(tmp_path, catalog, provider)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(service.search, "first", "image", context=context())
        try:
            assert entered.wait(3)
            duplicate = service.search("first", "image", context=context())
            assert duplicate["source_mode"] == "refreshing" and duplicate["network_attempted"] is False
            other = executor.submit(service.search, "second", "image", context=context())
            # Wait deterministically for the second entry without sleeping/polling.
            # The global bound is also asserted directly under the service lock.
            with service._lock:
                service._inflight.add("held-slot")
            capped = service.search("third", "image", context=context())
            assert capped["network_attempted"] is False and capped["source_mode"] == "refreshing"
        finally:
            release.set()
            with service._lock:
                service._inflight.discard("held-slot")
        assert future.result(timeout=4)["source_mode"] == "network"
        other.result(timeout=4)
    assert provider.search_calls.count("first") == 1 and "third" not in provider.search_calls


@pytest.mark.parametrize("bad", [[], {"schema_version": 1, "records": {}, "queries": {}, "settings": {"auto_refresh": "false"}},
                                  {"schema_version": 1, "records": {}, "queries": {}, "cooldown_until": "tomorrow"},
                                  {"schema_version": 1, "records": {}, "queries": {"x": {"attempt_ts": float("nan")}}}])
def test_corrupt_index_is_offline_safe_and_does_not_delete_weights(tmp_path, catalog, bad):
    path = tmp_path / "index.json"
    path.write_text(json.dumps(bad))
    weights = tmp_path / "keep.safetensors"
    weights.write_bytes(b"important weights")
    service = store(tmp_path, catalog)
    assert service.status()["cache_warning"]
    assert service.status()["indexed_repositories"] == 0
    assert weights.read_bytes() == b"important weights"


def test_settings_persist_explicit_environment_opt_out_wins_and_scheduler_stops(tmp_path, catalog, monkeypatch):
    monkeypatch.delenv("STUDIO_MODEL_AUTO_REFRESH", raising=False)
    service = store(tmp_path, catalog)
    service.configure(auto_refresh=True, interval_minutes=15)
    assert store(tmp_path, catalog).status()["settings"] == {"auto_refresh": True, "interval_minutes": 15}
    monkeypatch.setenv("STUDIO_MODEL_AUTO_REFRESH", "0")
    disabled = store(tmp_path, catalog)
    assert disabled.status()["settings"]["auto_refresh"] is False
    queued = threading.Event()
    monkeypatch.setattr(disabled, "queue_refresh", lambda **_kw: queued.set())
    disabled.start()
    try:
        assert disabled.status()["scheduler_running"] is True
        disabled.configure(auto_refresh=True, interval_minutes=15)
        assert queued.wait(3)
    finally:
        disabled.stop()
    assert disabled.status()["scheduler_running"] is False
    assert disabled.status()["next_refresh_at"] is None
    for kwargs in ({"auto_refresh": "false"}, {"auto_refresh": True, "interval_minutes": 1}, {"auto_refresh": True, "interval_minutes": True}):
        with pytest.raises(ValueError):
            disabled.configure(**kwargs)


class ResponseStub:
    def __init__(self, data=None, *, status=200, headers=None, body=None):
        self.status_code = status
        self.headers = headers or {}
        self.body = json.dumps(data).encode() if body is None else body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, chunk_size):
        yield self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError("Do not persist this request/credential error")


def test_public_provider_contract_uses_expand_without_tokens_and_pins_only_small_json(monkeypatch):
    import requests
    from studio.model_discovery import HuggingFaceMetadataProvider
    calls = []
    repo = "owner/known"

    def get(url, **kwargs):
        calls.append((url, kwargs))
        prepared = requests.Request("GET", url, headers={"Authorization": "Bearer implicit-secret"}).prepare()
        assert "Authorization" not in kwargs["auth"](prepared).headers
        assert kwargs["allow_redirects"] is False
        assert all("secret" not in str(v) for v in kwargs["headers"].values())
        if url.endswith("/api/models"):
            return ResponseStub([raw(repo)])
        if "/api/models/" in url:
            return ResponseStub({**raw(repo), "siblings": [{"rfilename": p} for p in ("model_index.json", "config.json", "README.md", "weights.safetensors")]})
        return ResponseStub({"_class_name": "SomePipeline"})

    monkeypatch.setattr(requests, "get", get)
    provider = HuggingFaceMetadataProvider()
    provider.search_page("known", "image", "downloads")
    expanded = [value for key, value in calls[0][1]["params"] if key == "expand"]
    assert "id" not in expanded and "pipeline_tag" in expanded and "cardData" in expanded
    assert not {"full", "cardData", "fetch_config"}.intersection(key for key, value in calls[0][1]["params"])
    inspected = provider.inspect(repo)
    assert set(inspected["_small_configs"]) == {"model_index.json", "config.json"}
    assert all("README.md" not in url and "weights.safetensors" not in url for url, _ in calls)
    assert all("/resolve/" + "a" * 40 + "/" in url for url, _ in calls if "/resolve/" in url)


@pytest.mark.parametrize("url", ["http://huggingface.co/api/models", "https://evil.test/api/models", "https://huggingface.co/owner/model", "https://huggingface.co@evil.test/api/models"])
def test_provider_rejects_external_pagination_and_urls(url):
    from studio.model_discovery import HuggingFaceMetadataProvider, MetadataError
    with pytest.raises(MetadataError):
        HuggingFaceMetadataProvider._safe_api_url(url)


def test_provider_size_rate_limit_redirect_and_deadline_guards(monkeypatch):
    import time
    import requests
    from studio.model_discovery import HuggingFaceMetadataProvider, MetadataError
    provider = HuggingFaceMetadataProvider()
    for response, code in [(ResponseStub(status=429, headers={"Retry-After": "5000"}), "rate_limited"),
                           (ResponseStub(headers={"Content-Length": "4000001"}), "metadata_too_large"),
                           (ResponseStub(body=b"x" * 20), "metadata_too_large"),
                           (ResponseStub(status=302, headers={"Location": "https://evil.test/steal"}), "redirect_blocked")]:
        monkeypatch.setattr(requests, "get", lambda *_a, **_kw: response)
        with pytest.raises(MetadataError) as error:
            provider._get_json("https://huggingface.co/api/models", max_bytes=10 if code == "metadata_too_large" else 4_000_000)
        assert error.value.code == code
        if code == "rate_limited":
            assert error.value.retry_after == 3600
    with pytest.raises(MetadataError):
        provider._get_json("https://huggingface.co/api/models", deadline=time.monotonic() - 1)
    monkeypatch.setattr(requests, "get", lambda *_a, **_kw: ResponseStub(status=302, headers={"Location": "/api/resolve-cache/models/repeat"}))
    with pytest.raises(MetadataError) as error:
        provider._get_json("https://huggingface.co/api/models")
    assert error.value.code == "redirect_blocked"


def test_oracle_reuses_equivalent_weights_but_does_not_sacrifice_semantic_scope(catalog):
    from studio.model_oracle import recommend_pipeline
    # Already downloaded LTX beats a request to download another generative stack.
    catalog.downloaded.add("ltx-video-13b-distilled")
    video = recommend_pipeline("video_flf2v", "max_quality_14b", catalog=catalog, context=context(ram=64, ram_free=56))
    assert video["model_id"] == "ltx-video-13b-distilled" and video["automatic_weight_downloads"] is False
    assert video["recommended_arguments"]["video_submode"] == "flf2v" and video["recommended_arguments"]["wait"] is False
    # A builtin CV comparator must not eclipse an eligible semantic audit by being
    # marked "downloaded" with zero local weights.
    coherence = recommend_pipeline("coherence_check", "max_quality_14b", catalog=catalog, context=context())
    assert coherence["model_id"] == "qwen25-vl-3b-coherence"
    assert coherence["recommended_arguments"]["mode"] == "vlm_qwen"
    cpu = recommend_pipeline("coherence_check", "zero_vram_cpu", catalog=catalog, context=context(gpu="CPU", ram=4, ram_free=3))
    assert cpu["model_id"] == "opencv-fast-coherence" and cpu["caveats"]
    none = recommend_pipeline("video_t2v", "zero_vram_cpu", catalog=catalog, context=context(gpu="CPU", ram=4, ram_free=3))
    assert none["status"] == "no_compatible_recipe" and none["recommended_tool"] is None
    montage = recommend_pipeline("montage", "zero_vram_cpu", catalog=catalog, context=context(gpu="CPU", ram=4, ram_free=3))
    assert montage["status"] == "ready" and montage["required_inputs"] == ["items"]


def mcp_payload(result):
    return json.loads(result["content"][0]["text"])


def test_mcp_discovery_schemas_dispatch_resources_and_invalid_arguments(temp_studio_env, tmp_path, catalog, monkeypatch):
    import app as app_module
    import studio.mcp_server as mcp
    client, _root = temp_studio_env
    provider = ProviderStub([raw("new/metadata-only")])
    service = store(tmp_path, catalog, provider)
    monkeypatch.setattr(app_module, "model_discovery", service)
    monkeypatch.setattr(mcp, "model_discovery", service)
    monkeypatch.setattr(mcp, "assessment_context", lambda **_kw: context())
    monkeypatch.setattr(mcp, "catalog_manager", catalog)
    hdr = {"X-Studio-MCP-Key": client.get("/api/mcp/info").json()["token"]}
    tools = mcp._build_mcp_tools_schema()
    schema = {tool["name"]: tool for tool in tools}
    expected = {"studio_search_models", "studio_inspect_model", "studio_refresh_model_index", "studio_model_index_status", "studio_get_model_guide"}
    assert expected <= schema.keys()
    assert {"studio_ask_user", "studio_answer_preference", "studio_get_work_state", "studio_mix_music_tracks", "studio_render_midi_synth"} <= schema.keys()
    assert len(schema) == len(tools)  # No duplicate tool names.
    assert schema["studio_inspect_model"]["inputSchema"]["oneOf"]
    assert "SANA" in schema["studio_generate_or_edit_image"]["inputSchema"]["properties"]["use_official_defaults"]["description"]
    search = mcp_payload(mcp._execute_mcp_tool("studio_search_models", {"query": "metadata-only", "category": "image"}))
    assert search["items"][0]["support"]["status"] == "candidate"
    assert provider.inspect_calls == []
    guide = mcp_payload(mcp._execute_mcp_tool("studio_get_model_guide", {}))
    assert guide["uri"] == "studio://manual/model-oracle" and "runnable_now" in guide["markdown"]
    for arguments in ({"model_id": "sdxl-base-1.0", "repo_id": "owner/model"}, {}, {"repo_id": "https://evil.test/model"}):
        result = mcp._execute_mcp_tool("studio_inspect_model", arguments)
        assert result["isError"] is True and "INVALID_MODEL_QUERY" in result["content"][0]["text"]
    for arguments in ({"limit": True}, {"limit": 2.5}, {"only_compatible": "false"}, {"offset": -1}):
        assert mcp._execute_mcp_tool("studio_search_models", arguments)["isError"] is True
    resources = client.post("/mcp", headers=hdr, json={"jsonrpc": "2.0", "id": 1, "method": "resources/list"}).json()["result"]["resources"]
    uris = {r["uri"] for r in resources}
    assert {"studio://manual/model-oracle", "studio://manual/model-research", "studio://models/discovery-status", "studio://models/recipes"} <= uris
    read = client.post("/mcp", headers=hdr, json={"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": "studio://models/recipes"}}).json()
    data = json.loads(read["result"]["contents"][0]["text"])
    assert data["metadata_only"] is True and data["recipes"]
    assert client.get("/api/models/manual?guide=operations").status_code == 200
    assert client.get("/api/models/manual?guide=../../secrets").status_code == 400
    assert client.get("/api/models/recommend?task=invented").status_code == 400
    assert client.get("/api/models/missing/assessment").status_code == 404
    assert client.get("/api/models/sdxl-base-1.0/assessment?gpu_profile=invented").status_code == 400
    assert client.get("/api/catalog?gpu_profile=invented").status_code == 400
    assert client.get("/api/catalog/hf-search?limit=51").status_code == 422
    assert client.post("/api/models/discovery/settings", json={"auto_refresh": False, "interval_minutes": 60}).json()["settings"]["auto_refresh"] is False
    assert client.post("/api/models/discovery/settings", json={"auto_refresh": True, "interval_minutes": 5}).status_code == 422


def test_unknown_imports_are_candidates_and_cannot_be_prepared_even_with_files(temp_studio_env, monkeypatch):
    import app as app_module
    import studio.mcp_server as mcp
    from studio.catalog import catalog_manager
    from studio.engines.image_engine import _is_image_lora_compatible
    client, _root = temp_studio_env
    result = client.post("/api/catalog/custom", json={"name": "Unreviewed small image", "repo_id": "example/not-a-loader", "category": "image", "estimated_gb": 0.1}).json()
    definition = result["model"]
    assert result["candidate_only"] is True and result["download_allowed"] is False
    assert definition["engine"] == "unvalidated" and definition["min_vram_gb"] is None
    assert not _is_image_lora_compatible({**definition, "category": "lora"}, "qwen-image-edit-2511-pack")
    monkeypatch.setattr(catalog_manager, "is_downloaded", lambda mid: mid == definition["id"])
    assert client.post(f"/api/catalog/{definition['id']}/download").status_code == 400
    assert client.post("/api/memory/load", json={"model_id": definition["id"], "auto_download": True}).status_code == 400
    assert client.post("/api/image/generate", json={"model_id": definition["id"], "prompt": "one scene"}).status_code == 400
    assert mcp._execute_mcp_tool("studio_prepare_model", {"model_id": definition["id"]})["isError"] is True
    tools = {tool["name"]: tool for tool in mcp._build_mcp_tools_schema()}
    assert definition["id"] not in tools["studio_generate_or_edit_image"]["inputSchema"]["properties"]["model_id"]["enum"]
    assert definition["id"] not in tools["studio_prepare_model"]["inputSchema"]["properties"]["model_id"]["enum"]


def test_image_official_defaults_and_explicit_zero_cfg_match_rest_and_mcp(temp_studio_env, monkeypatch):
    import app as app_module
    import studio.mcp_server as mcp
    from studio.model_recipes import image_defaults
    from studio.catalog import catalog_manager
    client, _root = temp_studio_env
    captured = []

    def submit(**kwargs):
        kwargs["fn"](SimpleNamespace())
        return SimpleNamespace(id="test-job", status="completed", to_dict=lambda: {"id": "test-job", "status": "completed"})

    monkeypatch.setattr(mcp.job_manager, "submit", submit)
    monkeypatch.setattr(mcp, "generate_or_edit_image", lambda **kw: captured.append(kw) or {})
    monkeypatch.setattr(app_module, "generate_or_edit_image", lambda **kw: captured.append(kw) or {})
    for model_id in ("sdxl-base-1.0", "sana-sprint-600m", "z-image-turbo", "flux2-klein-4b"):
        expected = image_defaults(catalog_manager.get_definition(model_id))
        result = mcp._execute_mcp_tool("studio_generate_or_edit_image", {"model_id": model_id, "prompt": "a quiet scene", "wait": False})
        assert result["isError"] is False
        assert captured[-1]["steps"] == expected["steps"] and captured[-1]["guidance_scale"] == expected["guidance_scale"]
        response = client.post("/api/image/generate", json={"model_id": model_id, "prompt": "a quiet scene", "use_official_defaults": True})
        assert response.status_code == 200
        assert captured[-1]["steps"] == expected["steps"] and captured[-1]["guidance_scale"] == expected["guidance_scale"]
    mcp._execute_mcp_tool("studio_generate_or_edit_image", {"model_id": "sdxl-base-1.0", "prompt": "scene", "wait": False, "use_official_defaults": False, "guidance_scale": 0})
    assert captured[-1]["guidance_scale"] == 0


def test_new_image_factories_are_explicit_local_and_mixed_precision(tmp_path, monkeypatch):
    import sys
    import studio.engines.image_engine as engine
    from studio.model_recipes import EXTRA_IMAGE_MODELS, REVIEWED_IMAGE_ENGINES
    captured = []
    moves = []
    fake_torch = SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32", cuda=SimpleNamespace(get_device_properties=lambda _i: SimpleNamespace(major=8)))
    fake_diffusers = SimpleNamespace()
    for contract in REVIEWED_IMAGE_ENGINES.values():
        def load(path, **kwargs):
            captured.append((path, kwargs))
            return SimpleNamespace(vae=SimpleNamespace(to=lambda dtype: moves.append(("vae", dtype))),
                                   transformer=SimpleNamespace(to=lambda dtype: moves.append(("transformer", dtype))),
                                   text_encoder=SimpleNamespace(to=lambda dtype: moves.append(("text_encoder", dtype))))
        setattr(fake_diffusers, contract["classes"][0], SimpleNamespace(from_pretrained=load))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(engine, "apply_runtime_compatibility_patches", lambda: None)
    for definition in EXTRA_IMAGE_MODELS:
        manifest = tmp_path / "model_index.json"
        manifest.write_text(json.dumps({"_class_name": REVIEWED_IMAGE_ENGINES[definition["engine"]]["classes"][0]}))
        engine._instantiate_image_pipeline(definition["engine"], tmp_path, definition)
        kwargs = captured[-1][1]
        assert kwargs["local_files_only"] is True and kwargs["use_safetensors"] is True
        assert "trust_remote_code" not in kwargs
    assert ("text_encoder", "bf16") in moves and ("vae", "fp32") in moves and ("transformer", "fp16") in moves
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        engine._instantiate_image_pipeline("unvalidated", tmp_path)
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    with pytest.raises(ValueError, match="clase"):
        engine._instantiate_image_pipeline("sana_image", tmp_path)


def test_new_image_artifact_filters_do_not_download_root_duplicates_or_code():
    from studio.catalog import _matches_artifact_filters
    from studio.model_recipes import EXTRA_IMAGE_MODELS
    for definition in EXTRA_IMAGE_MODELS:
        art = definition["artifacts"][0]
        allowed = art["allow_patterns"]
        ignored = art["ignore_patterns"]
        for filename in ("root-checkpoint.safetensors", "README.md", "custom_pipeline.py", "transformer/unsafe.py", "transformer/model.pth", "text_encoder/model.bin", "vae/model.onnx"):
            assert not _matches_artifact_filters(filename, allowed, ignored, True), (definition["id"], filename)
        assert _matches_artifact_filters("model_index.json", allowed, ignored, True)
        if definition["id"] == "sdxl-base-1.0":
            assert _matches_artifact_filters("unet/diffusion_pytorch_model.fp16.safetensors", allowed, ignored, True)
            assert not _matches_artifact_filters("unet/diffusion_pytorch_model.safetensors", allowed, ignored, True)
        else:
            assert _matches_artifact_filters("transformer/diffusion_pytorch_model.safetensors", allowed, ignored, True)
            assert not _matches_artifact_filters("transformer/diffusion_pytorch_model.bf16.safetensors", allowed, ignored, True)


def test_discovery_ui_uses_only_server_metadata_and_has_all_controls(temp_studio_env):
    from html.parser import HTMLParser
    client, _root = temp_studio_env
    html = client.get("/").text
    js = client.get("/static/app.js").text

    class IdParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
        def handle_starttag(self, tag, attrs):
            self.ids.extend(value for name, value in attrs if name == "id")
    parser = IdParser()
    parser.feed(html)
    assert len(parser.ids) == len(set(parser.ids)), "DOM IDs must remain unique"
    for control in ("hf-auto-refresh", "hf-refresh-interval", "btn-hf-refresh-index", "hf-inspection", "btn-hf-oracle", "btn-hf-prev", "btn-hf-next", "hf-sort"):
        assert control in parser.ids and f'$("{control}")' in js
    assert 'id="img-cfg" value="4.0" min="0.0"' in html
    assert "populateImageModelSelect" in js and "applyImageModelDefaults" in js
    assert "evaluateClientGpuSticker" not in js and "fetchHfModelsFromBrowser" not in js
    assert not re.search(r'fetch\([`\'"]https://huggingface', js)
    assert "data-save-candidate" in js and "data-pick-recipe" in js
    assert "/api/models/manual?guide=operations" in html


def test_new_recipe_completion_requires_components_and_all_shards(tmp_path):
    from studio.catalog import catalog_manager
    from studio.model_recipes import image_recipe_files_complete
    definition = catalog_manager.get_definition("sana-sprint-600m")
    manifest = {"_class_name": "SanaSprintPipeline", "transformer": ["diffusers", "SanaTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderDC"], "text_encoder": ["transformers", "Gemma2Model"]}
    (tmp_path / "model_index.json").write_text(json.dumps(manifest))
    assert image_recipe_files_complete(definition, tmp_path) is False
    for component in ("transformer", "vae", "text_encoder"):
        folder = tmp_path / component
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"nonempty-presence-test-not-real-tensors")
    assert image_recipe_files_complete(definition, tmp_path) is True
    index = tmp_path / "text_encoder/model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer": "missing-00001.safetensors"}}))
    assert image_recipe_files_complete(definition, tmp_path) is False
    (index.parent / "missing-00001.safetensors").write_bytes(b"shard")
    assert image_recipe_files_complete(definition, tmp_path) is True
    index.write_text(json.dumps({"weight_map": {"layer": "../model.safetensors"}}))
    assert image_recipe_files_complete(definition, tmp_path) is False
    assert (tmp_path / "transformer/model.safetensors").exists()


def test_oracle_templates_match_real_mcp_schemas_for_all_tasks_and_priorities(catalog):
    from studio.mcp_server import _build_mcp_tools_schema
    from studio.model_oracle import TASKS, PRIORITIES, recommend_pipeline
    schemas = {tool["name"]: tool["inputSchema"] for tool in _build_mcp_tools_schema()}
    ctx = context(gpu="A100", ram=192, ram_free=180, disk=512, vram_free=76)
    for task in TASKS:
        for priority in PRIORITIES:
            result = recommend_pipeline(task, priority, catalog=catalog, context=ctx)
            if result["recommended_tool"] is None:
                assert result["status"] == "no_compatible_recipe"
                continue
            schema = schemas[result["recommended_tool"]]
            props = schema["properties"]
            assert result["recommended_arguments"].keys() <= props.keys(), (task, priority, result)
            assert set(result["required_inputs"]) <= props.keys(), (task, priority, result)
            for name, value in result["recommended_arguments"].items():
                if "enum" in props[name]:
                    assert value in props[name]["enum"], (task, priority, name, value)
            assert result["recommended_arguments"]["wait"] is False
            assert result["automatic_weight_downloads"] is False and result["timing_guarantee"] is None


def test_mcp_no_fit_does_not_fall_back_to_a_download(monkeypatch):
    import studio.mcp_server as mcp
    monkeypatch.setattr(mcp, "_compute_recommendation", lambda **_kw: {"status": "no_compatible_recipe", "recommended_arguments": {}, "alternatives": []})
    monkeypatch.setattr(mcp.job_manager, "submit", lambda **_kw: pytest.fail("No-fit must not enqueue inference/download"))
    result = mcp._execute_mcp_tool("studio_generate_video", {"mode": "ai", "video_submode": "t2v", "prompt": "a quiet street"})
    assert result["isError"] is True
    assert mcp_payload(result)["error_code"] == "NO_COMPATIBLE_VIDEO_RECIPE"


def test_refresh_metadata_job_is_deduplicated_and_contains_no_weights(tmp_path, catalog, monkeypatch):
    import time
    import studio.model_discovery as discovery
    from studio.jobs import JobManager
    provider = ProviderStub([raw("new/refreshed")])
    service = store(tmp_path, catalog, provider)
    monkeypatch.setattr(discovery, "assessment_context", lambda **_kw: context())
    result = service.refresh(category="image")
    assert result["updated"] is True and result["metadata_only"] is True
    assert result["status"]["refresh_running"] is False
    assert service.status()["refresh_running"] is False
    # queue_refresh uses the shared bounded JobManager with a dedup key; mock the
    # scheduler only, not the refresh logic that runs in the resulting job.
    manager = JobManager()
    import studio.jobs as jobs_module
    monkeypatch.setattr(jobs_module, "job_manager", manager)
    try:
        queued = service.queue_refresh(category="image")
        assert queued["metadata_only"] is True
        job = manager.get(queued["job"]["id"])
        end = time.monotonic() + 3
        while job.status in ("queued", "running") and time.monotonic() < end:
            time.sleep(0.01)
        assert job.status == "completed" and job.result["metadata_only"] is True
    finally:
        manager._executor.shutdown(wait=True)


def test_public_all_search_does_not_mislabel_unknown_tasks_as_images(tmp_path):
    from studio.model_discovery import TASKS
    assert TASKS["all"] == [None]
    provider = ProviderStub([raw("new/unknown-language-model", pipeline="text-generation")])
    service = store(tmp_path, CatalogStub([]), provider)
    result = service.search("unknown", "all", context=context())
    assert result["items"][0]["category"] == "unknown"
    assert result["items"][0]["compatibility"]["compatible"] is False


def test_corrupt_cached_record_fields_are_dropped_not_rendered(tmp_path, catalog):
    from studio.model_discovery import normalize_metadata
    service = store(tmp_path, catalog)
    record = normalize_metadata(raw(), "image", now="date")
    record["source_urls"] = [{}]
    service._state["records"]["example/model"] = record
    service._save()
    restored = store(tmp_path, catalog)
    assert restored.status()["indexed_repositories"] == 0
    restored.search("example", "image", context=context())


def test_kokoro_uses_its_local_model_and_voice_without_generic_tts_fallback(tmp_path, catalog, monkeypatch, colab_memory):
    colab_memory(ram=8, available=6, gpu=False)
    import sys
    import studio.engines.tts_engine as tts
    calls = []
    registered = {}
    model_id = "kokoro-82m-tts"
    catalog.downloaded.add(model_id)
    catalog.model_dir = lambda _mid: tmp_path
    monkeypatch.setattr(tts, "catalog_manager", catalog)
    from studio.memory_manager import RuntimeMemoryManager
    manager = RuntimeMemoryManager()
    register = manager.register_loaded
    def register_and_capture(**kw):
        registered.update(kw)
        return register(**kw)
    monkeypatch.setattr(manager, "register_loaded", register_and_capture)
    monkeypatch.setattr(tts, "memory_manager", manager)
    monkeypatch.setattr(tts, "run_with_live_memory_progress", lambda **kw: kw["fn"]())

    class KModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self):
            return self
    def pipeline(**kwargs):
        assert kwargs["lang_code"] == "e" and isinstance(kwargs["model"], KModel)
        return "local-kokoro-pipeline"
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KModel=KModel, KPipeline=pipeline))
    result = tts.load_tts_model_into_memory(model_id)
    assert result["model_id"] == model_id
    assert calls[0]["model"] == str(tmp_path / "base/kokoro-v1_0.pth")
    assert calls[0]["config"] == str(tmp_path / "base/config.json")
    assert registered["handle"]["voice"] == str(tmp_path / "base/voices/ef_dora.pt")


def test_scalar_pipeline_options_are_not_mistaken_for_remote_code(tmp_path, catalog):
    from studio.model_discovery import normalize_metadata
    metadata = raw("stabilityai/stable-diffusion-xl-base-1.0")
    metadata["_small_configs"] = {"model_index.json": {"_class_name": "StableDiffusionXLPipeline",
        "force_zeros_for_empty_prompt": True, "add_watermarker": None, "requires_safety_checker": False,
        "unet": ["diffusers", "UNet2DConditionModel"], "text_encoder": ["transformers", "CLIPTextModel"]}}
    record = normalize_metadata(metadata, "image", now="today", inspected=True)
    assert record["requires_remote_code"] is False
    result = store(tmp_path, catalog).decorate(record, context(gpu="T4"))
    assert result["support"]["status"] == "reviewed_recipe" and result["compatibility"]["compatible"] is True


def test_cpu_capable_tts_matches_actual_automatic_device_policy(catalog):
    from studio.model_compatibility import assess_definition
    from studio.model_oracle import recommend_pipeline
    model_id = "mms-tts-spa"
    catalog.downloaded.add(model_id)
    definition = catalog.get_definition(model_id)
    cpu = assess_definition(definition, context(gpu="CPU", ram=4, ram_free=3), catalog=catalog)
    gpu = assess_definition(definition, context(), catalog=catalog)
    assert cpu["compatible"] and cpu["resources"]["vram"]["required_gb"] == 0
    assert gpu["resources"]["vram"]["required_gb"] == definition["min_vram_gb"] > 0
    ctx = context()
    ctx["runtime"]["versions"]["edge-tts"] = None
    chosen = recommend_pipeline("tts", "balanced_fast", catalog=catalog, context=ctx)
    assert chosen["model_id"] == model_id and chosen["executable_now"] is True
    zero = recommend_pipeline("tts", "zero_vram_cpu", catalog=catalog, context=ctx)
    assert zero["model_id"] != model_id


def test_fast_oracle_recommends_assessed_speed_stack_not_downloaded_draft(catalog):
    from studio.model_oracle import recommend_pipeline
    catalog.downloaded.add("wan21-vace-1.3b")
    result = recommend_pipeline("video_i2v", "max_quality_14b", catalog=catalog,
                                context=context(ram=64, ram_free=56, disk=200), require_fast=True)
    assert result["model_id"] != "wan21-vace-1.3b"
    args = result["recommended_arguments"]
    assert args["require_fast"] is True
    if result["acceleration"]["kind"] == "speed_lora":
        assert args["video_lora_id"] in result["assessment"]["stack"]["adapter_ids"]
    assert result["gpu_validation"] == "No ejecutada por este oráculo"


def test_fast_oracle_never_recommends_causal_lora_for_flf2v(catalog):
    from studio.model_oracle import recommend_pipeline
    result = recommend_pipeline("video_flf2v", "max_quality_14b", catalog=catalog,
                                context=context(ram=64, ram_free=56, disk=200), require_fast=True)
    assert result["status"] != "no_compatible_recipe"
    args = result["recommended_arguments"]
    if args.get("video_lora_id"):
        lora = catalog.get_definition(args["video_lora_id"])
        assert not lora.get("causal_attention")
        assert "flf2v" not in lora.get("incompatible_submodes", [])


def test_fast_quality_oracle_refuses_silent_1_3b_downgrade(catalog):
    from studio.model_oracle import recommend_pipeline
    catalog.definitions = [d for d in catalog.definitions if d["id"] in {
        "wan21-vace-1.3b", "lora-wan21-1.3b-bidirect2"}]
    result = recommend_pipeline("video_i2v", "max_quality_14b", catalog=catalog,
                                context=context(ram=64, ram_free=56), require_fast=True)
    assert result["status"] == "no_compatible_recipe"
    assert not result["recommended_arguments"]


def test_search_finds_named_lora_within_shared_repository(tmp_path, catalog):
    service = store(tmp_path, catalog, ProviderStub([]))
    for term in ("fastwan", "lora-wan22-ti2v-5b-fastwan", "Wan2_2_5B_FastWanFullAttn"):
        result = service.search(query=term, category="lora", only_compatible=True,
                                context=context(ram=64, ram_free=56, disk=200))
        target = [x for x in result["items"] if x["repo_id"] == "Kijai/WanVideo_comfy"]
        assert target, term
        assert target[0]["catalog_model_ids"] == ["lora-wan22-ti2v-5b-fastwan"]
        assert not target[0]["access"]["weights_access_verified"]


def test_l4_api_filter_excludes_incompatible_even_if_downloaded_or_loaded(temp_studio_env, monkeypatch):
    """A file on disk/in RAM must not bypass the chosen profile's capacity filter."""
    from studio.hardware import hardware_tracker
    from studio.catalog import catalog_manager
    from studio.memory_manager import memory_manager
    client, _ = temp_studio_env
    snap = hardware_tracker.get_snapshot()
    snap["ram"].update(total_gb=53.0, available_gb=48.0)
    snap["disk"].update(free_gb=200.0)
    monkeypatch.setattr(hardware_tracker, "get_snapshot", lambda **_kw: snap)
    old_is_downloaded = catalog_manager.is_downloaded
    heavy = {"ltx-2.3-full-fp8-av", "ltx-2.5-full-av"}
    monkeypatch.setattr(catalog_manager, "is_downloaded", lambda mid: mid in heavy or old_is_downloaded(mid))
    monkeypatch.setattr(memory_manager, "loaded_model_ids", lambda: list(heavy))
    all_models = client.get("/api/catalog?gpu_profile=L4&only_compatible=false").json()["models"]
    for model_id in heavy:
        item = next(m for m in all_models if m["id"] == model_id)
        assert item["downloaded"] and item["loaded_in_memory"]
        assert not item["compatibility"]["compatible"]
    filtered = client.get("/api/catalog?gpu_profile=L4&only_compatible=true").json()["models"]
    assert not heavy.intersection(m["id"] for m in filtered)
    assert all(m["compatibility"]["compatible"] is True for m in filtered)
    assert "ffmpeg-motion-cpu" in {m["id"] for m in filtered}
    # Same L4, more host RAM: FP8 may fit by estimate; the BF16 full
    # variant still exceeds the L4 VRAM budget including the safety reserve.
    snap["ram"].update(total_gb=128.0, available_gb=120.0)
    ample = client.get("/api/catalog?gpu_profile=L4&only_compatible=true").json()["models"]
    ids = {m["id"] for m in ample}
    assert "ltx-2.3-full-fp8-av" in ids
    assert "ltx-2.5-full-av" not in ids
    item = next(m for m in ample if m["id"] == "ltx-2.3-full-fp8-av")
    assert item["compatibility"]["is_simulated"] and not item["compatibility"]["gpu_tested"]
