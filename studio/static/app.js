/* Studio IA — Lógica de Interfaz Web */
(function () {
  "use strict";

  const state = {
    catalog: [],
    catalogFilter: "all",
    gpuProfile: "auto",
    activeGpuProfile: null,
    vidOnlyCompat: true,
    catOnlyCompat: false,
    hfSearchResults: [],
    hfOffset: 0,
    hfNextOffset: null,
    hfSearching: false,
    modelDiscovery: null,
    hfAuth: { has_token: false, masked_token: null, username: null },
    assets: { uploads: [], images: [], videos: [], audio: [], montages: [] },
    assetsSignature: "",
    assetsLoading: false,
    voices: [],
    jobs: [],
    selectedJobId: null,
    watchedJobs: new Set(),
    completedHandled: new Set(),
    baseImage: null, // { rel_path, media_url, filename, width, height }
    refs: [null, null, null],
    lastImageResult: null,
    lastVideoResult: null,
    lastMontageResult: null,
    videoMode: "ai", // "ai" | "ffmpeg_motion"
    videoSubmode: "i2v", // "i2v" | "flf2v" | "t2v" | "ffmpeg"
    lastVidUploadStartRel: null,
    timeline: [], // [{ media_rel, media_url, filename, type, duration, motion, audio_rel, fit_to_audio }]
    mask: {
      enabled: false,
      tool: "draw", // "draw" | "erase"
      drawing: false,
      lastX: 0,
      lastY: 0,
      history: [],
      dirty: false,
    },
    unexportedCount: 0,
    lastHardware: null,
    selectedImageModel: "qwen-image-edit-2511-pack",
    imgLoraGroupFilter: "all",
    activeImgLoras: {}, // { [loraId]: scale }
    activeVidExtraLoras: {}, // { [loraId]: scale }
  };

  // ==================== HELPERS ====================
  function $(id) {
    return document.getElementById(id);
  }

  function toast(msg, kind = "info") {
    const box = $("toast-container");
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => {
      el.remove();
    }, 4500);
  }

  async function api(url, opts = {}) {
    const headers = opts.headers || {};
    if (opts.body && !(opts.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const resp = await fetch(url, { ...opts, headers });
    let data = null;
    try {
      data = await resp.json();
    } catch (_e) {
      data = null;
    }
    if (!resp.ok) {
      const detail = data && data.detail ? data.detail : `Error HTTP ${resp.status}`;
      const err = new Error(typeof detail === "string" ? detail : detail.message || JSON.stringify(detail));
      err.payload = detail;
      throw err;
    }
    return data;
  }

  function setWorkspaceMode(mode) {
    state.workspaceMode = mode === "higgsfield" ? "higgsfield" : "normal";
    const isHf = state.workspaceMode === "higgsfield";
    if ($("btn-mode-normal")) $("btn-mode-normal").classList.toggle("active", !isHf);
    if ($("btn-mode-higgsfield")) $("btn-mode-higgsfield").classList.toggle("active", isHf);
    if ($("normal-tabs-group")) $("normal-tabs-group").hidden = isHf;
    if ($("higgsfield-tabs-group")) $("higgsfield-tabs-group").hidden = !isHf;
    if (isHf) {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach((p) => {
        p.classList.toggle("active", p.id === "tab-higgsfield");
      });
    } else {
      switchTab(state.lastNormalTab || "tab-image");
    }
  }

  function switchTab(tabId) {
    if (tabId !== "tab-catalog" && tabId !== "tab-higgsfield") {
      state.lastNormalTab = tabId;
    }
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tabId);
    });
    document.querySelectorAll(".view-panel").forEach((p) => {
      p.classList.toggle("active", p.id === tabId);
    });
    if (tabId === "tab-catalog" && !state.hfSearchResults.length) runHfLiveSearch();
  }

  function showSpaceErrorBanner(spaceDetail) {
    const banner = $("space-banner");
    const content = $("space-banner-content");
    if (!spaceDetail || typeof spaceDetail !== "object") {
      content.textContent = String(spaceDetail || "Espacio insuficiente en disco.");
      banner.hidden = false;
      return;
    }
    let html = `<strong>⚠️ No cabe en el disco de sesión:</strong> ${escapeHtml(spaceDetail.message || "")}`;
    if (Array.isArray(spaceDetail.suggestions) && spaceDetail.suggestions.length > 0) {
      html += `<div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;">`;
      spaceDetail.suggestions.forEach((s) => {
        html += `<button class="btn btn-sm btn-danger" data-del-sug="${escapeHtml(s.id)}">🗑 Borrar ${escapeHtml(s.name)} (-${s.frees_gb} GB)</button>`;
      });
      html += `</div>`;
    }
    content.innerHTML = html;
    content.querySelectorAll("[data-del-sug]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await deleteModelFromDisk(btn.dataset.delSug);
        banner.hidden = true;
      });
    });
    banner.hidden = false;
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ==================== TELEMETRÍA Y TRABAJOS (POLLING) ====================
  let _pollTimer = null;

  async function pollStatus() {
    let activeCount = 0;
    try {
      const snap = await api("/api/status");
      activeCount = snap.active_jobs_count || 0;
      renderHardware(snap.hardware, snap.memory, snap.unexported_outputs_count);
      renderHfAuthStatus(snap.hf_auth);
      renderRuntimeStatus(snap.runtime);
      renderJobs(snap.jobs);
      // Los trabajos que lanza el agente por MCP no pasan por la interfaz: si no se
      // vuelven a leer los assets, sus imágenes y vídeos no aparecen nunca en las
      // galerías hasta recargar. Se refresca solo cuando cambia el número o el último job.
      const signature = `${(snap.jobs || []).length}:${(snap.jobs || [])[0]?.id || ""}:${(snap.jobs || [])[0]?.status || ""}`;
      if (signature !== state.assetsSignature) {
        state.assetsSignature = signature;
        await loadAssets();
      }
    } catch (_e) {
      // Silencioso en polling puntual
    }
    return activeCount;
  }

  function startAdaptivePolling() {
    async function _tick() {
      const activeCount = await pollStatus();
      const nextDelay = activeCount > 0 ? 700 : 1600;
      _pollTimer = setTimeout(_tick, nextDelay);
    }
    if (_pollTimer) clearTimeout(_pollTimer);
    _tick();
  }

  function renderRuntimeStatus(runtime) {
    const pill = $("runtime-pill");
    if (!pill) return;
    const rebuilt = runtime && runtime.runtime_rebuilt_for_compatibility;
    pill.hidden = !rebuilt;
    if (!rebuilt) return;
    $("runtime-state").textContent = "Reinicia el servidor";
    $("runtime-detail").textContent = "Parches aplicados tras cargar las librerías de GPU";
    pill.title = (runtime && runtime.runtime_rebuild_notice) || "Reinicia el proceso del servidor para evitar clases mezcladas. No hay que volver a descargar pesos.";
  }

  function renderHardware(hw, mem, unexportedCount) {
    state.lastHardware = hw;
    state.unexportedCount = unexportedCount || 0;

    const gpu = hw.gpu;
    $("gpu-badge").textContent = gpu.has_gpu
      ? `GPU activa: ${gpu.gpu_name} (${gpu.vram_total_gb} GB VRAM)`
      : `Modo CPU (sin GPU activa · listo para pruebas y montaje FFmpeg)`;

    $("vram-text").textContent = gpu.has_gpu
      ? `${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB`
      : `0 / 0 GB (CPU)`;
    $("vram-bar").style.width = `${Math.min(100, gpu.vram_percent || 0)}%`;

    $("ram-text").textContent = `${hw.ram.used_gb} / ${hw.ram.total_gb} GB`;
    $("ram-bar").style.width = `${Math.min(100, hw.ram.percent || 0)}%`;

    $("disk-text").textContent = `${hw.disk.free_gb} / ${hw.disk.total_gb} GB libres`;
    $("disk-bar").style.width = `${Math.min(100, hw.disk.percent || 0)}%`;

    const cr = hw.credits;
    $("credits-text").textContent = `${cr.remaining_credits} u.`;
    const hrsTxt = cr.estimated_hours_left !== null ? ` · ~${cr.estimated_hours_left}h` : "";
    $("credits-sub").textContent = `${cr.hourly_rate} u/h (gastado ${cr.session_spent})${hrsTxt}`;

    $("ephemeral-note-text").textContent = hw.ephemeral_disk_notice;

    const loadedSlots = (mem && mem.loaded_slots) || [];
    $("nav-loaded-count").textContent = `${loadedSlots.length} en RAM/VRAM`;
    const guardian = mem && mem.guardian;
    if (guardian && $("memory-guardian-state")) {
      const observation = guardian.last_observation || {};
      const pressure = observation.pressure || {};
      const reserve = pressure.reserve || {};
      const pending = guardian.last_action === "pressure_pending_safe_checkpoint";
      $("memory-guardian-state").textContent = !guardian.running ? "Sin vigilancia activa"
        : pending ? "Esperando punto seguro"
        : pressure.level === "critical" ? "Presión crítica"
        : pressure.level === "pressure" ? "Liberando inactivos"
        : pressure.level === "unknown" || guardian.measurements_healthy === false ? "Medición no disponible" : "Vigilando";
      $("memory-guardian-reserve").textContent = reserve.ram_gb !== undefined
        ? `Margen: ${Number(reserve.ram_gb).toFixed(1)} RAM · ${Number(reserve.vram_gb || 0).toFixed(1)} VRAM GiB`
        : "Modelos en uso protegidos";
      $("memory-guardian-pill").classList.toggle("memory-warning", guardian.measurements_healthy === false || (pressure.level && pressure.level !== "normal"));
      const events = guardian.recent_events || [];
      const lastEvent = events.length ? events[events.length - 1].message : "Sin expulsiones recientes.";
      $("memory-guardian-pill").title = `Solo se liberan modelos inactivos. En uso: ${(guardian.protected_slots || []).join(", ") || "ninguno"}. ${lastEvent}`;
    }

    const badge = $("unexported-badge");
    if (state.unexportedCount > 0) {
      badge.hidden = false;
      badge.textContent = String(state.unexportedCount);
    } else {
      badge.hidden = true;
    }

    // Banner de inactividad
    const idleBanner = $("idle-banner");
    if (cr.idle_warning) {
      idleBanner.hidden = false;
      $("idle-banner-msg").textContent = `Llevas ${cr.idle_minutes} min sin lanzar procesos con la GPU activa (${cr.hourly_rate} u/h).`;
    } else {
      idleBanner.hidden = true;
    }

    updateEngineBars(loadedSlots);
  }

  function updateEngineBars(loadedSlots) {
    const imgSlot = loadedSlots.find((s) => s.slot === "image");
    const curImgMid = ($("select-image-model") && $("select-image-model").value) || "qwen-image-edit-2511-pack";
    const imgDef = state.catalog.find((m) => m.id === curImgMid);
    const imgDot = $("img-engine-dot");
    const btnQwen = $("btn-prepare-qwen");
    if (imgSlot && imgSlot.model_id === curImgMid) {
      imgDot.className = "status-dot loaded";
      $("img-engine-state").textContent = `Cargado en RAM/VRAM (${imgSlot.name})`;
      if (!btnQwen.disabled) btnQwen.textContent = "✅ Modelo Listo en VRAM";
    } else if (imgDef && imgDef.downloaded) {
      imgDot.className = "status-dot downloaded";
      $("img-engine-state").textContent = "Descargado en disco · Sin cargar en VRAM";
      if (!btnQwen.disabled) btnQwen.textContent = "⚡ Cargar Modelo en VRAM";
    } else {
      imgDot.className = "status-dot";
      const estGb = imgDef ? imgDef.estimated_gb : "16.6";
      $("img-engine-state").textContent = `No descargado todavía (~${estGb} GB)`;
      if (!btnQwen.disabled) btnQwen.textContent = `⬇⚡ Descargar y Cargar (~${estGb} GB)`;
    }

    const vidSlot = loadedSlots.find((s) => s.slot === "video");
    const vidDot = $("vid-engine-dot");
    if (vidSlot) {
      vidDot.className = "status-dot loaded";
      $("vid-engine-state").textContent = `Activo en VRAM: ${vidSlot.name}`;
    } else {
      vidDot.className = "status-dot";
      $("vid-engine-state").textContent = "Sin modelo IA en VRAM (FFmpeg CPU disponible siempre)";
    }

    // Ocultar enlace rápido de descarga NSFW si ya está en disco
    const nsfwDef = state.catalog.find((m) => m.id === "qwen-lora-nsfw");
    const quickDlBtn = $("btn-quick-dl-nsfw");
    if (nsfwDef && quickDlBtn) {
      quickDlBtn.hidden = Boolean(nsfwDef.downloaded);
    }
  }

  function renderJobs(jobs) {
    state.jobs = jobs || [];
    const running = state.jobs.filter((j) => j.status === "running" || j.status === "queued");
    $("active-jobs-label").textContent =
      running.length > 0 ? `${running.length} tarea(s) en ejecución` : "Sin tareas activas";

    // Detector de cuelgue (>30s sin log)
    const hungJob = running.find((j) => j.possibly_hung);
    const hangBanner = $("hang-banner");
    if (hungJob) {
      hangBanner.hidden = false;
      $("hang-banner-msg").textContent = `«${hungJob.title}» lleva ${hungJob.seconds_since_update} s sin emitir avances.`;
      $("btn-cancel-hung").onclick = () => cancelJob(hungJob.id);
    } else {
      hangBanner.hidden = true;
    }

    // Tarjeta de trabajo activo (panel derecho + barra global superior + barra inline del motor)
    const topRunning = running[0];
    const ajCard = $("active-job-card");
    const gjs = $("global-job-strip");
    const imgInline = $("img-inline-progress");
    const vidInline = $("vid-inline-progress");
    const btnQwen = $("btn-prepare-qwen");
    const btnVidPrep = $("btn-prepare-video");

    if (topRunning) {
      ajCard.hidden = false;
      $("aj-title").textContent = topRunning.title;
      $("aj-pct").textContent = `${topRunning.progress}%`;
      $("aj-bar").style.width = `${topRunning.progress}%`;
      $("aj-step").textContent = topRunning.current_step || "";
      $("aj-speed").textContent = topRunning.speed_text || "";
      $("aj-elapsed").textContent = `${topRunning.elapsed_seconds} s`;
      $("btn-cancel-active-job").onclick = () => cancelJob(topRunning.id);

      // Barra global fija arriba (visible en cualquier pestaña y móvil/escritorio)
      gjs.hidden = false;
      $("gjs-title").textContent = topRunning.title;
      $("gjs-step").textContent = topRunning.current_step || "";
      $("gjs-speed").textContent = topRunning.speed_text || "";
      $("gjs-elapsed").textContent = `${topRunning.elapsed_seconds}s`;
      $("gjs-pct").textContent = `${topRunning.progress}%`;
      $("gjs-bar").style.width = `${topRunning.progress}%`;
      const lastLogEntry =
        Array.isArray(topRunning.logs) && topRunning.logs.length > 0
          ? topRunning.logs[topRunning.logs.length - 1]
          : null;
      $("gjs-lastlog").textContent = lastLogEntry
        ? `[${lastLogEntry.time}] ${lastLogEntry.message}`
        : "";
      $("btn-gjs-cancel").onclick = () => cancelJob(topRunning.id);

      // Barra inline directa en el panel de Imagen cuando descarga/carga Qwen o edita imagen
      const isImgRelated =
        topRunning.kind === "image" ||
        (topRunning.title && /qwen|imagen|lora/i.test(topRunning.title));
      if (isImgRelated && imgInline) {
        imgInline.hidden = false;
        const spd = topRunning.speed_text ? ` · ${topRunning.speed_text}` : "";
        $("img-iep-step").textContent = `${topRunning.current_step || topRunning.title}${spd}`;
        $("img-iep-meta").textContent = `${topRunning.progress}% (${topRunning.elapsed_seconds}s)`;
        $("img-iep-bar").style.width = `${topRunning.progress}%`;
        if (topRunning.kind === "load_model" || topRunning.kind === "download") {
          btnQwen.disabled = true;
          btnQwen.textContent = `⏳ ${topRunning.progress}% · Preparando Qwen...`;
        }
      } else if (imgInline) {
        imgInline.hidden = true;
        btnQwen.disabled = false;
      }

      // Barra inline directa en el panel de Vídeo
      const isVidRelated =
        topRunning.kind === "video" ||
        (topRunning.title && /wan|ltx|sana|skyreels|cogvideo|vace|vídeo|video/i.test(topRunning.title));
      if (isVidRelated && vidInline) {
        vidInline.hidden = false;
        const spd = topRunning.speed_text ? ` · ${topRunning.speed_text}` : "";
        $("vid-iep-step").textContent = `${topRunning.current_step || topRunning.title}${spd}`;
        $("vid-iep-meta").textContent = `${topRunning.progress}% (${topRunning.elapsed_seconds}s)`;
        $("vid-iep-bar").style.width = `${topRunning.progress}%`;
        if (topRunning.kind === "load_model" || topRunning.kind === "download") {
          btnVidPrep.disabled = true;
          btnVidPrep.textContent = `⏳ ${topRunning.progress}% · Preparando...`;
        }
      } else if (vidInline) {
        vidInline.hidden = true;
        btnVidPrep.disabled = false;
      }

      state.selectedJobId = topRunning.id;
    } else {
      ajCard.hidden = true;
      gjs.hidden = true;
      if (imgInline) imgInline.hidden = true;
      if (vidInline) vidInline.hidden = true;
      if (btnQwen) btnQwen.disabled = false;
      if (btnVidPrep) {
        btnVidPrep.disabled = false;
        syncVideoPrepareButtonLabel();
      }
      if (!state.selectedJobId && state.jobs.length > 0) {
        state.selectedJobId = state.jobs[0].id;
      }
    }

    // Detectar trabajos observados que acaban de terminar para refrescar vistas
    for (const j of state.jobs) {
      if (state.watchedJobs.has(j.id) && !state.completedHandled.has(j.id)) {
        if (j.status === "completed") {
          state.completedHandled.add(j.id);
          handleJobCompleted(j);
        } else if (j.status === "failed") {
          state.completedHandled.add(j.id);
          toast(`Error en «${j.title}»: ${j.error || "Revisa el registro"}`, "error");
        }
      }
    }

    // Mostrar logs del trabajo seleccionado
    const chosen = state.jobs.find((j) => j.id === state.selectedJobId) || state.jobs[0];
    const logBox = $("log-console");
    if (chosen && Array.isArray(chosen.logs) && chosen.logs.length > 0) {
      logBox.innerHTML = chosen.logs
        .map(
          (ln) =>
            `<div class="log-line ${escapeHtml(ln.level)}"><span class="log-time">[${escapeHtml(ln.time)}]</span>${escapeHtml(ln.message)}</div>`
        )
        .join("");
      logBox.scrollTop = logBox.scrollHeight;
    }

    // Lista de trabajos recientes
    const listEl = $("recent-jobs-list");
    listEl.innerHTML = state.jobs
      .slice(0, 12)
      .map((j) => {
        const sel = j.id === (chosen && chosen.id) ? "selected" : "";
        const statusMap = {
          completed: "OK",
          running: `${j.progress}%`,
          queued: "Cola",
          failed: "Error",
          cancelled: "Cancelado",
        };
        return `
          <div class="job-row ${sel}" data-jid="${escapeHtml(j.id)}">
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(j.title)}</span>
            <span class="job-badge ${escapeHtml(j.status)}">${statusMap[j.status] || j.status}</span>
          </div>
        `;
      })
      .join("");

    listEl.querySelectorAll(".job-row").forEach((row) => {
      row.addEventListener("click", () => {
        state.selectedJobId = row.dataset.jid;
        renderJobs(state.jobs);
      });
    });
  }

  async function handleJobCompleted(job) {
    await Promise.all([loadCatalog(), loadAssets()]);
    const res = job.result || {};
    if (res && res.media_url && state.workspaceMode === "higgsfield") {
      showHiggsfieldStageResult(res);
    }
    if (job.kind === "image" && res.media_url) {
      showImageResult(res);
      toast("Imagen lista y verificada.", "success");
    } else if (job.kind === "video" && res.media_url) {
      showVideoResult(res);
      toast("Clip de vídeo MP4 listo.", "success");
    } else if (job.kind === "tts" && res.media_url) {
      toast(`Locución lista (${res.duration_seconds}s).`, "success");
    } else if (job.kind === "audio_bgm" && res.media_url) {
      toast(`Pista musical/ambiental lista (${res.duration_seconds}s).`, "success");
    } else if ((job.kind === "lipsync" || job.kind === "audio_mix") && res.media_url) {
      if (res.type === "video") {
        showVideoResult(res);
        if ($("lipsync-result-wrap") && $("lipsync-result-player")) {
          $("lipsync-result-wrap").hidden = false;
          $("lipsync-result-player").src = `${res.media_url}?t=${Date.now()}`;
        }
      }
      toast(`LipSync / Masterización completada (${res.duration_seconds}s).`, "success");
    } else if (job.kind === "montage" && res.media_url) {
      showMontageResult(res);
      toast(`Montaje MP4 completado (${res.duration_seconds}s).`, "success");
    } else if (job.kind === "download") {
      toast("Descarga completada y verificada en disco.", "success");
    } else if (job.kind === "model_discovery") {
      renderDiscoveryStatus((res || {}).status);
      runHfLiveSearch();
      toast(res.updated ? "Índice de metadatos actualizado; sin pesos." : "Sin actualización remota; consulta caché/estado de red.", "info");
    } else if (job.kind === "load_model") {
      toast("Modelo cargado en RAM/VRAM correctamente.", "success");
    }
  }

  async function cancelJob(jobId) {
    try {
      await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      toast("Solicitud de cancelación enviada.");
      await pollStatus();
    } catch (e) {
      toast(e.message, "error");
    }
  }

  function watchJob(jobObj) {
    if (jobObj && jobObj.id) {
      state.watchedJobs.add(jobObj.id);
      state.selectedJobId = jobObj.id;
      pollStatus();
    }
  }

  // ==================== CATÁLOGO Y MEMORIA ====================
  // ==================== CATÁLOGO, FILTRO GPU Y EXPLORADOR HF ====================
  function formatVideoModes(modes) {
    if (!Array.isArray(modes) || modes.length === 0) return [];
    const map = {
      i2v: "🖼️ Imagen → Vídeo",
      t2v: "📝 Texto → Vídeo",
      flf2v: "🎞️ Frame a Frame",
      av: "🔊 Vídeo + Audio",
      audio_video: "🔊 Vídeo + Audio",
    };
    return modes.map((k) => ({ key: k, label: map[k] || k }));
  }

  function syncGpuPillsUI() {
    document.querySelectorAll(".gpu-pill").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.gpu === state.gpuProfile);
      if (btn.dataset.gpu === "auto" && state.activeGpuProfile && state.gpuProfile === "auto") {
        btn.textContent = `⚡ Mi GPU: ${state.activeGpuProfile.short || "Auto"}`;
      } else if (btn.dataset.gpu === "auto" && state.gpuProfile !== "auto") {
        btn.textContent = "⚡ Mi GPU (Auto)";
      }
    });
    const vidChk = $("chk-vid-only-compat");
    if (vidChk) vidChk.checked = state.vidOnlyCompat;
    const catChk = $("chk-cat-only-compat");
    if (catChk) catChk.checked = state.catOnlyCompat;
  }

  function syncVideoPrepareButtonLabel() {
    const btn = $("btn-prepare-video");
    const sel = $("select-video-model");
    if (!btn || !sel || btn.disabled) return;
    const m = state.catalog.find((item) => item.id === sel.value);
    if (!m) {
      btn.textContent = "⚡ Descargar y Cargar Modelo";
      return;
    }
    if (m.builtin_no_download) {
      btn.textContent = "✅ Integrado (0 GB · Listo)";
    } else if (m.loaded_in_memory) {
      btn.textContent = "✅ Activo en RAM/VRAM";
    } else if (m.downloaded) {
      btn.textContent = "⚡ Cargar en RAM/VRAM";
    } else {
      btn.textContent = `⬇⚡ Descargar y Cargar (${m.estimated_gb} GB)`;
    }
  }

  async function loadCatalog() {
    const data = await api(
      `/api/catalog?gpu_profile=${encodeURIComponent(state.gpuProfile)}&only_compatible=false`
    );
    state.catalog = data.models || [];
    state.activeGpuProfile = data.active_gpu_profile || null;
    renderDiscoveryStatus(data.discovery_status);
    populateImageModelSelect();
    syncGpuPillsUI();
    renderCatalog();
    populateVideoModelSelect();
    renderImageLoraRack();
    populateHiggsfieldLoraSelects();
    populateTtsEngineSelect();
    if (state.lastHardware) {
      updateEngineBars(data.loaded_slots || []);
    }
  }

  function populateImageModelSelect() {
    const select = $("select-image-model");
    if (!select) return;
    const previous = select.value || state.selectedImageModel;
    const models = state.catalog.filter((m) => m.category === "image" && m.validation_state !== "candidate");
    select.innerHTML = models.map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)}</option>`).join("");
    if (models.some((m) => m.id === previous)) select.value = previous;
    state.selectedImageModel = select.value;
  }

  function applyImageModelDefaults(modelId) {
    const model = state.catalog.find((m) => m.id === modelId);
    if (!model) return;
    state.selectedImageModel = modelId;
    const isQwen = ["qwen_image_edit", "qwen_image_t2i"].includes(model.engine);
    const defaults = model.image_defaults || {};
    $("img-steps").value = isQwen && !$("chk-lightning").checked ? 28 : (defaults.steps == null ? 4 : defaults.steps);
    $("img-cfg").value = defaults.guidance_scale == null ? 4.0 : defaults.guidance_scale;
    $("chk-lightning").disabled = !isQwen;
    $("chk-nsfw-lora").disabled = !isQwen;
  }

  function populateTtsEngineSelect() {
    const sel = $("tts-engine-select");
    if (!sel) return;
    const prev = sel.value;
    const ttsModels = state.catalog.filter((m) => m.category === "tts" && m.validation_state !== "candidate");
    if (ttsModels.length === 0) return;
    sel.innerHTML = ttsModels
      .map((m) => {
        const tag = m.builtin_no_download
          ? "0 GB · Instantáneo"
          : m.downloaded
          ? "En disco"
          : `${m.estimated_gb} GB`;
        return `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)} [${tag}]</option>`;
      })
      .join("");
    if (prev && ttsModels.some((m) => m.id === prev)) {
      sel.value = prev;
    }
  }

  function computeClientModelDimensions(modelObj, natW, natH) {
    const spec = (modelObj && modelObj.official_spec) || {};
    if (Array.isArray(spec.fixed_resolution) && spec.fixed_resolution.length === 2) {
      return { w: spec.fixed_resolution[0], h: spec.fixed_resolution[1] };
    }
    let ar = Number(natH || 704) / Math.max(1, Number(natW || 1280));
    if (ar >= 1.62 && Array.isArray(spec.portrait)) {
      return { w: spec.portrait[0], h: spec.portrait[1] };
    }
    if (ar <= 1 / 1.62 && Array.isArray(spec.landscape)) {
      return { w: spec.landscape[0], h: spec.landscape[1] };
    }
    if (ar >= 0.9 && ar <= 1.11 && Array.isArray(spec.square)) {
      return { w: spec.square[0], h: spec.square[1] };
    }
    const mod = Number(spec.mod_value || 32);
    const maxArea = Number(spec.max_area || 704 * 1280);
    ar = Math.max(9 / 16, Math.min(16 / 9, ar));
    const h = Math.max(mod * 8, Math.floor(Math.round(Math.sqrt(maxArea * ar)) / mod) * mod);
    const w = Math.max(mod * 8, Math.floor(Math.round(Math.sqrt(maxArea / ar)) / mod) * mod);
    return { w, h };
  }

  function setVideoSubmode(submode) {
    state.videoSubmode = submode;
    if (submode === "ffmpeg") {
      state.videoMode = "ffmpeg_motion";
    } else {
      state.videoMode = "ai";
      // Si el modelo actual no admite el submodo pulsado (ej. Wan 2.2 TI2V al pulsar Frame a Frame),
      // seleccionar automáticamente el mejor modelo compatible para evitar que el usuario lance el modo equivocado.
      const sel = $("select-video-model");
      const curModel = sel ? state.catalog.find((m) => m.id === sel.value) : null;
      const curModes = (curModel && curModel.video_modes) || [];
      if (!curModel || curModel.engine === "ffmpeg_motion" || !curModes.includes(submode)) {
        const bestCompat = state.catalog.find(
          (m) =>
            m.category === "video" &&
            m.engine !== "ffmpeg_motion" &&
            Array.isArray(m.video_modes) &&
            m.video_modes.includes(submode) &&
            (!m.compatibility || m.compatibility.compatible)
        );
        if (bestCompat && sel) {
          sel.value = bestCompat.id;
          applyVideoModelDefaults(bestCompat.id);
          populateVideoModelSelect();
          toast(
            `Seleccionado automáticamente «${bestCompat.name.split("(")[0].trim()}» (compatible con este modo). Si no está en VRAM se cargará al pulsar Generar.`
          );
          return;
        }
      }
    }
    syncVideoSubmodeUI();
    populateVideoModelSelect();
  }

  function syncVideoSubmodeUI() {
    const sub = state.videoSubmode || "i2v";
    document.querySelectorAll(".vsub-card").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.submode === sub);
    });

    const isFfmpeg = sub === "ffmpeg";
    const isT2v = sub === "t2v";
    const isFlf = sub === "flf2v";
    const ffmpegTwoFrames = isFfmpeg && $("chk-ffmpeg-two-frames") && $("chk-ffmpeg-two-frames").checked;
    const showEndSlot = isFlf || ffmpegTwoFrames;

    if ($("vid-frames-area")) $("vid-frames-area").hidden = isT2v;
    if ($("vid-t2v-note")) $("vid-t2v-note").hidden = !isT2v;
    if ($("vid-slot-end")) $("vid-slot-end").hidden = !showEndSlot;
    if ($("vid-swap-col")) $("vid-swap-col").hidden = !showEndSlot;
    if ($("lbl-vid-quick-add-end")) $("lbl-vid-quick-add-end").hidden = showEndSlot || isT2v;
    if ($("vid-start-label")) {
      $("vid-start-label").textContent = showEndSlot ? "1️⃣ Fotograma Inicial (Inicio)" : "🖼️ Imagen Inicial";
    }

    if ($("vid-ai-fields")) $("vid-ai-fields").hidden = isFfmpeg;
    if ($("vid-official-spec-bar")) $("vid-official-spec-bar").hidden = isFfmpeg;
    if ($("vid-ai-steps-row")) $("vid-ai-steps-row").hidden = isFfmpeg;
    if ($("vid-ffmpeg-fields")) $("vid-ffmpeg-fields").hidden = !isFfmpeg;

    const pInput = $("vid-prompt");
    const pLabel = $("vid-prompt-label");
    if (pInput && pLabel) {
      if (isT2v) {
        pLabel.textContent = "Descripción completa de la escena (Texto a Vídeo)";
        pInput.placeholder =
          "Ej: Un bebé riendo y bailando junto a un gato naranja en un salón soleado, luz natural cinematográfica, 24 fps...";
      } else if (isFlf) {
        pLabel.textContent = "Cómo debe ocurrir la transición del Fotograma Inicial al Final";
        pInput.placeholder =
          "Ej: La cámara avanza suavemente mientras el personaje gira y la iluminación pasa del día al atardecer...";
      } else {
        pLabel.textContent = "Prompt de movimiento sobre tu imagen";
        pInput.placeholder =
          "Ej: La cámara se acerca suavemente, brisa ligera en el cabello, movimiento natural fluido...";
      }
    }

    const runBtn = $("btn-run-video");
    if (runBtn) {
      if (isFfmpeg) {
        runBtn.textContent = ffmpegTwoFrames
          ? "⚡ Crear Transición FFmpeg Inicio → Fin (0 GB)"
          : "⚡ Crear Clip FFmpeg Instantáneo (0 GB)";
      } else if (isFlf) {
        runBtn.textContent = "🎞️ Generar Vídeo Frame a Frame (Inicio → Fin)";
      } else if (isT2v) {
        runBtn.textContent = "📝 Generar Vídeo desde Texto";
      } else {
        runBtn.textContent = "🖼️ Generar Vídeo desde Imagen";
      }
    }

    const curVidSel = $("select-video-model");
    if (curVidSel && curVidSel.value && !isFfmpeg) {
      populateVideoLoraSelect(curVidSel.value);
      updateVideoStepsSafetyNote();
    }

    updateVideoSourceThumb();
    updateVideoEndThumb();
    renderVideoCompatBanner();
  }

  function renderVideoCompatBanner() {
    const banner = $("vid-submode-compat-banner");
    const sel = $("select-video-model");
    if (!banner || !sel) return;

    const sub = state.videoSubmode;
    if (sub === "ffmpeg") {
      banner.hidden = true;
      return;
    }

    const curModel = state.catalog.find((m) => m.id === sel.value);
    if (!curModel || curModel.engine === "ffmpeg_motion") {
      if (curModel && curModel.engine === "ffmpeg_motion") {
        banner.hidden = false;
        banner.innerHTML = `
          <div>ℹ️ Tienes elegido el motor <strong>FFmpeg (0 GB CPU)</strong> arriba. Para generar con IA elige un modelo GPU como <strong>Wan 2.2 TI2V-5B</strong>, <strong>LTX-Video 0.9.5</strong> o <strong>Wan 2.1 VACE 1.3B</strong>.</div>
        `;
      } else {
        banner.hidden = true;
      }
      return;
    }

    const modes = Array.isArray(curModel.video_modes) ? curModel.video_modes : ["i2v", "t2v"];
    if (modes.includes(sub)) {
      banner.hidden = true;
      return;
    }

    const subNameMap = {
      flf2v: "Frame a Frame (Inicio → Fin)",
      i2v: "Imagen a Vídeo",
      t2v: "Texto a Vídeo",
    };
    const compatCandidates = state.catalog.filter(
      (m) =>
        m.category === "video" &&
        m.engine !== "ffmpeg_motion" &&
        Array.isArray(m.video_modes) &&
        m.video_modes.includes(sub) &&
        (!m.compatibility || m.compatibility.compatible)
    );

    const btnsHtml = compatCandidates
      .slice(0, 3)
      .map(
        (c) =>
          `<button type="button" class="btn btn-secondary btn-xs" data-pick-compat="${escapeHtml(c.id)}">Elegir ${escapeHtml(
            c.name.split("(")[0].trim()
          )}</button>`
      )
      .join("");

    banner.hidden = false;
    banner.innerHTML = `
      <div>⚠️ <strong>«${escapeHtml(curModel.name)}»</strong> no incluye el modo <strong>${escapeHtml(
      subNameMap[sub] || sub
    )}</strong>. Cámbialo con 1 clic a un modelo compatible:</div>
      <div class="vid-compat-actions">${btnsHtml}</div>
    `;
    banner.querySelectorAll("[data-pick-compat]").forEach((b) => {
      b.addEventListener("click", () => {
        const targetId = b.dataset.pickCompat;
        sel.value = targetId;
        applyVideoModelDefaults(targetId);
        populateVideoModelSelect();
        renderVideoCompatBanner();
        toast("Modelo compatible seleccionado. Pulsa «Descargar y Cargar» si aún no está en VRAM.");
      });
    });
  }

  function updateModelPresetsAndSpecBar(m) {
    const spec = (m && m.official_spec) || {};
    const land = spec.landscape || [1280, 704];
    const port = spec.portrait || [704, 1280];
    const sq = spec.square || [960, 960];

    const b169 = $("vid-preset-16-9");
    const b916 = $("vid-preset-9-16");
    const b11 = $("vid-preset-1-1");
    if (b169) {
      b169.dataset.vw = String(land[0]);
      b169.dataset.vh = String(land[1]);
      b169.textContent = `Horizontal (${land[0]}×${land[1]})`;
    }
    if (b916) {
      b916.dataset.vw = String(port[0]);
      b916.dataset.vh = String(port[1]);
      b916.textContent =
        spec.fixed_resolution ? `Fijo (${port[0]}×${port[1]})` : `Vertical (${port[0]}×${port[1]})`;
    }
    if (b11) {
      b11.dataset.vw = String(sq[0]);
      b11.dataset.vh = String(sq[1]);
      b11.textContent = spec.fixed_resolution ? `Nativo (${sq[0]}×${sq[1]})` : `1:1 (${sq[0]}×${sq[1]})`;
    }

    const barTxt = $("vid-official-spec-text");
    if (barTxt) {
      barTxt.textContent = spec.official_note
        ? `📘 Oficial ${m.name.split("(")[0].trim()}: ${spec.official_note}`
        : `📘 Ajustes cargados para ${m ? m.name : "Vídeo IA"}`;
    }
  }

  function syncLoraQualityPills() {
    const pillsWrap = $("vid-lora-quality-pills");
    const lid = $("vid-lora-select") ? $("vid-lora-select").value : "";
    if (!pillsWrap) return;
    pillsWrap.hidden = !lid;
    if (!lid) return;

    const steps = Number(($("vid-steps") && $("vid-steps").value) || 6);
    const scale = Number(($("vid-lora-scale") && $("vid-lora-scale").value) || 1.0);
    const cfg = Number(($("vid-cfg") && $("vid-cfg").value) || 1.0);

    pillsWrap.querySelectorAll(".lora-q-pill").forEach((pill) => {
      const ps = Number(pill.dataset.lsteps);
      const psc = Number(pill.dataset.lscale);
      const pc = Number(pill.dataset.lcfg);
      const match = ps === steps && Math.abs(psc - scale) < 0.03 && Math.abs(pc - cfg) < 0.06;
      pill.classList.toggle("active", match);
    });
  }

  function populateVideoLoraSelect(modelId) {
    const loraBox = $("vid-lora-box");
    const loraSel = $("vid-lora-select");
    const scaleWrap = $("vid-lora-scale-wrap");
    const dlBtn = $("btn-dl-vid-lora");
    const pillsWrap = $("vid-lora-quality-pills");
    if (!loraBox || !loraSel) return;

    const curSub = state.videoSubmode || "i2v";
    const prev = loraSel.value;

    renderVideoExtraLoraRack(modelId);

    const compatLoras = state.catalog.filter(
      (m) =>
        m.category === "lora" &&
        m.lora_target === "video" &&
        !m.stackable_creative &&
        (!Array.isArray(m.compatible_models) || m.compatible_models.includes(modelId)) &&
        (!Array.isArray(m.incompatible_submodes) || !m.incompatible_submodes.includes(curSub))
    );

    if (compatLoras.length === 0) {
      loraBox.hidden = true;
      loraSel.innerHTML = `<option value="">Sin LoRA acelerador</option>`;
      loraSel.value = "";
      if (scaleWrap) scaleWrap.hidden = true;
      if (dlBtn) dlBtn.hidden = true;
      if (pillsWrap) pillsWrap.hidden = true;
      // Si antes había un LoRA causal/destilado activo y pasamos a Frame a Frame, restaurar pasos/CFG oficiales
      if (prev) {
        const m = state.catalog.find((x) => x.id === modelId);
        const spec = (m && m.official_spec) || {};
        const recSteps = Number(spec.default_steps || (m && m.default_steps) || 30);
        const recCfg = Number(spec.default_cfg || (m && m.default_cfg) || 5.0);
        if ($("vid-steps")) $("vid-steps").value = recSteps;
        if ($("vid-steps-slider")) $("vid-steps-slider").value = recSteps;
        if ($("vid-cfg")) $("vid-cfg").value = recCfg;
      }
      return;
    }

    loraBox.hidden = false;
    loraSel.innerHTML =
      `<option value="">Sin LoRA acelerador (Modo Oficial Estándar)</option>` +
      compatLoras
        .map((l) => {
          const tag = l.downloaded ? "📦 En disco" : `⬇️ ${l.estimated_gb} GB · auto-descarga`;
          return `<option value="${escapeHtml(l.id)}">[${escapeHtml(tag)}] ${escapeHtml(l.name)}</option>`;
        })
        .join("");

    if (prev && compatLoras.some((l) => l.id === prev)) {
      loraSel.value = prev;
    } else {
      loraSel.value = "";
    }

    const chosen = compatLoras.find((l) => l.id === loraSel.value);
    if (scaleWrap) scaleWrap.hidden = !chosen;
    if (dlBtn) {
      dlBtn.hidden = !chosen || Boolean(chosen.downloaded);
      if (chosen && !chosen.downloaded) {
        dlBtn.textContent = `⬇️ Pre-descargar (${chosen.estimated_gb} GB)`;
      }
    }
    syncLoraQualityPills();
  }

  function renderImageLoraRack() {
    const box = $("img-lora-rack-list");
    const countBadge = $("img-lora-active-count");
    if (!box) return;

    const imgModelId = ($("select-image-model") && $("select-image-model").value) || "qwen-image-edit-2511-pack";
    const allCompatImgLoras = state.catalog.filter(
      (m) =>
        m.category === "lora" &&
        (m.lora_target || "image") === "image" &&
        m.id !== "qwen-lora-lightning-8steps" &&
        m.id !== "qwen-lora-nsfw" &&
        (!Array.isArray(m.compatible_models) || m.compatible_models.includes(imgModelId))
    );

    // Mantener solo LoRAs activos que sigan siendo compatibles con el modelo seleccionado
    const compatSet = new Set(allCompatImgLoras.map((x) => x.id));
    Object.keys(state.activeImgLoras).forEach((k) => {
      if (!compatSet.has(k)) delete state.activeImgLoras[k];
    });

    const activeCount = Object.keys(state.activeImgLoras).length;
    if (countBadge) {
      countBadge.textContent = `${activeCount} activo${activeCount === 1 ? "" : "s"}`;
      countBadge.classList.toggle("has-active", activeCount > 0);
    }

    const grp = state.imgLoraGroupFilter || "all";
    const visible = allCompatImgLoras.filter((m) => grp === "all" || m.lora_group === grp);
    if (visible.length === 0) {
      box.innerHTML = `<div class="muted" style="font-size:0.78rem;padding:4px 2px;">No hay LoRAs en este grupo para el modelo actual. Prueba la pestaña «Todos».</div>`;
      return;
    }

    box.innerHTML = visible
      .map((l) => {
        const isChecked = Object.prototype.hasOwnProperty.call(state.activeImgLoras, l.id);
        const curScale = isChecked ? state.activeImgLoras[l.id] : Number(l.default_lora_scale || 0.85);
        const dlBadge = l.downloaded
          ? `<span class="lora-dl-tag ok">✓ Disco</span>`
          : `<button type="button" class="lora-dl-btn" data-dl-lora="${escapeHtml(l.id)}" title="Pre-descargar ahora (también se auto-descarga al generar)">⬇ ${l.estimated_gb} GB</button>`;
        const trigBtn = l.trigger_phrase
          ? `<button type="button" class="lora-trig-btn" data-trig="${escapeHtml(l.trigger_phrase)}" data-target="img-prompt" title="Insertar frase de activación oficial en el prompt">+ Prompt</button>`
          : "";
        return `
          <div class="lora-rack-item ${isChecked ? "active" : ""}">
            <div class="lri-top">
              <label class="lri-check">
                <input type="checkbox" data-img-lora-chk="${escapeHtml(l.id)}" data-def-scale="${curScale}" ${isChecked ? "checked" : ""} />
                <span class="lri-title" title="${escapeHtml(l.description || "")}">${escapeHtml(l.name)}</span>
              </label>
              <div class="lri-actions">
                ${trigBtn}
                ${dlBadge}
              </div>
            </div>
            ${
              isChecked
                ? `<div class="lri-scale-row">
                    <span>Fuerza: <strong id="lbl-isl-${escapeHtml(l.id)}">${curScale.toFixed(2)}</strong></span>
                    <input type="range" min="0.10" max="1.40" step="0.05" value="${curScale}" data-img-lora-scale="${escapeHtml(l.id)}" />
                  </div>`
                : ""
            }
          </div>
        `;
      })
      .join("");

    box.querySelectorAll("[data-img-lora-chk]").forEach((chk) => {
      chk.addEventListener("change", (ev) => {
        const lid = ev.target.dataset.imgLoraChk;
        if (ev.target.checked) {
          state.activeImgLoras[lid] = Number(ev.target.dataset.defScale || 0.85);
        } else {
          delete state.activeImgLoras[lid];
        }
        renderImageLoraRack();
      });
    });

    box.querySelectorAll("[data-img-lora-scale]").forEach((rng) => {
      rng.addEventListener("input", (ev) => {
        const lid = ev.target.dataset.imgLoraScale;
        const val = Number(ev.target.value || 0.85);
        state.activeImgLoras[lid] = val;
        const lbl = $(`lbl-isl-${lid}`);
        if (lbl) lbl.textContent = val.toFixed(2);
      });
    });

    box.querySelectorAll("[data-dl-lora]").forEach((btn) => {
      btn.addEventListener("click", () => downloadModel(btn.dataset.dlLora));
    });

    box.querySelectorAll("[data-trig]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const trig = btn.dataset.trig || "";
        const inp = $(btn.dataset.target || "img-prompt");
        if (!inp || !trig) return;
        if (!inp.value.toLowerCase().includes(trig.toLowerCase())) {
          inp.value = inp.value.trim() ? `${trig} ${inp.value.trim()}` : trig;
          toast(`Frase activadora insertada: «${trig}»`, "info");
        }
      });
    });
  }

  function renderVideoExtraLoraRack(modelId) {
    const wrap = $("vid-extra-lora-box");
    const box = $("vid-extra-lora-list");
    const countBadge = $("vid-extra-lora-count");
    if (!wrap || !box) return;

    const curSub = state.videoSubmode || "i2v";
    const creativeLoras = state.catalog.filter(
      (m) =>
        m.category === "lora" &&
        m.lora_target === "video" &&
        Boolean(m.stackable_creative) &&
        (!Array.isArray(m.compatible_models) || m.compatible_models.includes(modelId)) &&
        (!Array.isArray(m.incompatible_submodes) || !m.incompatible_submodes.includes(curSub))
    );

    const compatSet = new Set(creativeLoras.map((x) => x.id));
    Object.keys(state.activeVidExtraLoras).forEach((k) => {
      if (!compatSet.has(k)) delete state.activeVidExtraLoras[k];
    });

    if (creativeLoras.length === 0) {
      wrap.hidden = true;
      return;
    }
    wrap.hidden = false;

    const activeCount = Object.keys(state.activeVidExtraLoras).length;
    if (countBadge) {
      countBadge.textContent = `${activeCount} activo${activeCount === 1 ? "" : "s"}`;
      countBadge.classList.toggle("has-active", activeCount > 0);
    }

    box.innerHTML = creativeLoras
      .map((l) => {
        const isChecked = Object.prototype.hasOwnProperty.call(state.activeVidExtraLoras, l.id);
        const curScale = isChecked ? state.activeVidExtraLoras[l.id] : Number(l.default_lora_scale || 0.75);
        const dlBadge = l.downloaded
          ? `<span class="lora-dl-tag ok">✓ Disco</span>`
          : `<button type="button" class="lora-dl-btn" data-dl-vlora="${escapeHtml(l.id)}">⬇ ${l.estimated_gb} GB</button>`;
        const trigBtn = l.trigger_phrase
          ? `<button type="button" class="lora-trig-btn" data-vtrig="${escapeHtml(l.trigger_phrase)}">+ Prompt</button>`
          : "";
        return `
          <div class="lora-rack-item ${isChecked ? "active" : ""}">
            <div class="lri-top">
              <label class="lri-check">
                <input type="checkbox" data-vid-elora-chk="${escapeHtml(l.id)}" data-def-scale="${curScale}" ${isChecked ? "checked" : ""} />
                <span class="lri-title" title="${escapeHtml(l.description || "")}">${escapeHtml(l.name)}</span>
              </label>
              <div class="lri-actions">
                ${trigBtn}
                ${dlBadge}
              </div>
            </div>
            ${
              isChecked
                ? `<div class="lri-scale-row">
                    <span>Fuerza: <strong id="lbl-vsl-${escapeHtml(l.id)}">${curScale.toFixed(2)}</strong></span>
                    <input type="range" min="0.10" max="1.30" step="0.05" value="${curScale}" data-vid-elora-scale="${escapeHtml(l.id)}" />
                  </div>`
                : ""
            }
          </div>
        `;
      })
      .join("");

    box.querySelectorAll("[data-vid-elora-chk]").forEach((chk) => {
      chk.addEventListener("change", (ev) => {
        const lid = ev.target.dataset.vidEloraChk;
        if (ev.target.checked) {
          state.activeVidExtraLoras[lid] = Number(ev.target.dataset.defScale || 0.75);
        } else {
          delete state.activeVidExtraLoras[lid];
        }
        renderVideoExtraLoraRack(modelId);
      });
    });

    box.querySelectorAll("[data-vid-elora-scale]").forEach((rng) => {
      rng.addEventListener("input", (ev) => {
        const lid = ev.target.dataset.vidEloraScale;
        const val = Number(ev.target.value || 0.75);
        state.activeVidExtraLoras[lid] = val;
        const lbl = $(`lbl-vsl-${lid}`);
        if (lbl) lbl.textContent = val.toFixed(2);
      });
    });

    box.querySelectorAll("[data-dl-vlora]").forEach((btn) => {
      btn.addEventListener("click", () => downloadModel(btn.dataset.dlVlora));
    });

    box.querySelectorAll("[data-vtrig]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const trig = btn.dataset.vtrig || "";
        const inp = $("vid-prompt");
        if (!inp || !trig) return;
        if (!inp.value.toLowerCase().includes(trig.toLowerCase())) {
          inp.value = inp.value.trim() ? `${trig}, ${inp.value.trim()}` : trig;
          toast(`Frase activadora insertada: «${trig}»`, "info");
        }
      });
    });
  }

  function populateHiggsfieldLoraSelects() {
    const imgLoras = state.catalog.filter(
      (m) =>
        m.category === "lora" &&
        (m.lora_target || "image") === "image" &&
        m.id !== "qwen-lora-lightning-8steps" &&
        m.id !== "qwen-lora-nsfw" &&
        (!Array.isArray(m.compatible_models) || m.compatible_models.includes("qwen-image-edit-2511-pack"))
    );
    const vidLoras = state.catalog.filter(
      (m) => m.category === "lora" && m.lora_target === "video" && Boolean(m.stackable_creative)
    );

    const imgOpts = imgLoras
      .map((l) => `<option value="${escapeHtml(l.id)}">${l.downloaded ? "✓ " : "⬇ "}${escapeHtml(l.name)}</option>`)
      .join("");
    const vidOpts = vidLoras
      .map((l) => `<option value="${escapeHtml(l.id)}">${l.downloaded ? "✓ " : "⬇ "}${escapeHtml(l.name)}</option>`)
      .join("");

    ["hf-dir-img-lora", "hf-cin-img-lora"].forEach((sid) => {
      const el = $(sid);
      if (!el) return;
      const prev = el.value;
      el.innerHTML = `<option value="">— Ninguno extra (solo Lightning 4p) —</option>` + imgOpts;
      if (prev) el.value = prev;
    });

    ["hf-dir-vid-lora", "hf-mot-vid-lora"].forEach((sid) => {
      const el = $(sid);
      if (!el) return;
      const prev = el.value;
      el.innerHTML = `<option value="">— Ninguno extra —</option>` + vidOpts;
      if (prev) el.value = prev;
    });
  }

  function getLoraRecommendedForSubmode(loraObj, submode) {
    if (!loraObj) return null;
    const isFlf = submode === "flf2v";
    return {
      steps: Number(
        (isFlf && loraObj.flf2v_recommended_steps) || loraObj.recommended_steps || 6
      ),
      cfg: Number(
        isFlf && loraObj.flf2v_recommended_cfg != null
          ? loraObj.flf2v_recommended_cfg
          : loraObj.recommended_cfg != null
          ? loraObj.recommended_cfg
          : 1.0
      ),
      scale: Number(
        (isFlf && loraObj.flf2v_recommended_scale) || loraObj.default_lora_scale || 1.0
      ),
    };
  }

  function resetVideoStepsToRecommended() {
    const mid = $("select-video-model") ? $("select-video-model").value : "";
    const m = state.catalog.find((x) => x.id === mid);
    const spec = (m && m.official_spec) || {};
    const lid = $("vid-lora-select") ? $("vid-lora-select").value : "";
    const loraObj = lid ? state.catalog.find((x) => x.id === lid) : null;
    const loraRec = getLoraRecommendedForSubmode(loraObj, state.videoSubmode || "i2v");

    const recSteps = loraRec
      ? loraRec.steps
      : Number(spec.default_steps || (m && m.default_steps) || 45);
    const recCfg = loraRec
      ? loraRec.cfg
      : Number(spec.default_cfg || (m && m.default_cfg) || 5.0);

    if ($("vid-steps")) $("vid-steps").value = recSteps;
    if ($("vid-steps-slider")) $("vid-steps-slider").value = recSteps;
    if ($("vid-cfg")) $("vid-cfg").value = recCfg;
    if (loraRec && $("vid-lora-scale")) {
      $("vid-lora-scale").value = loraRec.scale;
    }
    updateVideoStepsSafetyNote();
  }

  function updateVideoStepsSafetyNote() {
    const noteEl = $("vid-steps-safety-note");
    const fastBtn = $("btn-steps-fast-safe");
    const slider = $("vid-steps-slider");
    const numInput = $("vid-steps");
    if (!noteEl || !numInput) return;

    const mid = $("select-video-model") ? $("select-video-model").value : "";
    const m = state.catalog.find((x) => x.id === mid);
    const spec = (m && m.official_spec) || {};
    const lid = $("vid-lora-select") ? $("vid-lora-select").value : "";
    const loraObj = lid ? state.catalog.find((x) => x.id === lid) : null;

    const steps = Number(numInput.value || 45);
    const cfg = Number(($("vid-cfg") && $("vid-cfg").value) || 5.0);
    const scale = Number(($("vid-lora-scale") && $("vid-lora-scale").value) || 1.0);
    if (slider) slider.value = Math.max(4, Math.min(55, steps));
    syncLoraQualityPills();

    const fastSafe = Number(spec.fast_safe_steps || 28);
    const minSafe = Number(spec.min_safe_steps || 22);
    const defSteps = Number(spec.default_steps || (m && m.default_steps) || 45);

    if (fastBtn) {
      if (loraObj || defSteps <= 10) {
        fastBtn.hidden = true;
      } else {
        fastBtn.hidden = false;
        fastBtn.textContent = `⚡ Rápido seguro (${fastSafe})`;
      }
    }

    if (loraObj) {
      const dlBtn = $("btn-dl-vid-lora");
      if (dlBtn) dlBtn.hidden = Boolean(loraObj.downloaded);
      if (steps <= 12) {
        const isHeavy14b = mid.includes("14b") || mid.includes("13b");
        const isMid5b = mid.includes("5b");
        const secPerStep = isHeavy14b
          ? cfg > 1.01
            ? 27.0
            : 15.5
          : isMid5b
          ? cfg > 1.01
            ? 24.0
            : 13.5
          : cfg > 1.01
          ? 5.5
          : 3.4;
        const estDiffMin = Math.max(0.4, Math.round((steps * secPerStep) / 6.0) / 10);
        noteEl.className = "vid-steps-note turbo";
        noteEl.textContent =
          state.videoSubmode === "flf2v"
            ? `⚡ LoRA Bidireccional Frame a Frame (${steps} pasos · Fuerza ${scale} · CFG ${cfg}): interpola Inicio → Fin en ~${estDiffMin} min en L4 manteniendo visibilidad de ambos fotogramas.`
            : cfg > 1.01
            ? `🎯 LoRA + Fidelidad Extra (${steps} pasos · Fuerza ${scale} · CFG ${cfg}): difusión en ~${estDiffMin} min en L4 con mayor nitidez y obediencia al prompt/pose.`
            : `⚡ LoRA Acelerador activo (${steps} pasos · Fuerza ${scale} · CFG ${cfg}): difusión en ~${estDiffMin} min en GPU L4 sin deformar.`;
      } else {
        noteEl.className = "vid-steps-note warn";
        noteEl.textContent = `⚠️ Con el LoRA Acelerador activo lo ideal son 6–10 pasos (Fuerza 0.65–1.0). Pulsa «↺ Resetear pasos» o elige un modo arriba.`;
      }
      return;
    }

    if (defSteps <= 10) {
      noteEl.className = "vid-steps-note turbo";
      noteEl.textContent = `⚡ Modelo ya destilado de fábrica (${steps} pasos · CFG ${cfg}): velocidad máxima nativa (~45 s – 1 min en L4 sin necesitar LoRA externo).`;
      return;
    }

    if (steps < minSafe) {
      noteEl.className = "vid-steps-note warn";
      noteEl.textContent = `⚠️ Menos de ${minSafe} pasos sin LoRA puede deformar piernas/brazos o dejar borrones. Usa ≥${fastSafe} pasos o activa arriba un LoRA Bidireccional de 6–8 pasos.`;
    } else if (state.videoSubmode === "flf2v") {
      const is14b = mid.includes("14b");
      noteEl.className = "vid-steps-note ok";
      noteEl.textContent = is14b
        ? `🎞️ Modo 14B Frame a Frame oficial (${steps} pasos · CFG ${cfg}): máxima calidad de transición; para producir ×5 más rápido (~1.6 min en L4) activa arriba el LoRA FusionX o LightX2V.`
        : `🎞️ Modo Frame a Frame bidireccional (${steps} pasos · CFG ${cfg}): interpola de forma continua entre ambas poses (~1.2 min base, o ~25 s activando arriba un LoRA Bidireccional 1.3B).`;
    } else if (steps < defSteps - 4) {
      const savedPct = Math.max(5, Math.round((1 - steps / defSteps) * 100));
      noteEl.className = "vid-steps-note ok";
      noteEl.textContent = `✅ Zona rápida segura (${steps} pasos): ahorras ~${savedPct}% de tiempo respecto a ${defSteps} pasos sin deformar el movimiento.`;
    } else {
      noteEl.className = "vid-steps-note ok";
      noteEl.textContent = `✅ Valor oficial predeterminado (${steps} pasos): máxima precisión de anatomía y movimiento recomendada por el fabricante.`;
    }
  }

  function applyVideoModelDefaults(modelId) {
    const m = state.catalog.find((x) => x.id === modelId);
    if (!m) return;
    if (m.engine === "ffmpeg_motion") {
      setVideoSubmode("ffmpeg");
      $("vid-width").value = 1280;
      $("vid-height").value = 720;
      $("vid-fps").value = 24;
      $("vid-duration").value = 4.0;
      return;
    }
    if (state.videoSubmode === "ffmpeg") {
      const modes = Array.isArray(m.video_modes) ? m.video_modes : ["i2v"];
      state.videoMode = "ai";
      state.videoSubmode = modes.includes("i2v") ? "i2v" : modes[0] || "t2v";
    }
    const spec = m.official_spec || {};
    updateModelPresetsAndSpecBar(m);
    populateVideoLoraSelect(modelId);

    const curW = Number($("vid-width").value || 1280);
    const curH = Number($("vid-height").value || 704);
    const wasPortrait = curH > curW * 1.08;
    const targetRes = spec.fixed_resolution
      ? spec.fixed_resolution
      : wasPortrait
      ? spec.portrait || [704, 1280]
      : spec.landscape || [1280, 704];
    $("vid-width").value = targetRes[0];
    $("vid-height").value = targetRes[1];

    if (spec.default_duration_sec || m.default_duration_sec) {
      $("vid-duration").value = spec.default_duration_sec || m.default_duration_sec;
    }
    if (spec.default_fps || m.default_fps) {
      $("vid-fps").value = spec.default_fps || m.default_fps;
    }
    resetVideoStepsToRecommended();
    syncVideoPrepareButtonLabel();
    syncVideoSubmodeUI();
  }

  function populateVideoModelSelect() {
    const sel = $("select-video-model");
    const cardsBox = $("video-model-cards");
    const prev = sel.value;
    let vModels = state.catalog.filter((m) => m.category === "video" && m.validation_state !== "candidate");
    if (state.vidOnlyCompat) {
      const runnable = vModels.filter((m) => m.compatibility && m.compatibility.compatible);
      if (runnable.length > 0) vModels = runnable;
    }

    sel.innerHTML = vModels
      .map((m) => {
        const st = m.loaded_in_memory
          ? "🟢 En VRAM"
          : m.builtin_no_download
          ? "⚡ 0 GB"
          : m.downloaded
          ? "💾 En disco"
          : `⬇ ${m.estimated_gb} GB`;
        const stk = (m.compatibility && m.compatibility.sticker) || "";
        return `<option value="${escapeHtml(m.id)}">${stk} · ${escapeHtml(m.name)} [${st}]</option>`;
      })
      .join("");

    if (prev && vModels.some((m) => m.id === prev)) {
      sel.value = prev;
    } else if (vModels.length > 0) {
      sel.value = vModels[0].id;
    }

    const activeId = sel.value;
    const curSub = state.videoSubmode;
    if (cardsBox) {
      cardsBox.innerHTML = vModels
        .map((m) => {
          const comp = m.compatibility || {};
          const tier = comp.tier || "good";
          const modes = formatVideoModes(m.video_modes)
            .map(
              (mo) =>
                `<span class="vmc-mode-chip ${mo.key === curSub ? "highlight" : ""}">${escapeHtml(mo.label)}</span>`
            )
            .join("");
          const diskTxt = m.builtin_no_download
            ? "0 GB (integrado)"
            : m.loaded_in_memory
            ? "🟢 Activo en VRAM"
            : m.downloaded
            ? `💾 En disco (${m.actual_disk_gb || m.estimated_gb} GB)`
            : `☁️ ${m.estimated_gb} GB`;
          return `
            <div class="vm-card ${m.id === activeId ? "selected" : ""}" data-vmid="${escapeHtml(m.id)}">
              <div class="vm-card-top">
                <div class="vm-card-title">${escapeHtml(m.name)}</div>
              </div>
              <div class="vm-card-meta">
                <span class="gpu-sticker ${escapeHtml(tier)}">${escapeHtml(comp.label || comp.sticker || "")}</span>
                <span>· ${escapeHtml(diskTxt)}</span>
              </div>
              <div class="vmc-modes-row">${modes}</div>
            </div>
          `;
        })
        .join("");

      cardsBox.querySelectorAll(".vm-card").forEach((card) => {
        card.addEventListener("click", () => {
          sel.value = card.dataset.vmid;
          cardsBox.querySelectorAll(".vm-card").forEach((c) => c.classList.remove("selected"));
          card.classList.add("selected");
          applyVideoModelDefaults(sel.value);
        });
      });
    }
    const curSelVid = $("select-video-model") ? $("select-video-model").value : "";
    if (curSelVid) {
      populateVideoLoraSelect(curSelVid);
      updateVideoStepsSafetyNote();
    }
    syncVideoPrepareButtonLabel();
    renderVideoCompatBanner();
  }

  function renderCatalog() {
    const grid = $("catalog-grid");
    const filtered = state.catalog.filter((m) => {
      if (state.catOnlyCompat && !(m.hardware_assessment || m.compatibility || {}).compatible) return false;
      if (state.catalogFilter === "all") return true;
      if (state.catalogFilter === "image") {
        return m.category === "image" || (m.category === "lora" && (m.lora_target || "image") === "image");
      }
      if (state.catalogFilter === "video") {
        return m.category === "video" || (m.category === "lora" && m.lora_target === "video");
      }
      return m.category === state.catalogFilter;
    });

    if (filtered.length === 0) {
      grid.innerHTML = `<div class="muted" style="padding:14px;">No hay modelos en esta categoría para el filtro de GPU seleccionado. Desmarca «Ocultar los que no caben» o cambia de GPU arriba.</div>`;
      return;
    }

    grid.innerHTML = filtered
      .map((m) => {
        const comp = m.hardware_assessment || m.compatibility || {};
        const tier = comp.tier || "good";
        const diskLabel = m.builtin_no_download
          ? "Integrado (0 GB)"
          : m.downloaded
          ? `En disco: ${m.actual_disk_gb || m.estimated_gb} GB`
          : `Peso descarga: ~${m.estimated_gb} GB (Libre: ${m.free_disk_gb} GB)`;

        const modeTags = formatVideoModes(m.video_modes)
          .map((mo) => `<span class="mc-tag">${escapeHtml(mo.label)}</span>`)
          .join("");

        const matrixHtml = Array.isArray(m.gpu_matrix)
          ? `<div class="gpu-mini-matrix">` +
            m.gpu_matrix
              .map(
                (gm) =>
                  `<span class="gmm-chip ${escapeHtml(gm.tier)}" title="${escapeHtml(gm.gpu_short)}: ${escapeHtml(gm.sticker)}">${escapeHtml(gm.gpu_short)}: ${escapeHtml(gm.sticker)}</span>`
              )
              .join("") +
            `</div>`
          : "";

        let btns = "";
        if (m.validation_state === "candidate") {
          btns = `<span class="mc-tag">Candidato sin receta · descarga/carga bloqueadas</span>`;
        } else if (m.category === "lora") {
          if (!m.downloaded) {
            btns += `<button class="btn btn-primary btn-sm" data-dl="${escapeHtml(m.id)}">⬇ Descargar LoRA (${m.estimated_gb} GB)</button>`;
          } else {
            btns += `<span class="mc-tag ok">✅ Listo en disco (actívalo en ${m.lora_target === "video" ? "2. Vídeo" : "1. Imagen"})</span>`;
            btns += `<button class="btn btn-danger btn-sm" data-del="${escapeHtml(m.id)}">🗑 Borrar de disco</button>`;
          }
        } else if (!m.builtin_no_download) {
          if (!m.downloaded) {
            btns += `<button class="btn btn-secondary btn-sm" data-dl="${escapeHtml(m.id)}">⬇ Descargar a disco</button>`;
            btns += `<button class="btn btn-primary btn-sm" data-dl-load="${escapeHtml(m.id)}">⚡ Descargar + Cargar</button>`;
          } else {
            if (!m.loaded_in_memory) {
              btns += `<button class="btn btn-primary btn-sm" data-load="${escapeHtml(m.id)}">⚡ Cargar en RAM/VRAM</button>`;
            } else {
              btns += `<button class="btn btn-secondary btn-sm" data-unload="${escapeHtml(m.id)}">🧹 Liberar de RAM/VRAM</button>`;
            }
            btns += `<button class="btn btn-danger btn-sm" data-del="${escapeHtml(m.id)}">🗑 Borrar de disco</button>`;
          }
        } else {
          btns += `<span class="mc-tag ok">Integrado · comprobar runtime/red y RAM</span>`;
        }

        btns += `<button type="button" class="btn btn-secondary btn-sm" data-model-inspect="${escapeHtml(m.id)}">Ver diagnóstico</button>`;
        return `
          <div class="model-card ${m.is_default ? "is-default" : ""}">
            <div>
              <div class="mc-badges">
                <span class="mc-tag ${m.is_default ? "primary" : ""}">${escapeHtml(m.badge || m.category)}</span>
                ${modeTags}
              </div>
              <div style="margin:6px 0;">
                <span class="gpu-sticker ${escapeHtml(tier)}">${escapeHtml(comp.label || comp.sticker || "")}</span>
              </div>
              <h4 class="mc-title">${escapeHtml(m.name)}</h4>
              <p class="mc-desc">${escapeHtml(m.description || "")}</p>
              <p class="muted">${escapeHtml(modelResourceSummary(comp))}</p>
              ${matrixHtml ? `<span class="muted">Matriz orientativa de VRAM (no rendimiento ni RAM/disco)</span>${matrixHtml}` : ""}
            </div>
            <div>
              <div class="mc-stats">
                <span>${escapeHtml(diskLabel)}</span>
                <span>${m.loaded_in_memory ? "🟢 Cargado en VRAM" : "⚪ Fuera de VRAM"}</span>
              </div>
              <div class="mc-actions" style="margin-top:8px;">${btns}</div>
            </div>
          </div>
        `;
      })
      .join("");

    grid.querySelectorAll("[data-model-inspect]").forEach((button) => button.addEventListener("click", () => {
      showModelInspection(null, button.dataset.modelInspect);
      $("hf-inspection").scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    grid.querySelectorAll("[data-dl]").forEach((b) =>
      b.addEventListener("click", () => downloadModel(b.dataset.dl))
    );
    grid.querySelectorAll("[data-dl-load]").forEach((b) =>
      b.addEventListener("click", () => loadModelIntoMemory(b.dataset.dlLoad, true))
    );
    grid.querySelectorAll("[data-load]").forEach((b) =>
      b.addEventListener("click", () => loadModelIntoMemory(b.dataset.load, false))
    );
    grid.querySelectorAll("[data-unload]").forEach((b) =>
      b.addEventListener("click", () => unloadFromMemory(b.dataset.unload))
    );
    grid.querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", () => deleteModelFromDisk(b.dataset.del))
    );
  }

  function renderHfAuthStatus(hfAuth) {
    if (!hfAuth) return;
    state.hfAuth = hfAuth;
    const topBtn = $("btn-open-hf-token");
    const boxBtn = $("btn-hf-box-token");
    const statusBox = $("hf-token-status-box");
    const who = hfAuth.username ? ` (@${hfAuth.username})` : "";
    if (hfAuth.has_token) {
      if (topBtn) topBtn.textContent = `🟢 HF Conectado${who}`;
      if (boxBtn) boxBtn.textContent = `🟢 Token Activo${who}`;
      if (statusBox) {
        statusBox.textContent = `✅ Token activo en esta sesión: ${hfAuth.masked_token || "hf_***"}${who}`;
      }
    } else {
      if (topBtn) topBtn.textContent = "🔑 Token HF";
      if (boxBtn) boxBtn.textContent = "🔑 Token HF (opcional)";
      if (statusBox) {
        statusBox.textContent = "Estado actual: Sin token activo (modo público sin registro)";
      }
    }
  }

  function formatModelNumber(value, suffix = " GiB") {
    return value == null || !Number.isFinite(Number(value)) ? "desconocido" : `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`;
  }

  function modelResourceSummary(comp) {
    const resources = (comp && comp.resources) || {};
    return [
      `VRAM ≥ ${formatModelNumber((resources.vram || {}).required_gb)}`,
      `RAM ≥ ${formatModelNumber((resources.ram || {}).required_gb)}`,
      `Descarga + margen: ${formatModelNumber((resources.disk || {}).download_budget_gb)}`,
    ].join(" · ");
  }

  function renderDiscoveryStatus(status) {
    if (!status) return;
    state.modelDiscovery = status;
    const auto = $("hf-auto-refresh");
    const interval = $("hf-refresh-interval");
    if (auto) auto.checked = Boolean((status.settings || {}).auto_refresh);
    if (interval && document.activeElement !== interval) {
      const minutes = String((status.settings || {}).interval_minutes || 360);
      if (!Array.from(interval.options).some((option) => option.value === minutes)) interval.add(new Option(`Cada ${minutes} minutos`, minutes));
      interval.value = minutes;
    }
    const box = $("hf-index-status");
    if (!box) return;
    const date = status.last_success_at ? new Date(status.last_success_at).toLocaleString() : "sin consulta HF confirmada";
    const error = status.last_error ? ` · ${status.last_error.message}` : "";
    const retry = status.retry_after_seconds > 0 ? ` · Reintento en ${status.retry_after_seconds} s` : "";
    const running = status.refresh_running ? " · Actualizando metadatos…" : "";
    box.textContent = `${status.indexed_repositories || 0} repos indexados · ${status.research_repositories || 0} referencias documentales · Último éxito: ${date}${running}${retry}${error}`;
  }

  async function refreshDiscoveryStatus() {
    try {
      const previousSuccess = state.modelDiscovery && state.modelDiscovery.last_success_at;
      const status = await api("/api/models/discovery/status");
      renderDiscoveryStatus(status);
      if (previousSuccess && status.last_success_at !== previousSuccess && state.hfSearchResults.length && !state.hfSearching && $("tab-catalog").classList.contains("active")) {
        runHfLiveSearch(null, null, state.hfOffset);
      }
    } catch (_error) {
      if ($("hf-index-status")) $("hf-index-status").textContent = "No se pudo consultar el estado del índice. Se conservan los resultados visibles.";
    }
  }

  function selectRecipeInUI(modelId) {
    const model = state.catalog.find((m) => m.id === modelId);
    if (!model || model.validation_state === "candidate") return;
    if (model.category === "image") {
      $("select-image-model").value = modelId;
      applyImageModelDefaults(modelId);
      renderImageLoraRack();
      switchTab("tab-image");
    } else if (model.category === "video") {
      const sel = $("select-video-model");
      // Selection is NOT preparation. A hidden-by-filter recipe is not loaded.
      if (!Array.from(sel.options).some((option) => option.value === modelId)) {
        state.vidOnlyCompat = false;
        populateVideoModelSelect();
        syncGpuPillsUI();
      }
      sel.value = modelId;
      applyVideoModelDefaults(modelId);
      switchTab("tab-video");
    } else if (model.category === "tts") {
      $("tts-engine-select").value = modelId;
      switchTab("tab-tts");
    } else {
      switchTab("tab-catalog");
    }
    toast(`Receta «${modelId}» seleccionada; no se han descargado ni cargado pesos. Revisa su diagnóstico.`, "info");
  }

  async function showModelInspection(repoId, localModelId = null) {
    const panel = $("hf-inspection");
    if (!panel) return;
    panel.hidden = false;
    const body = $("hf-inspection-body");
    body.textContent = "Consultando diagnóstico y pequeños metadatos; sin pesos…";
    try {
      let item;
      let response;
      if (localModelId) {
        response = await api(`/api/models/${encodeURIComponent(localModelId)}/assessment?gpu_profile=${encodeURIComponent(state.gpuProfile)}`);
        const definition = state.catalog.find((m) => m.id === localModelId) || {};
        item = { repo_id: repoId || localModelId, compatibility: response.assessment, source: "studio_catalog", license: definition.license,
          recipes: [{ model_id: localModelId, engine: definition.engine }], source_urls: definition.source_url ? [definition.source_url] : [] };
      } else {
        response = await api(`/api/models/discovery/inspect?repo_id=${encodeURIComponent(repoId)}&gpu_profile=${encodeURIComponent(state.gpuProfile)}`);
        item = response.item || {};
        renderDiscoveryStatus(response.discovery_status);
      }
      const comp = item.compatibility || {};
      const reasons = [...(comp.reasons || []), ...(comp.warnings || [])];
      const source = item.source_urls && item.source_urls[0];
      const safeSource = source && /^https:\/\/(github\.com|huggingface\.co)\//.test(source) ? source : null;
      body.innerHTML = `
        <h4>${escapeHtml(item.repo_id || localModelId || repoId)}</h4>
        <p><span class="gpu-sticker ${escapeHtml(comp.tier || "candidate")}">${escapeHtml(comp.label || comp.sticker || "Sin validar")}</span></p>
        <p class="muted">${escapeHtml(modelResourceSummary(comp))}</p>
        <p class="muted">Pipeline: ${escapeHtml(item.pipeline_class || "ver receta / aún no inspeccionado")} · Licencia: ${escapeHtml(item.license || "desconocida")} · Revisión: ${escapeHtml(item.revision || "no consultada")}</p>
        <p class="muted">Fuente: ${escapeHtml(item.source || "sin datos")} · Metadatos: ${escapeHtml(item.metadata_checked_at || "sin consulta HF confirmada")} · Acceso a pesos no verificado.</p>
        ${reasons.length ? `<ul>${reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>` : ""}
        <p class="muted">Encaje estimado, no inferencia GPU validada. Archivo/tensores ≠ VRAM de ejecución. Las importaciones sin receta no son ejecutables.</p>
        <div class="row-gap discovery-actions">
          ${(item.recipes || []).map((recipe) => `<button type="button" class="btn btn-secondary btn-sm" data-pick-recipe="${escapeHtml(recipe.model_id)}">Elegir ${escapeHtml(recipe.model_id)}</button>`).join("")}
          ${safeSource ? `<a class="btn btn-secondary btn-sm" href="${escapeHtml(safeSource)}" target="_blank" rel="noopener noreferrer">Fuente oficial ↗</a>` : ""}
        </div>
        <details class="discovery-json"><summary>Diagnóstico completo (JSON)</summary><pre>${escapeHtml(JSON.stringify(response, null, 2))}</pre></details>`;
      body.querySelectorAll("[data-pick-recipe]").forEach((button) => button.addEventListener("click", () => selectRecipeInUI(button.dataset.pickRecipe)));
    } catch (error) {
      body.textContent = `No se pudo inspeccionar: ${error.message}. No se han descargado pesos.`;
    }
  }

  let _hfDebounceTimer = null;
  let _hfAbortCtrl = null;

  function scheduleRealtimeHfSearch() {
    if (_hfDebounceTimer) clearTimeout(_hfDebounceTimer);
    _hfDebounceTimer = setTimeout(() => runHfLiveSearch(), 280);
  }

  async function runHfLiveSearch(customQuery = null, customCat = null, offset = 0, refresh = false) {
    const qInput = $("hf-live-query");
    const cSelect = $("hf-live-cat");
    const box = $("hf-live-results");
    const btn = $("btn-hf-live-search");
    const ind = $("hf-live-indicator");
    if (!box) return;
    if (customQuery != null && qInput) qInput.value = customQuery;
    if (customCat != null && cSelect) cSelect.value = customCat;
    const q = (qInput && qInput.value.trim()) || "";
    const cat = (cSelect && cSelect.value) || "video";
    if (_hfAbortCtrl) _hfAbortCtrl.abort();
    const ctrl = new AbortController();
    _hfAbortCtrl = ctrl;
    state.hfSearching = true;
    if (ind) {
      ind.classList.add("searching");
      ind.textContent = "Consultando metadatos…";
    }
    if (btn) btn.disabled = true;
    try {
      const sort = ($("hf-sort") && $("hf-sort").value) || "downloads";
      const params = new URLSearchParams({ query: q, category: cat, gpu_profile: state.gpuProfile, only_compatible: String(state.catOnlyCompat), limit: "12", offset: String(offset), sort, refresh: String(refresh) });
      const resp = await fetch(`/api/catalog/hf-search?${params}`, { signal: ctrl.signal });
      const data = await resp.json();
      if (!resp.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Consulta no válida");
      if (ctrl.signal.aborted) return;
      const items = data.items || [];
      state.hfSearchResults = items;
      state.hfOffset = offset;
      state.hfNextOffset = data.next_offset;
      renderDiscoveryStatus(data.discovery_status);
      if (data.source_mode === "refreshing") {
        setTimeout(() => {
          if (_hfAbortCtrl === ctrl && !ctrl.signal.aborted) runHfLiveSearch(null, null, offset);
        }, 1500);
      }
      const modes = { network: "HF consultado ahora", cache: "Caché", offline: "Sin conexión · local/caché", refreshing: "Consulta ya en curso" };
      if (ind) {
        ind.classList.remove("searching");
        ind.textContent = `${modes[data.source_mode] || "Metadatos"}${data.stale ? " · sin consulta reciente" : ""}`;
      }
      if ($("hf-search-meta")) {
        const date = data.metadata_checked_at ? new Date(data.metadata_checked_at).toLocaleString() : "sin consulta HF confirmada";
        $("hf-search-meta").textContent = `${data.total_local_matches || 0} coincidencias locales/indexadas · Fecha de consulta: ${date}${data.active_gpu_profile && data.active_gpu_profile.is_simulated ? " · SIMULACIÓN SOLO DE GPU; RAM/disco reales" : ""}${data.query_limit_reached ? " · Límite de cobertura alcanzado; concreta la búsqueda" : ""}`;
      }
      if ($("btn-hf-prev")) $("btn-hf-prev").disabled = offset <= 0;
      if ($("btn-hf-next")) $("btn-hf-next").disabled = data.next_offset == null;
      if ($("hf-page-label")) $("hf-page-label").textContent = `Página ${Math.floor(offset / 12) + 1}`;
      box.innerHTML = items.length ? items.map((item, index) => {
        const comp = item.compatibility || {};
        const support = (item.support || {}).status || "candidate";
        const supportLabels = { reviewed_recipe: "Receta revisada · no benchmark", candidate: "Candidato sin receta", unsupported: "Backend/formato no soportado" };
        const access = (item.access || {}).status;
        const gate = access === "requires_acceptance" ? "🔒 Aceptar licencia / comprobar permiso" : access === "public_metadata" ? "Metadatos públicos; permiso de pesos no verificado" : "Acceso/licencia sin confirmar";
        const popularity = item.downloads == null ? "Descargas: desconocidas" : `⬇ ${Number(item.downloads).toLocaleString()}`;
        const issue = (comp.reasons || [])[0] || (comp.warnings || [])[0] || "Receta con estimaciones; revisar valores predeterminados y memoria actual.";
        const known = (item.recipes || [])[0];
        return `<article class="hf-live-card">
          <div><div class="hf-card-heading"><strong>${escapeHtml(item.repo_id)}</strong><span class="muted">${escapeHtml(popularity)}</span></div>
          <div class="row-gap discovery-tags"><span class="mc-tag">${escapeHtml(supportLabels[support] || support)}</span><span class="gpu-sticker ${escapeHtml(comp.tier || "candidate")}">${escapeHtml(comp.sticker || "Sin validar")}</span></div>
          <p class="muted">${escapeHtml(modelResourceSummary(comp))}</p>
          <p class="muted">${escapeHtml(issue)}</p>
          <p class="muted">${escapeHtml(gate)} · Licencia: ${escapeHtml(item.license || "desconocida")}</p>
          <p class="muted">Fuente: ${escapeHtml(item.source || "local")} · ${escapeHtml(item.metadata_checked_at || "referencia sin consulta HF actual")}</p></div>
          <div class="row-gap discovery-actions">
            <button type="button" class="btn btn-secondary btn-sm" data-hf-inspect="${index}">Inspeccionar</button>
            ${known ? `<button type="button" class="btn btn-secondary btn-sm" data-pick-recipe="${escapeHtml(known.model_id)}">Elegir receta</button>` : item.category === "unknown" ? `<span class="muted">Tarea sin clasificar; inspeccionar antes de guardar</span>` : `<button type="button" class="btn btn-secondary btn-sm" data-save-candidate="${index}">Guardar candidato</button>`}
          </div></article>`;
      }).join("") : `<div class="muted discovery-empty">Sin coincidencias para estos filtros. «Solo recetas compatibles» también comprueba RAM/disco: simular una L4 no aumenta la RAM real. Desmarca el filtro para investigar candidatos sin receta.</div>`;
      box.querySelectorAll("[data-hf-inspect]").forEach((button) => button.addEventListener("click", () => showModelInspection(items[Number(button.dataset.hfInspect)].repo_id)));
      box.querySelectorAll("[data-pick-recipe]").forEach((button) => button.addEventListener("click", () => selectRecipeInUI(button.dataset.pickRecipe)));
      box.querySelectorAll("[data-save-candidate]").forEach((button) => button.addEventListener("click", async () => {
        const item = items[Number(button.dataset.saveCandidate)];
        try {
          const saved = await api("/api/catalog/custom", { method: "POST", body: JSON.stringify({ name: item.name || item.repo_id, repo_id: item.repo_id, category: item.category, estimated_gb: 0 }) });
          toast(saved.message || "Candidato guardado; sin descarga ni carga de pesos.", "info");
          await loadCatalog();
        } catch (error) { toast(error.message, "error"); }
      }));
    } catch (error) {
      if (error && error.name === "AbortError") return;
      if (ind) {
        ind.classList.remove("searching");
        ind.textContent = "Error de consulta · no en vivo";
      }
      box.textContent = `No se pudo consultar el índice: ${error.message}`;
    } finally {
      if (_hfAbortCtrl === ctrl) {
        state.hfSearching = false;
        if (btn) btn.disabled = false;
      }
    }
  }

  async function updateDiscoverySettings() {
    try {
      renderDiscoveryStatus(await api("/api/models/discovery/settings", { method: "POST", body: JSON.stringify({ auto_refresh: $("hf-auto-refresh").checked, interval_minutes: Number($("hf-refresh-interval").value) }) }));
      toast("Actualizador de metadatos configurado. No descarga pesos.", "success");
    } catch (error) { toast(error.message, "error"); }
  }

  async function runHardwareOracle() {
    const box = $("hf-oracle-result");
    box.textContent = "Evaluando el hardware real, sin pesos…";
    try {
      const params = new URLSearchParams({ task: $("hf-oracle-task").value, priority: $("hf-oracle-priority").value });
      const result = await api(`/api/models/recommend?${params}`);
      box.innerHTML = `<p><strong>${escapeHtml(result.model_id || "Sin receta apta")}</strong> · ${escapeHtml(result.status)}</p>
        <p>${escapeHtml(result.why_this_choice || "")}</p>
        <p class="muted">${escapeHtml(result.detected_gpu || "")} · ${escapeHtml(modelResourceSummary(result.assessment))}</p>
        <ul>${(result.next_actions || []).map((action) => `<li>${escapeHtml(action)}</li>`).join("")}</ul>
        ${result.model_id ? `<button type="button" class="btn btn-secondary btn-sm" id="hf-oracle-pick">Elegir receta (sin cargar)</button>` : ""}
        <details class="discovery-json"><summary>Plantilla MCP y diagnóstico completo</summary><pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre></details>`;
      if ($("hf-oracle-pick")) $("hf-oracle-pick").addEventListener("click", () => selectRecipeInUI(result.model_id));
    } catch (error) { box.textContent = error.message; }
  }

  async function downloadModel(modelId) {
    try {
      const res = await api(`/api/catalog/${modelId}/download`, { method: "POST" });
      toast("Descarga iniciada. Mira el porcentaje y velocidad a la derecha.");
      watchJob(res.job);
    } catch (e) {
      if (e.payload && typeof e.payload === "object") {
        showSpaceErrorBanner(e.payload);
      } else {
        toast(e.message, "error");
      }
    }
  }

  async function loadModelIntoMemory(modelId, autoDownload = false) {
    try {
      const extraLoras = [];
      if ($("chk-nsfw-lora") && $("chk-nsfw-lora").checked) {
        extraLoras.push("qwen-lora-nsfw");
      }
      const res = await api("/api/memory/load", {
        method: "POST",
        body: JSON.stringify({
          model_id: modelId,
          auto_download: autoDownload,
          extra_lora_ids: extraLoras,
        }),
      });
      toast("Cargando modelo en RAM/VRAM...");
      watchJob(res.job);
    } catch (e) {
      if (e.payload && typeof e.payload === "object") {
        showSpaceErrorBanner(e.payload);
      } else {
        toast(e.message, "error");
      }
    }
  }

  async function unloadFromMemory(target = "all") {
    try {
      const res = await api("/api/memory/unload", {
        method: "POST",
        body: JSON.stringify({ target }),
      });
      toast(res.message, "success");
      await Promise.all([loadCatalog(), pollStatus()]);
    } catch (e) {
      toast(e.message, "error");
    }
  }

  async function deleteModelFromDisk(modelId) {
    const def = state.catalog.find((m) => m.id === modelId);
    const name = def ? def.name : modelId;
    if (!confirm(`¿Seguro que quieres borrar «${name}» del disco para liberar espacio?`)) {
      return;
    }
    try {
      const res = await api(`/api/catalog/${modelId}`, { method: "DELETE" });
      toast(res.message, "success");
      await Promise.all([loadCatalog(), pollStatus()]);
    } catch (e) {
      toast(e.message, "error");
    }
  }

  // ==================== GESTIÓN DE ACTIVOS Y SUBIDAS ====================
  async function uploadFile(file) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await api("/api/assets/upload", { method: "POST", body: fd });
    await loadAssets();
    return res.asset;
  }

  async function loadAssets() {
    if (state.assetsLoading) return;
    state.assetsLoading = true;
    try {
      await _loadAssetsOnce();
    } finally {
      state.assetsLoading = false;
    }
  }

  async function _loadAssetsOnce() {
    state.assets = await api("/api/assets");
    renderImageGallery();
    renderVideoSelectsAndGallery();
    renderAudioList();
    renderLipSyncSelects();
    renderHiggsfieldAssetSelects();
    renderMontagePickers();
    renderMontageGallery();
  }

  function renderLipSyncSelects() {
    const medSel = $("lipsync-media-select");
    const voxSel = $("lipsync-voice-select");
    const ambSel = $("lipsync-ambient-select");
    const musSel = $("lipsync-music-select");
    if (!medSel || !voxSel) return;

    const visualPool = [
      ...state.assets.videos,
      ...state.assets.images,
      ...state.assets.uploads.filter((u) => u.type === "image" || u.type === "video"),
    ];
    const audioPool = [
      ...state.assets.audio,
      ...state.assets.uploads.filter((u) => u.type === "audio"),
    ];

    const prevMed = medSel.value;
    medSel.innerHTML =
      `<option value="">-- Elige un vídeo MP4 o imagen retrato --</option>` +
      visualPool
        .map(
          (a) =>
            `<option value="${escapeHtml(a.rel_path)}">[${a.type === "video" ? "VÍDEO" : "IMAGEN"}] ${escapeHtml(a.filename)}</option>`
        )
        .join("");
    if (prevMed) medSel.value = prevMed;

    const audioOpts = audioPool
      .map(
        (a) =>
          `<option value="${escapeHtml(a.rel_path)}">${escapeHtml(a.filename)}${a.duration_seconds ? ` (${a.duration_seconds}s)` : ""}</option>`
      )
      .join("");
    const prevVox = voxSel.value;
    voxSel.innerHTML = `<option value="">-- Elige locución de voz --</option>` + audioOpts;
    if (prevVox) voxSel.value = prevVox;

    if (ambSel) {
      const prevAmb = ambSel.value;
      ambSel.innerHTML = `<option value="">(Sin ambiente de fondo)</option>` + audioOpts;
      if (prevAmb) ambSel.value = prevAmb;
    }
    if (musSel) {
      const prevMus = musSel.value;
      musSel.innerHTML = `<option value="">(Sin música de fondo)</option>` + audioOpts;
      if (prevMus) musSel.value = prevMus;
    }
  }

  function setBaseImage(asset) {
    state.baseImage = asset;
    clearMaskCanvas();
    if (!asset) {
      $("canvas-empty-hint").hidden = false;
      $("canvas-wrap").hidden = true;
      return;
    }
    const imgEl = $("base-preview-img");
    imgEl.onload = () => {
      const nw = imgEl.naturalWidth || 1024;
      const nh = imgEl.naturalHeight || 1024;
      const cvs = $("mask-canvas");
      cvs.width = nw;
      cvs.height = nh;
      state.mask.history = [];
      state.mask.dirty = false;

      // Ajustar automáticamente Ancho × Alto a la proporción real de la foto subida (múltiplos de 64, máx 1024)
      const scale = 1024 / Math.max(nw, nh);
      const snap64 = (v) => Math.max(512, Math.min(1280, Math.round((v * scale) / 64) * 64));
      $("img-width").value = snap64(nw);
      $("img-height").value = snap64(nh);
    };
    imgEl.src = asset.media_url;
    $("canvas-empty-hint").hidden = true;
    $("canvas-wrap").hidden = false;
  }

  function setRefSlot(slotIdx, asset) {
    state.refs[slotIdx] = asset;
    const box = $(`ref-preview-${slotIdx}`);
    if (!asset) {
      box.innerHTML = `<span>+ Ref ${slotIdx + 1}</span>`;
    } else {
      box.innerHTML = `<img src="${escapeHtml(asset.media_url)}" alt="Ref" />`;
    }
  }

  // ==================== LIENZO DE INPAINTING ====================
  function setupMaskCanvas() {
    const cvs = $("mask-canvas");
    const ctx = cvs.getContext("2d");

    function getCoords(ev) {
      const rect = cvs.getBoundingClientRect();
      const clientX = ev.touches ? ev.touches[0].clientX : ev.clientX;
      const clientY = ev.touches ? ev.touches[0].clientY : ev.clientY;
      const scaleX = cvs.width / Math.max(1, rect.width);
      const scaleY = cvs.height / Math.max(1, rect.height);
      return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
        scale: (scaleX + scaleY) / 2,
      };
    }

    function startStroke(ev) {
      if (!state.mask.enabled || !state.baseImage) return;
      ev.preventDefault();
      state.mask.history.push(ctx.getImageData(0, 0, cvs.width, cvs.height));
      if (state.mask.history.length > 15) state.mask.history.shift();
      state.mask.drawing = true;
      state.mask.dirty = true;
      const pt = getCoords(ev);
      state.mask.lastX = pt.x;
      state.mask.lastY = pt.y;
      drawSegment(pt.x, pt.y, pt.scale);
    }

    function moveStroke(ev) {
      if (!state.mask.drawing) return;
      ev.preventDefault();
      const pt = getCoords(ev);
      drawSegment(pt.x, pt.y, pt.scale);
      state.mask.lastX = pt.x;
      state.mask.lastY = pt.y;
    }

    function endStroke() {
      state.mask.drawing = false;
    }

    function drawSegment(x, y, scale) {
      const size = Number($("brush-size").value || 36) * scale;
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = size;
      if (state.mask.tool === "erase") {
        ctx.globalCompositeOperation = "destination-out";
        ctx.strokeStyle = "rgba(0,0,0,1)";
      } else {
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = "rgba(255, 48, 130, 0.62)";
      }
      ctx.beginPath();
      ctx.moveTo(state.mask.lastX, state.mask.lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.restore();
    }

    cvs.addEventListener("mousedown", startStroke);
    window.addEventListener("mousemove", moveStroke);
    window.addEventListener("mouseup", endStroke);
    cvs.addEventListener("touchstart", startStroke, { passive: false });
    cvs.addEventListener("touchmove", moveStroke, { passive: false });
    window.addEventListener("touchend", endStroke);
  }

  function clearMaskCanvas() {
    const cvs = $("mask-canvas");
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    state.mask.history = [];
    state.mask.dirty = false;
  }

  function undoMaskStroke() {
    const cvs = $("mask-canvas");
    const ctx = cvs.getContext("2d");
    const prev = state.mask.history.pop();
    if (prev) {
      ctx.putImageData(prev, 0, 0);
    } else {
      clearMaskCanvas();
    }
  }

  async function exportBinaryMaskIfPainted() {
    if (!state.mask.dirty || !state.baseImage) return null;
    const cvs = $("mask-canvas");
    const ctx = cvs.getContext("2d");
    const imgData = ctx.getImageData(0, 0, cvs.width, cvs.height);
    const d = imgData.data;

    let anyPainted = false;
    const outCanvas = document.createElement("canvas");
    outCanvas.width = cvs.width;
    outCanvas.height = cvs.height;
    const outCtx = outCanvas.getContext("2d");
    const outImg = outCtx.createImageData(cvs.width, cvs.height);
    const od = outImg.data;

    for (let i = 0; i < d.length; i += 4) {
      const alpha = d[i + 3];
      const val = alpha > 20 ? 255 : 0;
      if (val === 255) anyPainted = true;
      od[i] = val;
      od[i + 1] = val;
      od[i + 2] = val;
      od[i + 3] = 255;
    }

    if (!anyPainted) return null;
    outCtx.putImageData(outImg, 0, 0);
    const dataUrl = outCanvas.toDataURL("image/png");
    const uploaded = await api("/api/assets/upload-mask", {
      method: "POST",
      body: JSON.stringify({ data_url: dataUrl }),
    });
    return uploaded.rel_path;
  }

  // ==================== EJECUCIÓN DE IMAGEN ====================
  async function runImageGeneration(cpuBt709Only = false) {
    try {
      let maskRel = null;
      if (!cpuBt709Only) {
        maskRel = await exportBinaryMaskIfPainted();
      }
      const refRels = state.refs.filter(Boolean).map((r) => r.rel_path);
      const chosenImgModel = ($("select-image-model") && $("select-image-model").value) || "qwen-image-edit-2511-pack";
      const customLoraIds = Object.keys(state.activeImgLoras);
      const payload = {
        prompt: $("img-prompt").value,
        negative_prompt: $("img-neg-prompt").value,
        model_id: chosenImgModel,
        base_image_rel: state.baseImage ? state.baseImage.rel_path : null,
        reference_rels: refRels,
        mask_image_rel: maskRel,
        width: Number($("img-width").value || 1024),
        height: Number($("img-height").value || 1024),
        steps: Number($("img-steps").value || 4),
        guidance_scale: Number($("img-cfg").value || 4.0),
        seed: Number($("img-seed").value || 0),
        use_lightning_lora: !$("chk-lightning").disabled && $("chk-lightning").checked,
        use_nsfw_lora: !$("chk-nsfw-lora").disabled && $("chk-nsfw-lora").checked,
        nsfw_lora_weight: 1.0,
        custom_lora_ids: customLoraIds,
        lora_scales: { ...state.activeImgLoras },
        use_ffmpeg_bt709: $("chk-bt709").checked,
        mask_feather_px: Number($("mask-feather").value || 8),
        cpu_bt709_only: cpuBt709Only,
      };
      const res = await api("/api/image/generate", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      watchJob(res.job);
    } catch (e) {
      toast(e.message, "error");
    }
  }

  function showImageResult(meta) {
    state.lastImageResult = meta;
    $("img-result-empty").hidden = true;
    $("img-result-content").hidden = false;
    $("img-result-view").src = `${meta.media_url}?t=${Date.now()}`;
    $("btn-res-download").href = meta.media_url;
    $("btn-res-download").download = meta.filename || "resultado.jpg";

    $("img-provenance").innerHTML = `
      <span><strong>Modelo:</strong> ${escapeHtml(meta.model_name || meta.model_id)}</span>
      <span><strong>Modo:</strong> ${escapeHtml(meta.mode || "edición")}</span>
      <span><strong>Resolución:</strong> ${meta.width}×${meta.height}</span>
      <span><strong>Semilla:</strong> ${meta.seed}</span>
      <span><strong>Pasos:</strong> ${meta.steps}</span>
      <span><strong>Tiempo:</strong> ${meta.elapsed_seconds}s</span>
    `;
  }

  function renderImageGallery() {
    const allImgs = [
      ...state.assets.images,
      ...state.assets.uploads.filter((u) => u.type === "image"),
    ];
    const strip = $("image-gallery-strip");
    if (allImgs.length === 0) {
      strip.innerHTML = `<div class="muted">Aún no hay imágenes subidas ni generadas.</div>`;
      return;
    }
    strip.innerHTML = allImgs
      .slice(0, 18)
      .map(
        (im, idx) => `
        <div class="asset-card">
          <div class="asset-thumb"><img src="${escapeHtml(im.media_url)}" alt="" loading="lazy" /></div>
          <div class="asset-meta" title="${escapeHtml(im.prompt || im.filename)}">${escapeHtml(im.filename)}</div>
          <div class="asset-mini-btns">
            <button class="mini-btn" data-act="base" data-idx="${idx}">Base</button>
            <button class="mini-btn" data-act="ref" data-idx="${idx}">+Ref</button>
            <button class="mini-btn" data-act="vid" data-idx="${idx}" title="Enviar como Imagen Inicial de Vídeo">🎬 Inicio</button>
            <button class="mini-btn" data-act="vidend" data-idx="${idx}" title="Enviar como Fotograma Final (Frame a Frame)">🏁 Fin</button>
            <button class="mini-btn" data-act="mon" data-idx="${idx}">+Montaje</button>
          </div>
        </div>
      `
      )
      .join("");

    strip.querySelectorAll(".mini-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const asset = allImgs[Number(btn.dataset.idx)];
        if (!asset) return;
        const act = btn.dataset.act;
        if (act === "base") {
          setBaseImage(asset);
          toast(`Cargada como imagen base: ${asset.filename}`);
        } else if (act === "ref") {
          const emptySlot = state.refs.findIndex((r) => r === null);
          const targetSlot = emptySlot >= 0 ? emptySlot : 0;
          setRefSlot(targetSlot, asset);
          toast(`Añadida en Referencia ${targetSlot + 1}`);
        } else if (act === "vid") {
          switchTab("tab-video");
          if (state.videoSubmode === "t2v") setVideoSubmode("i2v");
          $("vid-source-image").value = asset.rel_path;
          updateVideoSourceThumb();
          toast(`Cargada como Imagen Inicial: ${asset.filename}`);
        } else if (act === "vidend") {
          switchTab("tab-video");
          setVideoSubmode("flf2v");
          $("vid-end-image").value = asset.rel_path;
          updateVideoEndThumb();
          toast(`Cargada como Fotograma Final (Frame a Frame): ${asset.filename}`);
        } else if (act === "mon") {
          addShotToTimeline(asset);
        }
      });
    });
  }

  // ==================== VÍDEO ====================
  function renderVideoSelectsAndGallery() {
    const allImgs = [
      ...state.assets.images,
      ...state.assets.uploads.filter((u) => u.type === "image"),
    ];
    const imgSel = $("vid-source-image");
    const prevImg = imgSel.value;
    const emptyHint =
      allImgs.length === 0
        ? "-- Pulsa «📤 Subir imagen» arriba o crea una en 1. Imagen --"
        : "-- Elige una imagen inicial --";
    imgSel.innerHTML =
      `<option value="">${emptyHint}</option>` +
      allImgs
        .map((im) => `<option value="${escapeHtml(im.rel_path)}">${escapeHtml(im.filename)}</option>`)
        .join("");
    if (prevImg && allImgs.some((i) => i.rel_path === prevImg)) {
      imgSel.value = prevImg;
    } else if (allImgs.length > 0 && !imgSel.value) {
      imgSel.value = allImgs[0].rel_path;
    }
    updateVideoSourceThumb();

    const endSel = $("vid-end-image");
    if (endSel) {
      const prevEnd = endSel.value;
      const endHint =
        allImgs.length === 0
          ? "-- Pulsa «📤 Subir final» para el Fotograma Final --"
          : "-- Elige el Fotograma Final (Destino) --";
      endSel.innerHTML =
        `<option value="">${endHint}</option>` +
        allImgs
          .map((im) => `<option value="${escapeHtml(im.rel_path)}">${escapeHtml(im.filename)}</option>`)
          .join("");
      if (prevEnd && allImgs.some((i) => i.rel_path === prevEnd)) {
        endSel.value = prevEnd;
      } else if (allImgs.length >= 2 && !endSel.value) {
        const diffImg = allImgs.find((i) => i.rel_path !== imgSel.value) || allImgs[0];
        endSel.value = diffImg.rel_path;
      }
      updateVideoEndThumb();
    }

    const allAudio = [
      ...state.assets.audio,
      ...state.assets.uploads.filter((u) => u.type === "audio"),
    ];
    const audSel = $("vid-source-audio");
    const prevAud = audSel.value;
    audSel.innerHTML =
      `<option value="">-- Sin pista de voz (mudo) --</option>` +
      allAudio
        .map(
          (a) =>
            `<option value="${escapeHtml(a.rel_path)}">${escapeHtml(a.filename)} (${a.duration_seconds || "?"}s)</option>`
        )
        .join("");
    if (prevAud) audSel.value = prevAud;

    const vStrip = $("video-gallery-strip");
    const vids = state.assets.videos;
    if (vids.length === 0) {
      vStrip.innerHTML = `<div class="muted">Todavía no hay clips generados.</div>`;
    } else {
      vStrip.innerHTML = vids
        .slice(0, 12)
        .map(
          (v, idx) => `
          <div class="asset-card">
            <div class="asset-thumb"><video src="${escapeHtml(v.media_url)}" muted preload="metadata"></video></div>
            <div class="asset-meta">${escapeHtml(v.filename)} (${v.duration_seconds || "?"}s)</div>
            <div class="asset-mini-btns">
              <button class="mini-btn" data-vplay="${idx}">Ver</button>
              <button class="mini-btn" data-vmon="${idx}">+Montaje</button>
            </div>
          </div>
        `
        )
        .join("");

      vStrip.querySelectorAll("[data-vplay]").forEach((b) =>
        b.addEventListener("click", () => showVideoResult(vids[Number(b.dataset.vplay)]))
      );
      vStrip.querySelectorAll("[data-vmon]").forEach((b) =>
        b.addEventListener("click", () => addShotToTimeline(vids[Number(b.dataset.vmon)]))
      );
    }
  }

  function updateVideoSourceThumb() {
    const sel = $("vid-source-image");
    const box = $("vid-source-preview-box");
    if (!sel || !box) return;
    const rel = sel.value;
    if (!rel) {
      box.innerHTML = `<span class="muted">Pulsa «📤 Subir imagen» arriba o elige una generada en 1. Imagen</span>`;
      return;
    }
    box.innerHTML = `<img id="vid-src-thumb-img" src="/media/${escapeHtml(rel)}" alt="Origen vídeo" />`;
    const thumbImg = $("vid-src-thumb-img");
    if (thumbImg && state.videoMode === "ai") {
      thumbImg.onload = () => {
        const nw = thumbImg.naturalWidth || 1280;
        const nh = thumbImg.naturalHeight || 704;
        const vidSel = $("select-video-model");
        const curModel = vidSel ? state.catalog.find((m) => m.id === vidSel.value) : null;
        const dims = computeClientModelDimensions(curModel, nw, nh);
        $("vid-width").value = dims.w;
        $("vid-height").value = dims.h;
      };
    }
  }

  function updateVideoEndThumb() {
    const sel = $("vid-end-image");
    const box = $("vid-end-preview-box");
    if (!sel || !box) return;
    const rel = sel.value;
    if (!rel) {
      box.innerHTML = `<span class="muted">Pulsa «📤 Subir final» o selecciona la imagen de destino</span>`;
      return;
    }
    box.innerHTML = `<img src="/media/${escapeHtml(rel)}" alt="Fotograma final" />`;
  }

  function showVideoResult(meta) {
    state.lastVideoResult = meta;
    $("vid-result-empty").hidden = true;
    $("vid-result-content").hidden = false;
    const player = $("vid-result-player");
    player.src = `${meta.media_url}?t=${Date.now()}`;
    $("btn-vid-download").href = meta.media_url;
    $("btn-vid-download").download = meta.filename;
    const subMap = {
      t2v: "Texto → Vídeo",
      i2v: "Imagen → Vídeo",
      flf2v: "Frame a Frame (Inicio → Fin)",
    };
    const subTxt = meta.video_submode ? subMap[meta.video_submode] || meta.video_submode : "Vídeo";
    const loraHtml = meta.active_lora_name
      ? `<span><strong>LoRA:</strong> ${escapeHtml(meta.active_lora_name.split("(")[0].trim())}</span>`
      : "";
    $("vid-provenance").innerHTML = `
      <span><strong>Motor:</strong> ${escapeHtml(meta.model_name || meta.model_id)}</span>
      ${loraHtml}
      <span><strong>Modo:</strong> ${escapeHtml(subTxt)}</span>
      <span><strong>Pasos / CFG:</strong> ${meta.steps || "?"}p · CFG ${meta.guidance_scale || "?"}</span>
      <span><strong>Resolución:</strong> ${meta.width}×${meta.height} @ ${meta.fps}fps (${meta.duration_seconds}s)</span>
      <span><strong>Tiempo:</strong> ${meta.elapsed_seconds}s</span>
    `;
  }

  // ==================== VOCES / TTS ====================
  async function loadVoices() {
    const data = await api("/api/tts/voices");
    state.voices = data.voices || [];
    const sel = $("tts-voice-select");
    sel.innerHTML = state.voices
      .map((v) => `<option value="${escapeHtml(v.id)}">${escapeHtml(v.label)}</option>`)
      .join("");
  }

  function renderAudioList() {
    const allAudio = [
      ...state.assets.audio,
      ...state.assets.uploads.filter((u) => u.type === "audio"),
    ];
    const box = $("audio-list-container");
    if (allAudio.length === 0) {
      box.innerHTML = `<div class="muted">Aún no hay voces generadas ni pistas de audio subidas.</div>`;
      return;
    }
    box.innerHTML = allAudio
      .map(
        (a, idx) => `
        <div class="audio-item">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong>${escapeHtml(a.filename)}</strong>
            <span class="muted">${a.duration_seconds ? `${a.duration_seconds}s` : ""}</span>
          </div>
          ${a.text ? `<div class="muted">"${escapeHtml(a.text)}"</div>` : ""}
          <audio controls preload="none" src="${escapeHtml(a.media_url)}"></audio>
          <div class="row-gap">
            <button class="btn btn-secondary btn-sm" data-aud-vid="${idx}">🎬 Enviar a Vídeo Hablado</button>
            <button class="btn btn-ghost btn-sm" data-aud-bg="${idx}">🎵 Poner de fondo en Montaje</button>
          </div>
        </div>
      `
      )
      .join("");

    box.querySelectorAll("[data-aud-vid]").forEach((b) =>
      b.addEventListener("click", () => {
        const a = allAudio[Number(b.dataset.audVid)];
        switchTab("tab-video");
        $("vid-source-audio").value = a.rel_path;
        toast(`Pista «${a.filename}» seleccionada para el clip de vídeo.`);
      })
    );
    box.querySelectorAll("[data-aud-bg]").forEach((b) =>
      b.addEventListener("click", () => {
        const a = allAudio[Number(b.dataset.audBg)];
        switchTab("tab-montage");
        $("montage-global-audio").value = a.rel_path;
        toast(`Pista «${a.filename}» asignada como audio global del montaje.`);
      })
    );
  }

  // ==================== MONTAJE FINAL (TIMELINE FFMPEG) ====================
  function renderMontagePickers() {
    const visualAssets = [
      ...state.assets.videos,
      ...state.assets.images,
      ...state.assets.uploads.filter((u) => u.type === "image" || u.type === "video"),
    ];
    const picker = $("montage-asset-picker");
    picker.innerHTML =
      `<option value="">-- Elige una imagen o clip generado para añadir --</option>` +
      visualAssets
        .map(
          (a) =>
            `<option value="${escapeHtml(a.rel_path)}">[${a.type === "video" ? "VÍDEO" : "IMAGEN"}] ${escapeHtml(a.filename)}</option>`
        )
        .join("");

    const allAudio = [
      ...state.assets.audio,
      ...state.assets.uploads.filter((u) => u.type === "audio"),
    ];
    const bgSel = $("montage-global-audio");
    const prevBg = bgSel.value;
    bgSel.innerHTML =
      `<option value="">-- Sin música / audio global --</option>` +
      allAudio
        .map((a) => `<option value="${escapeHtml(a.rel_path)}">${escapeHtml(a.filename)} (${a.duration_seconds || "?"}s)</option>`)
        .join("");
    if (prevBg) bgSel.value = prevBg;

    renderTimeline();
  }

  function addShotToTimeline(asset) {
    state.timeline.push({
      media_rel: asset.rel_path,
      media_url: asset.media_url,
      filename: asset.filename,
      type: asset.type || "image",
      duration: asset.duration_seconds || 3.5,
      motion: asset.type === "image" ? "zoom_in" : "none",
      audio_rel: "",
      fit_to_audio: true,
    });
    renderTimeline();
    toast(`Añadido al montaje: ${asset.filename}`, "success");
  }

  function renderTimeline() {
    const box = $("timeline-list");
    if (state.timeline.length === 0) {
      box.innerHTML = `
        <div class="result-placeholder">
          La línea de tiempo está vacía. Sube fotos/clips directamente arriba o pulsa «+Montaje» en cualquier resultado.
        </div>`;
      return;
    }

    const allAudio = [
      ...state.assets.audio,
      ...state.assets.uploads.filter((u) => u.type === "audio"),
    ];

    box.innerHTML = state.timeline
      .map((shot, idx) => {
        const thumbHtml =
          shot.type === "video"
            ? `<video src="${escapeHtml(shot.media_url)}" muted preload="metadata"></video>`
            : `<img src="${escapeHtml(shot.media_url)}" alt="" />`;

        const audioOpts =
          `<option value="">Sin voz en este plano</option>` +
          allAudio
            .map(
              (a) =>
                `<option value="${escapeHtml(a.rel_path)}" ${shot.audio_rel === a.rel_path ? "selected" : ""}>🎙️ ${escapeHtml(a.filename)}</option>`
            )
            .join("");

        const motionSelect =
          shot.type === "image"
            ? `<select data-t-motion="${idx}" title="Movimiento sobre imagen">
                 <option value="none" ${shot.motion === "none" ? "selected" : ""}>Plano fijo</option>
                 <option value="zoom_in" ${shot.motion === "zoom_in" ? "selected" : ""}>Zoom suave +</option>
                 <option value="zoom_out" ${shot.motion === "zoom_out" ? "selected" : ""}>Zoom suave -</option>
                 <option value="pan_right" ${shot.motion === "pan_right" ? "selected" : ""}>Paneo lateral</option>
               </select>`
            : `<span class="mc-tag">Clip Vídeo</span>`;

        return `
          <div class="timeline-shot">
            <div class="shot-thumb">${thumbHtml}</div>
            <div>
              <div style="font-weight:600;font-size:12px;margin-bottom:4px;">
                #${idx + 1} · ${escapeHtml(shot.filename)}
              </div>
              <div class="shot-controls">
                <label class="muted">Dur(s):
                  <input type="number" data-t-dur="${idx}" value="${shot.duration}" min="0.5" max="60" step="0.5" class="num-xs" />
                </label>
                ${motionSelect}
                <select data-t-aud="${idx}">${audioOpts}</select>
              </div>
            </div>
            <div class="row-gap">
              <button class="mini-btn" data-t-up="${idx}" title="Subir">↑</button>
              <button class="mini-btn" data-t-dn="${idx}" title="Bajar">↓</button>
              <button class="mini-btn" data-t-rm="${idx}" title="Quitar">✕</button>
            </div>
          </div>
        `;
      })
      .join("");

    box.querySelectorAll("[data-t-dur]").forEach((inp) =>
      inp.addEventListener("change", () => {
        state.timeline[Number(inp.dataset.tDur)].duration = Number(inp.value || 3.5);
      })
    );
    box.querySelectorAll("[data-t-motion]").forEach((sel) =>
      sel.addEventListener("change", () => {
        state.timeline[Number(sel.dataset.tMotion)].motion = sel.value;
      })
    );
    box.querySelectorAll("[data-t-aud]").forEach((sel) =>
      sel.addEventListener("change", () => {
        state.timeline[Number(sel.dataset.tAud)].audio_rel = sel.value;
      })
    );
    box.querySelectorAll("[data-t-up]").forEach((b) =>
      b.addEventListener("click", () => {
        const i = Number(b.dataset.tUp);
        if (i > 0) {
          [state.timeline[i - 1], state.timeline[i]] = [state.timeline[i], state.timeline[i - 1]];
          renderTimeline();
        }
      })
    );
    box.querySelectorAll("[data-t-dn]").forEach((b) =>
      b.addEventListener("click", () => {
        const i = Number(b.dataset.tDn);
        if (i < state.timeline.length - 1) {
          [state.timeline[i + 1], state.timeline[i]] = [state.timeline[i], state.timeline[i + 1]];
          renderTimeline();
        }
      })
    );
    box.querySelectorAll("[data-t-rm]").forEach((b) =>
      b.addEventListener("click", () => {
        state.timeline.splice(Number(b.dataset.tRm), 1);
        renderTimeline();
      })
    );
  }

  function showMontageResult(meta) {
    state.lastMontageResult = meta;
    $("montage-result-empty").hidden = true;
    $("montage-result-content").hidden = false;
    $("montage-result-player").src = `${meta.media_url}?t=${Date.now()}`;
    $("btn-montage-download").href = meta.media_url;
    $("btn-montage-download").download = meta.filename;
    $("montage-provenance").innerHTML = `
      <span><strong>Archivo:</strong> ${escapeHtml(meta.filename)}</span>
      <span><strong>Planos:</strong> ${meta.shots_count}</span>
      <span><strong>Duración:</strong> ${meta.duration_seconds}s</span>
      <span><strong>Formato:</strong> ${meta.width}×${meta.height} @ ${meta.fps}fps</span>
      <span><strong>Peso:</strong> ${meta.size_mb} MB</span>
    `;
  }

  function renderMontageGallery() {
    const strip = $("montage-gallery-strip");
    const list = state.assets.montages || [];
    if (list.length === 0) {
      strip.innerHTML = `<div class="muted">Todavía no has exportado ningún montaje MP4.</div>`;
      return;
    }
    strip.innerHTML = list
      .map(
        (m, idx) => `
        <div class="asset-card">
          <div class="asset-thumb"><video src="${escapeHtml(m.media_url)}" muted preload="metadata"></video></div>
          <div class="asset-meta">${escapeHtml(m.filename)} (${m.duration_seconds || "?"}s)</div>
          <div class="asset-mini-btns">
            <button class="mini-btn" data-mplay="${idx}">Reproducir</button>
            <a class="mini-btn" href="${escapeHtml(m.media_url)}" download style="text-align:center;text-decoration:none;">⬇ MP4</a>
          </div>
        </div>
      `
      )
      .join("");

    strip.querySelectorAll("[data-mplay]").forEach((b) =>
      b.addEventListener("click", () => showMontageResult(list[Number(b.dataset.mplay)]))
    );
  }

  // ==================== EVENTOS DE UI ====================
  function bindEvents() {
    // Pestañas
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // Liberar RAM/VRAM global y apagado de GPU
    $("btn-free-all-mem").addEventListener("click", () => unloadFromMemory("all"));

    const triggerShutdown = async () => {
      const warnExtra =
        state.unexportedCount > 0
          ? `\n\n⚠️ ATENCIÓN: Tienes ${state.unexportedCount} archivo(s) generados sin descargar en .zip. Si desconectas Colab se borrará el disco de sesión.`
          : "";
      if (!confirm(`¿Desconectar y liberar la GPU de Google Colab ahora para detener el consumo?${warnExtra}`)) {
        return;
      }
      try {
        const res = await api("/api/system/shutdown", { method: "POST" });
        toast(res.message, "success");
        await pollStatus();
      } catch (e) {
        toast(e.message, "error");
      }
    };
    $("btn-shutdown-gpu").addEventListener("click", triggerShutdown);
    $("btn-idle-shutdown").addEventListener("click", triggerShutdown);

    // Modal de créditos
    $("btn-open-credits").addEventListener("click", () => {
      if (state.lastHardware && state.lastHardware.credits) {
        const c = state.lastHardware.credits;
        $("cfg-initial-credits").value = c.initial_credits;
        $("cfg-hourly-rate").value = c.hourly_rate;
        $("cfg-idle-min").value = c.idle_alert_minutes;
        $("cfg-reset-session").checked = false;
      }
      $("credits-modal").hidden = false;
    });
    $("btn-close-credits").addEventListener("click", () => {
      $("credits-modal").hidden = true;
    });
    $("btn-auto-gpu-rate").addEventListener("click", async () => {
      try {
        const res = await api("/api/credits/config", {
          method: "POST",
          body: JSON.stringify({ use_auto_gpu_rate: true }),
        });
        const newRate = res.hardware.credits.hourly_rate;
        $("cfg-hourly-rate").value = newRate;
        toast(`Tarifa ajustada automáticamente a ${newRate} u/h.`, "success");
        await pollStatus();
      } catch (e) {
        toast(e.message, "error");
      }
    });
    $("btn-save-credits").addEventListener("click", async () => {
      try {
        await api("/api/credits/config", {
          method: "POST",
          body: JSON.stringify({
            initial_credits: Number($("cfg-initial-credits").value),
            hourly_rate: Number($("cfg-hourly-rate").value),
            idle_alert_minutes: Number($("cfg-idle-min").value),
            reset_session_spent: $("cfg-reset-session").checked,
          }),
        });
        $("credits-modal").hidden = true;
        toast("Contador de créditos actualizado.", "success");
        await pollStatus();
      } catch (e) {
        toast(e.message, "error");
      }
    });

    // Pestaña 1: Imagen
    if ($("select-image-model")) {
      $("select-image-model").addEventListener("change", () => {
        const mid = $("select-image-model").value;
        state.selectedImageModel = mid;
        applyImageModelDefaults(mid);
        renderImageLoraRack();
        if (state.lastHardware) updateEngineBars(state.lastHardware.loaded_slots || []);
      });
    }
    document.querySelectorAll("#img-lora-group-pills .lora-grp-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("#img-lora-group-pills .lora-grp-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        state.imgLoraGroupFilter = chip.dataset.lgrp || "all";
        renderImageLoraRack();
      });
    });
    if ($("btn-clear-img-loras")) {
      $("btn-clear-img-loras").addEventListener("click", () => {
        state.activeImgLoras = {};
        renderImageLoraRack();
      });
    }
    if ($("btn-clear-vid-extra-loras")) {
      $("btn-clear-vid-extra-loras").addEventListener("click", () => {
        state.activeVidExtraLoras = {};
        const mid = ($("select-video-model") && $("select-video-model").value) || "";
        if (mid) renderVideoExtraLoraRack(mid);
      });
    }
    $("btn-prepare-qwen").addEventListener("click", () =>
      loadModelIntoMemory(($("select-image-model") && $("select-image-model").value) || "qwen-image-edit-2511-pack", true)
    );
    $("btn-unload-image").addEventListener("click", () => unloadFromMemory("image"));
    $("btn-quick-dl-nsfw").addEventListener("click", () => downloadModel("qwen-lora-nsfw"));

    $("input-upload-image").addEventListener("change", async (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      try {
        const asset = await uploadFile(f);
        setBaseImage(asset);
        toast(`Imagen subida: ${asset.filename}`, "success");
      } catch (e) {
        toast(e.message, "error");
      }
      ev.target.value = "";
    });

    $("btn-clear-base-image").addEventListener("click", () => setBaseImage(null));

    // Controles de máscara Inpainting
    $("btn-toggle-mask").addEventListener("click", () => {
      if (!state.baseImage) {
        toast("Sube o selecciona una imagen base primero para pintar máscara encima.", "error");
        return;
      }
      state.mask.enabled = !state.mask.enabled;
      $("mask-mode-label").textContent = state.mask.enabled ? "ON" : "OFF";
      $("btn-toggle-mask").classList.toggle("active", state.mask.enabled);
      $("mask-controls").hidden = !state.mask.enabled;
      $("mask-canvas").classList.toggle("drawing-enabled", state.mask.enabled);
    });

    $("btn-brush-draw").addEventListener("click", () => {
      state.mask.tool = "draw";
      $("btn-brush-draw").classList.add("active");
      $("btn-brush-erase").classList.remove("active");
    });
    $("btn-brush-erase").addEventListener("click", () => {
      state.mask.tool = "erase";
      $("btn-brush-erase").classList.add("active");
      $("btn-brush-draw").classList.remove("active");
    });
    $("btn-undo-mask").addEventListener("click", undoMaskStroke);
    $("btn-clear-mask").addEventListener("click", clearMaskCanvas);

    // Referencias 1..3
    document.querySelectorAll(".ref-file-input").forEach((inp) => {
      inp.addEventListener("change", async (ev) => {
        const f = ev.target.files[0];
        if (!f) return;
        const slot = Number(inp.dataset.slot);
        try {
          const asset = await uploadFile(f);
          setRefSlot(slot, asset);
        } catch (e) {
          toast(e.message, "error");
        }
        ev.target.value = "";
      });
    });
    document.querySelectorAll(".ref-clear-btn").forEach((btn) => {
      btn.addEventListener("click", () => setRefSlot(Number(btn.dataset.slot), null));
    });

    // Presets resolución imagen
    document.querySelectorAll(".chip-preset").forEach((chip) => {
      chip.addEventListener("click", () => {
        $("img-width").value = chip.dataset.w;
        $("img-height").value = chip.dataset.h;
      });
    });

    $("chk-lightning").addEventListener("change", (ev) => {
      $("img-steps").value = ev.target.checked ? 4 : 20;
    });

    $("btn-run-image").addEventListener("click", () => runImageGeneration(false));
    $("btn-run-bt709-cpu").addEventListener("click", () => runImageGeneration(true));

    $("btn-res-to-base").addEventListener("click", () => {
      if (state.lastImageResult) {
        setBaseImage({
          ...state.lastImageResult,
          rel_path: `outputs/images/${state.lastImageResult.filename}`,
        });
        toast("Resultado cargado como nueva base de edición.");
      }
    });
    $("btn-res-to-video").addEventListener("click", () => {
      if (state.lastImageResult) {
        switchTab("tab-video");
        if (state.videoSubmode === "t2v") setVideoSubmode("i2v");
        $("vid-source-image").value = `outputs/images/${state.lastImageResult.filename}`;
        updateVideoSourceThumb();
      }
    });
    $("btn-res-to-montage").addEventListener("click", () => {
      if (state.lastImageResult) {
        addShotToTimeline({
          ...state.lastImageResult,
          rel_path: `outputs/images/${state.lastImageResult.filename}`,
          type: "image",
        });
      }
    });

    // Pestaña 2: Vídeo
    $("select-video-model").addEventListener("change", (ev) => {
      const mid = ev.target.value;
      document.querySelectorAll(".vm-card").forEach((c) => {
        c.classList.toggle("selected", c.dataset.vmid === mid);
      });
      applyVideoModelDefaults(mid);
    });

    $("btn-prepare-video").addEventListener("click", () => {
      const mid = $("select-video-model").value || "wan22-ti2v-5b";
      loadModelIntoMemory(mid, true);
    });
    $("btn-unload-video").addEventListener("click", () => unloadFromMemory("video"));
    $("vid-source-image").addEventListener("change", updateVideoSourceThumb);
    if ($("vid-end-image")) {
      $("vid-end-image").addEventListener("change", updateVideoEndThumb);
    }
    if ($("btn-vid-swap-frames")) {
      $("btn-vid-swap-frames").addEventListener("click", () => {
        const a = $("vid-source-image").value;
        const b = $("vid-end-image").value;
        $("vid-source-image").value = b;
        $("vid-end-image").value = a;
        updateVideoSourceThumb();
        updateVideoEndThumb();
      });
    }
    if ($("chk-ffmpeg-two-frames")) {
      $("chk-ffmpeg-two-frames").addEventListener("change", () => syncVideoSubmodeUI());
    }

    document.querySelectorAll(".vsub-card").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sub = btn.dataset.submode || "i2v";
        setVideoSubmode(sub);
        if (sub === "ffmpeg") {
          $("vid-width").value = 1280;
          $("vid-height").value = 720;
          $("vid-fps").value = 24;
        } else {
          const mid = $("select-video-model") ? $("select-video-model").value : null;
          if (mid) applyVideoModelDefaults(mid);
        }
      });
    });

    if ($("btn-restore-official-vid")) {
      $("btn-restore-official-vid").addEventListener("click", () => {
        const mid = $("select-video-model") ? $("select-video-model").value : null;
        if ($("vid-lora-select")) $("vid-lora-select").value = "";
        if (mid) {
          applyVideoModelDefaults(mid);
          toast("Restaurados todos los parámetros oficiales del modelo.", "success");
        }
      });
    }

    if ($("vid-lora-select")) {
      $("vid-lora-select").addEventListener("change", () => {
        const lid = $("vid-lora-select").value;
        const loraObj = lid ? state.catalog.find((x) => x.id === lid) : null;
        if ($("vid-lora-scale-wrap")) $("vid-lora-scale-wrap").hidden = !loraObj;
        if ($("btn-dl-vid-lora")) {
          $("btn-dl-vid-lora").hidden = !loraObj || Boolean(loraObj.downloaded);
          if (loraObj && !loraObj.downloaded) {
            $("btn-dl-vid-lora").textContent = `⬇️ Pre-descargar (${loraObj.estimated_gb} GB)`;
          }
        }
        resetVideoStepsToRecommended();
        if (loraObj) {
          const rec = getLoraRecommendedForSubmode(loraObj, state.videoSubmode || "i2v");
          toast(
            `LoRA «${loraObj.name.split("(")[0].trim()}» activo: ajustados ${rec.steps} pasos, Fuerza ${rec.scale} y CFG ${rec.cfg}.`,
            "success"
          );
        }
      });
    }

    if ($("btn-dl-vid-lora")) {
      $("btn-dl-vid-lora").addEventListener("click", () => {
        const lid = $("vid-lora-select") ? $("vid-lora-select").value : "";
        if (lid) downloadModel(lid);
      });
    }

    if ($("vid-steps-slider")) {
      $("vid-steps-slider").addEventListener("input", (ev) => {
        if ($("vid-steps")) $("vid-steps").value = ev.target.value;
        updateVideoStepsSafetyNote();
      });
    }

    if ($("vid-steps")) {
      $("vid-steps").addEventListener("input", () => {
        updateVideoStepsSafetyNote();
      });
    }

    if ($("vid-cfg")) {
      $("vid-cfg").addEventListener("input", () => {
        updateVideoStepsSafetyNote();
      });
    }

    if ($("vid-lora-scale")) {
      $("vid-lora-scale").addEventListener("input", () => {
        updateVideoStepsSafetyNote();
      });
    }

    document.querySelectorAll(".lora-q-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        const lsteps = Number(pill.dataset.lsteps || 6);
        const lscale = Number(pill.dataset.lscale || 1.0);
        const lcfg = Number(pill.dataset.lcfg || 1.0);
        if ($("vid-steps")) $("vid-steps").value = lsteps;
        if ($("vid-steps-slider")) $("vid-steps-slider").value = lsteps;
        if ($("vid-lora-scale")) $("vid-lora-scale").value = lscale;
        if ($("vid-cfg")) $("vid-cfg").value = lcfg;
        updateVideoStepsSafetyNote();
        toast(
          `Ajustado modo LoRA: ${lsteps} pasos · Fuerza ${lscale} · CFG ${lcfg}.`,
          "success"
        );
      });
    });

    if ($("btn-steps-fast-safe")) {
      $("btn-steps-fast-safe").addEventListener("click", () => {
        const mid = $("select-video-model") ? $("select-video-model").value : "";
        const m = state.catalog.find((x) => x.id === mid);
        const fastSafe = Number((m && m.official_spec && m.official_spec.fast_safe_steps) || 28);
        if ($("vid-steps")) $("vid-steps").value = fastSafe;
        if ($("vid-steps-slider")) $("vid-steps-slider").value = fastSafe;
        updateVideoStepsSafetyNote();
        toast(`Ajustados ${fastSafe} pasos (modo rápido dentro del rango seguro sin deformar).`, "success");
      });
    }

    if ($("btn-reset-vid-steps")) {
      $("btn-reset-vid-steps").addEventListener("click", () => {
        resetVideoStepsToRecommended();
        toast("Pasos y CFG devueltos al valor predeterminado recomendado.", "success");
      });
    }

    document.querySelectorAll(".vid-preset-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        $("vid-width").value = chip.dataset.vw;
        $("vid-height").value = chip.dataset.vh;
      });
    });

    // Filtro por GPU de Colab sincronizado entre Vídeo y Catálogo HF
    document.querySelectorAll(".gpu-pill").forEach((pill) => {
      pill.addEventListener("click", async () => {
        state.gpuProfile = pill.dataset.gpu || "auto";
        await loadCatalog();
        if (state.hfSearchResults.length > 0) {
          runHfLiveSearch();
        }
      });
    });

    if ($("chk-vid-only-compat")) {
      $("chk-vid-only-compat").addEventListener("change", (ev) => {
        state.vidOnlyCompat = ev.target.checked;
        populateVideoModelSelect();
      });
    }
    if ($("chk-cat-only-compat")) {
      $("chk-cat-only-compat").addEventListener("change", (ev) => {
        state.catOnlyCompat = ev.target.checked;
        renderCatalog();
        if (state.hfSearchResults.length > 0) {
          runHfLiveSearch();
        }
      });
    }

    $("input-vid-upload-img").addEventListener("change", async (ev) => {
      const files = Array.from(ev.target.files || []);
      if (files.length === 0) return;
      try {
        if (files.length >= 2) {
          const assetStart = await uploadFile(files[0]);
          const assetEnd = await uploadFile(files[1]);
          $("vid-source-image").value = assetStart.rel_path;
          if ($("vid-end-image")) $("vid-end-image").value = assetEnd.rel_path;
          state.lastVidUploadStartRel = null;
          setVideoSubmode("flf2v");
          updateVideoSourceThumb();
          updateVideoEndThumb();
          toast(
            `🎞️ ¡2 fotos listas! Inicio: ${assetStart.filename} → Final: ${assetEnd.filename} (modo Frame a Frame activado).`,
            "success"
          );
        } else {
          const prevStart = $("vid-source-image").value;
          const hadRecentSingleStart =
            state.videoSubmode === "i2v" &&
            Boolean(state.lastVidUploadStartRel) &&
            prevStart === state.lastVidUploadStartRel;

          const asset = await uploadFile(files[0]);
          if (hadRecentSingleStart && asset.rel_path !== prevStart) {
            $("vid-source-image").value = prevStart;
            if ($("vid-end-image")) $("vid-end-image").value = asset.rel_path;
            state.lastVidUploadStartRel = null;
            setVideoSubmode("flf2v");
            updateVideoSourceThumb();
            updateVideoEndThumb();
            toast(
              `🎞️ 2ª foto detectada (${asset.filename}): activado Frame a Frame (Inicio → Fin). Usa «⇄ Invertir» si quieres cambiar el orden.`,
              "success"
            );
          } else {
            $("vid-source-image").value = asset.rel_path;
            state.lastVidUploadStartRel = state.videoSubmode === "i2v" ? asset.rel_path : null;
            updateVideoSourceThumb();
            toast(`Imagen inicial lista: ${asset.filename}`, "success");
          }
        }
      } catch (e) {
        toast(e.message, "error");
      }
      ev.target.value = "";
    });

    if ($("input-vid-quick-end-img")) {
      $("input-vid-quick-end-img").addEventListener("change", async (ev) => {
        const f = ev.target.files[0];
        if (!f) return;
        try {
          const prevStart = $("vid-source-image").value;
          const asset = await uploadFile(f);
          if (prevStart) $("vid-source-image").value = prevStart;
          if ($("vid-end-image")) $("vid-end-image").value = asset.rel_path;
          state.lastVidUploadStartRel = null;
          setVideoSubmode("flf2v");
          updateVideoSourceThumb();
          updateVideoEndThumb();
          toast(`🏁 Fotograma final listo (${asset.filename}): modo Frame a Frame activado.`, "success");
        } catch (e) {
          toast(e.message, "error");
        }
        ev.target.value = "";
      });
    }

    if ($("input-vid-upload-end-img")) {
      $("input-vid-upload-end-img").addEventListener("change", async (ev) => {
        const f = ev.target.files[0];
        if (!f) return;
        try {
          const asset = await uploadFile(f);
          if ($("vid-end-image")) $("vid-end-image").value = asset.rel_path;
          state.lastVidUploadStartRel = null;
          updateVideoEndThumb();
          toast(`Fotograma final listo: ${asset.filename}`, "success");
        } catch (e) {
          toast(e.message, "error");
        }
        ev.target.value = "";
      });
    }

    $("btn-run-video").addEventListener("click", async () => {
      try {
        const chosenVidModel = $("select-video-model").value;
        const sub = state.videoSubmode || "i2v";
        const effectiveMode =
          chosenVidModel === "ffmpeg-motion-cpu" || sub === "ffmpeg" ? "ffmpeg_motion" : "ai";

        let effectiveSubmode = sub;
        const startRel = $("vid-source-image").value || null;
        let endRel = $("vid-end-image") ? $("vid-end-image").value || null : null;

        if (effectiveMode === "ffmpeg_motion") {
          const ffTwo = $("chk-ffmpeg-two-frames") && $("chk-ffmpeg-two-frames").checked;
          effectiveSubmode = ffTwo && endRel ? "flf2v" : "i2v";
          if (!startRel) {
            toast("Sube o elige una imagen inicial con «📤 Subir imagen».", "error");
            return;
          }
          if (!ffTwo) endRel = null;
        } else if (effectiveSubmode === "t2v") {
          if (!($("vid-prompt").value || "").trim()) {
            toast("Escribe una descripción en el prompt para Texto a Vídeo.", "error");
            return;
          }
        } else if (effectiveSubmode === "i2v") {
          if (!startRel) {
            toast("Modo Imagen a Vídeo: sube o elige una imagen inicial (o cambia a «Texto a Vídeo»).", "error");
            return;
          }
          endRel = null;
        } else if (effectiveSubmode === "flf2v") {
          if (!startRel || !endRel) {
            toast(
              "Modo Frame a Frame: sube o elige tanto el Fotograma Inicial como el Fotograma Final.",
              "error"
            );
            return;
          }
        }

        const chosenVidLora = $("vid-lora-select") ? $("vid-lora-select").value || null : null;
        const chosenVidLoraScale = $("vid-lora-scale") ? Number($("vid-lora-scale").value || 1.0) : 1.0;
        const extraVidLoraIds = effectiveMode === "ai" ? Object.keys(state.activeVidExtraLoras) : [];

        const res = await api("/api/video/generate", {
          method: "POST",
          body: JSON.stringify({
            mode: effectiveMode,
            video_submode: effectiveSubmode,
            model_id: chosenVidModel || null,
            video_lora_id: effectiveMode === "ai" ? chosenVidLora : null,
            video_lora_scale: chosenVidLoraScale,
            extra_video_lora_ids: extraVidLoraIds,
            extra_video_lora_scales: { ...state.activeVidExtraLoras },
            prompt: $("vid-prompt").value,
            negative_prompt: $("vid-neg-prompt") ? $("vid-neg-prompt").value : "",
            image_rel: effectiveSubmode === "t2v" ? null : startRel,
            end_image_rel: effectiveSubmode === "flf2v" ? endRel : null,
            audio_rel: $("vid-source-audio").value || null,
            width: Number($("vid-width").value || 1280),
            height: Number($("vid-height").value || 704),
            duration_sec: Number($("vid-duration").value || 5),
            fps: Number($("vid-fps").value || 24),
            steps: Number($("vid-steps").value || 45),
            guidance_scale: $("vid-cfg") ? Number($("vid-cfg").value || 5.0) : null,
            seed: Number($("vid-seed").value || 0),
            motion_style: $("vid-motion-style").value,
          }),
        });
        watchJob(res.job);
      } catch (e) {
        toast(e.message, "error");
      }
    });

    $("btn-vid-to-montage").addEventListener("click", () => {
      if (state.lastVideoResult) {
        addShotToTimeline({
          ...state.lastVideoResult,
          rel_path: `outputs/videos/${state.lastVideoResult.filename}`,
          type: "video",
        });
      }
    });

    // Pestaña 3: Voces / TTS
    $("input-upload-audio").addEventListener("change", async (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      try {
        const asset = await uploadFile(f);
        toast(`Audio subido: ${asset.filename}`, "success");
      } catch (e) {
        toast(e.message, "error");
      }
      ev.target.value = "";
    });

    $("btn-run-tts").addEventListener("click", async () => {
      try {
        const res = await api("/api/tts/generate", {
          method: "POST",
          body: JSON.stringify({
            text: $("tts-text").value,
            engine_id: $("tts-engine-select").value,
            voice_id: $("tts-voice-select").value,
            rate_pct: Number($("tts-rate").value || 0),
            pitch_hz: Number($("tts-pitch").value || 0),
          }),
        });
        watchJob(res.job);
      } catch (e) {
        toast(e.message, "error");
      }
    });

    if ($("btn-run-bgm")) {
      $("btn-run-bgm").addEventListener("click", async () => {
        try {
          const res = await api("/api/audio/generate-bgm", {
            method: "POST",
            body: JSON.stringify({
              model_id: $("bgm-model-select").value,
              preset: $("bgm-preset-select").value,
              prompt: $("bgm-prompt").value || "warm restaurant ambience",
              duration_seconds: Number($("bgm-duration").value || 10),
            }),
          });
          watchJob(res.job);
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    if ($("btn-dl-lipsync")) {
      $("btn-dl-lipsync").addEventListener("click", () => downloadModel("edtalk-256-lipsync-hd"));
    }

    if ($("btn-run-lipsync")) {
      $("btn-run-lipsync").addEventListener("click", async () => {
        const mediaRel = $("lipsync-media-select") ? $("lipsync-media-select").value : "";
        const voiceRel = $("lipsync-voice-select") ? $("lipsync-voice-select").value : "";
        if (!mediaRel) {
          toast("Selecciona primero la imagen o clip de vídeo de los personajes.", "error");
          return;
        }
        if (!voiceRel) {
          toast("Selecciona la pista de voz del personaje.", "error");
          return;
        }
        try {
          const res = await api("/api/video/lipsync", {
            method: "POST",
            body: JSON.stringify({
              media_rel: mediaRel,
              voice_audio_rel: voiceRel,
              target_face: $("lipsync-target-face").value || "largest",
              ambient_rel: ($("lipsync-ambient-select") && $("lipsync-ambient-select").value) || null,
              music_rel: ($("lipsync-music-select") && $("lipsync-music-select").value) || null,
            }),
          });
          watchJob(res.job);
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    // Pestaña 4: Montaje FFmpeg
    $("input-montage-files").addEventListener("change", async (ev) => {
      const files = Array.from(ev.target.files || []);
      for (const f of files) {
        try {
          const asset = await uploadFile(f);
          addShotToTimeline(asset);
        } catch (e) {
          toast(e.message, "error");
        }
      }
      ev.target.value = "";
    });

    $("btn-add-picked-shot").addEventListener("click", () => {
      const rel = $("montage-asset-picker").value;
      if (!rel) return;
      const pool = [
        ...state.assets.videos,
        ...state.assets.images,
        ...state.assets.uploads,
      ];
      const found = pool.find((a) => a.rel_path === rel);
      if (found) addShotToTimeline(found);
    });

    $("btn-clear-timeline").addEventListener("click", () => {
      state.timeline = [];
      renderTimeline();
    });

    $("btn-render-montage").addEventListener("click", async () => {
      if (state.timeline.length === 0) {
        toast("Añade al menos una imagen o clip a la línea de tiempo.", "error");
        return;
      }
      const [wStr, hStr] = $("montage-res-preset").value.split("x");
      try {
        const res = await api("/api/montage/render", {
          method: "POST",
          body: JSON.stringify({
            title: $("montage-title").value || "montaje",
            width: Number(wStr),
            height: Number(hStr),
            fps: Number($("montage-fps").value || 24),
            fade_seconds: Number($("montage-fade").value || 0.2),
            global_audio_rel: $("montage-global-audio").value || null,
            global_audio_volume: Number($("montage-bg-vol").value || 0.3),
            items: state.timeline.map((s) => ({
              media_rel: s.media_rel,
              duration: Number(s.duration || 3.5),
              motion: s.motion || "none",
              audio_rel: s.audio_rel || null,
              fit_to_audio: true,
            })),
          }),
        });
        watchJob(res.job);
      } catch (e) {
        toast(e.message, "error");
      }
    });

    // Pestaña 5: Catálogo, Buscador HF en tiempo real, Token HF opcional y Añadir HF custom
    const openHfModal = () => {
      $("hf-token-modal").hidden = false;
    };
    if ($("btn-open-hf-token")) $("btn-open-hf-token").addEventListener("click", openHfModal);
    if ($("btn-hf-box-token")) $("btn-hf-box-token").addEventListener("click", openHfModal);
    if ($("btn-close-hf-token")) {
      $("btn-close-hf-token").addEventListener("click", () => {
        $("hf-token-modal").hidden = true;
      });
    }
    if ($("btn-open-mcp-modal")) {
      $("btn-open-mcp-modal").addEventListener("click", async () => {
        try {
          // X-Studio-UI identifica al panel servido por este mismo host: solo así
          // el servidor devuelve el token (nunca a un tercero que abra la URL del túnel).
          const info = await api("/api/mcp/info", { headers: { "X-Studio-UI": "1" } });
          state.lastMcpInfo = info;
          const cmds = info.commands || {};
          if ($("mcp-cmd-agent-prompt")) $("mcp-cmd-agent-prompt").value = cmds.agent_chat_prompt || "";
          if ($("mcp-url-with-token")) $("mcp-url-with-token").value = info.mcp_url_with_token || "";
          if ($("mcp-cmd-claude-stdio")) $("mcp-cmd-claude-stdio").value = cmds.claude_code_stdio || "";
          if ($("mcp-cmd-claude-ps")) $("mcp-cmd-claude-ps").value = cmds.claude_code_powershell || "";
          if ($("mcp-cmd-codex-stdio")) $("mcp-cmd-codex-stdio").value = cmds.codex_cli_stdio || "";
          if ($("mcp-cmd-http")) $("mcp-cmd-http").value = cmds.claude_code_http || "";
          if ($("mcp-cmd-json")) $("mcp-cmd-json").value = cmds.mcp_servers_json || "";
          refreshGitHubChannel();
          if (info.token_exposed === false) {
            toast("Este panel no es de confianza para ver el token: ábrelo desde este mismo dominio.");
          }

          const presets = info.workflow_presets || [];
          const pSel = $("mcp-preset-select");
          if (pSel && presets.length > 0) {
            pSel.innerHTML = presets
              .map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.title)}</option>`)
              .join("");
            const syncPresetDesc = () => {
              const cur = presets.find((x) => x.id === pSel.value) || presets[0];
              if ($("mcp-preset-desc") && cur) {
                $("mcp-preset-desc").textContent = `${cur.when_to_use} — ${cur.solves_problem}`;
              }
            };
            pSel.onchange = syncPresetDesc;
            syncPresetDesc();
          }
          $("mcp-modal").hidden = false;
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }
    if ($("btn-copy-mcp-preset-order")) {
      $("btn-copy-mcp-preset-order").addEventListener("click", async () => {
        const pid = ($("mcp-preset-select") && $("mcp-preset-select").value) || "coherencia_personajes_entorno_avance";
        const scene = (($("mcp-preset-scene-input") && $("mcp-preset-scene-input").value) || "").trim() ||
          "dos personas andando por la calle hablando al atardecer";
        const guideUrl = (state.lastMcpInfo && state.lastMcpInfo.agent_guide_url) || "";
        const guideLine = guideUrl ? ` (Si aún no estás conectado, lee primero con curl ${guideUrl}).` : "";
        const msg =
          `Usa obligatoriamente la herramienta MCP \`studio_get_workflow_preset\` con \`preset_id="${pid}"\` y ` +
          `\`scene_description="${scene}"\` antes de generar nada, y ejecuta paso a paso su \`execution_plan\` ` +
          `respetando todas sus \`golden_rules_mandatory\` (avance espacial real metro a metro sin congelar el fondo, ` +
          `auditoría \`expect_spatial_progression=true\` y prompt de vídeo \`flf2v\` en 3 vectores).${guideLine}`;
        try {
          await navigator.clipboard.writeText(msg);
          toast("🧭 Orden con Preset copiada para pegar a tu Agente.", "success");
        } catch (_e) {
          toast("No se pudo copiar automáticamente.", "error");
        }
      });
    }
    if ($("btn-close-mcp-modal")) {
      $("btn-close-mcp-modal").addEventListener("click", () => {
        $("mcp-modal").hidden = true;
      });
    }
    document.querySelectorAll("[data-copy-target]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const el = $(btn.dataset.copyTarget);
        if (!el || !el.value) return;
        try {
          await navigator.clipboard.writeText(el.value);
          toast("Comando MCP copiado al portapapeles.", "success");
        } catch (_err) {
          el.select();
          document.execCommand("copy");
          toast("Comando MCP copiado.", "success");
        }
      });
    });

    if ($("btn-save-hf-token")) {
      $("btn-save-hf-token").addEventListener("click", async () => {
        const val = ($("inp-hf-token").value || "").trim();
        if (!val) {
          toast("Pega un token hf_... o pulsa «Quitar Token».", "error");
          return;
        }
        try {
          const res = await api("/api/hf/token", {
            method: "POST",
            body: JSON.stringify({ token: val }),
          });
          $("inp-hf-token").value = "";
          renderHfAuthStatus(res.hf_auth);
          $("hf-token-modal").hidden = true;
          toast("Token de Hugging Face activado en esta sesión.", "success");
          runHfLiveSearch();
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }
    if ($("btn-clear-hf-token")) {
      $("btn-clear-hf-token").addEventListener("click", async () => {
        try {
          const res = await api("/api/hf/token", { method: "DELETE" });
          $("inp-hf-token").value = "";
          renderHfAuthStatus(res.hf_auth);
          toast("Token de Hugging Face eliminado.", "success");
          runHfLiveSearch();
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    if ($("btn-vid-jump-hf")) {
      $("btn-vid-jump-hf").addEventListener("click", () => {
        switchTab("tab-catalog");
        if ($("hf-live-cat")) $("hf-live-cat").value = "video";
        const qEl = $("hf-live-query");
        if (qEl) qEl.focus();
        runHfLiveSearch(null, "video");
      });
    }

    document.querySelectorAll(".filter-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        state.catalogFilter = chip.dataset.cat;
        document.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        renderCatalog();
      });
    });

    if ($("btn-hf-live-search")) {
      $("btn-hf-live-search").addEventListener("click", () => runHfLiveSearch(null, null, 0, true));
    }
    if ($("hf-sort")) $("hf-sort").addEventListener("change", () => runHfLiveSearch());
    if ($("hf-auto-refresh")) $("hf-auto-refresh").addEventListener("change", updateDiscoverySettings);
    if ($("hf-refresh-interval")) $("hf-refresh-interval").addEventListener("change", updateDiscoverySettings);
    if ($("btn-hf-prev")) $("btn-hf-prev").addEventListener("click", () => runHfLiveSearch(null, null, Math.max(0, state.hfOffset - 12)));
    if ($("btn-hf-next")) $("btn-hf-next").addEventListener("click", () => {
      if (state.hfNextOffset != null) runHfLiveSearch(null, null, state.hfNextOffset);
    });
    if ($("btn-hf-close-inspection")) $("btn-hf-close-inspection").addEventListener("click", () => { $("hf-inspection").hidden = true; });
    if ($("btn-hf-oracle")) $("btn-hf-oracle").addEventListener("click", runHardwareOracle);
    if ($("btn-hf-refresh-index")) $("btn-hf-refresh-index").addEventListener("click", async () => {
      try {
        const result = await api("/api/models/discovery/refresh", { method: "POST", body: JSON.stringify({ category: "all" }) });
        watchJob(result.job);
        toast("Actualización de metadatos en cola. No se descargan pesos.", "info");
        refreshDiscoveryStatus();
      } catch (error) { toast(error.message, "error"); }
    });
    if ($("hf-live-query")) {
      // Búsqueda instantánea en tiempo real mientras escribes (debounce 280 ms)
      $("hf-live-query").addEventListener("input", () => scheduleRealtimeHfSearch());
      $("hf-live-query").addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") {
          if (_hfDebounceTimer) clearTimeout(_hfDebounceTimer);
          runHfLiveSearch();
        }
      });
    }
    if ($("hf-live-cat")) {
      $("hf-live-cat").addEventListener("change", () => runHfLiveSearch());
    }
    document.querySelectorAll(".hf-topic-chip").forEach((chip) => {
      chip.addEventListener("click", () => runHfLiveSearch(chip.dataset.q, chip.dataset.c));
    });

    $("btn-add-custom-hf").addEventListener("click", async () => {
      try {
        await api("/api/catalog/custom", {
          method: "POST",
          body: JSON.stringify({
            name: $("custom-hf-name").value,
            repo_id: $("custom-hf-repo").value,
            category: $("custom-hf-cat").value,
            filename: $("custom-hf-file").value || null,
            estimated_gb: Number($("custom-hf-gb").value || 1.0),
          }),
        });
        $("custom-hf-name").value = "";
        $("custom-hf-repo").value = "";
        $("custom-hf-file").value = "";
        toast("Candidato guardado. Falta receta revisada; no se han descargado pesos.", "info");
        await loadCatalog();
      } catch (e) {
        toast(e.message, "error");
      }
    });
  }

  // ==================== MODO HIGGSFIELD (100% GRATIS LOCAL / COLAB) ====================
  function fillKeyLabelSelect(selId, dictObj, preferredKey = "") {
    const el = $(selId);
    if (!el || !dictObj) return;
    const prev = el.value || preferredKey;
    el.innerHTML = Object.entries(dictObj)
      .map(([k, v]) => `<option value="${escapeHtml(k)}">${escapeHtml(v.label || k)}</option>`)
      .join("");
    if (prev && dictObj[prev]) el.value = prev;
  }

  async function loadHiggsfieldCatalog() {
    try {
      const data = await api("/api/higgsfield/catalog");
      state.higgsfield = data;
      fillKeyLabelSelect("hf-dir-camera", data.cameras, "arri_full_frame_8k");
      fillKeyLabelSelect("hf-cin-camera", data.cameras, "arri_full_frame_8k");
      fillKeyLabelSelect("hf-dir-lens", data.lenses, "classic_anamorphic");
      fillKeyLabelSelect("hf-cin-lens", data.lenses, "classic_anamorphic");
      fillKeyLabelSelect("hf-dir-focal", data.focal_lengths, "35");
      fillKeyLabelSelect("hf-cin-focal", data.focal_lengths, "35");
      fillKeyLabelSelect("hf-dir-aperture", data.apertures, "f/2.8");
      fillKeyLabelSelect("hf-cin-aperture", data.apertures, "f/2.8");
      fillKeyLabelSelect("hf-dir-lighting", data.lighting_moods, "warm_restaurant");
      fillKeyLabelSelect("hf-cin-lighting", data.lighting_moods, "natural_cinema");
      fillKeyLabelSelect("hf-dir-motion", data.motions, "over_the_shoulder");

      const pillsBox = $("hf-motion-pills");
      if (pillsBox && data.motions) {
        const curMot = ($("hf-mot-selected") && $("hf-mot-selected").value) || "dolly_in_dramatic";
        pillsBox.innerHTML = Object.entries(data.motions)
          .map(
            ([k, v]) =>
              `<button type="button" class="hf-motion-pill ${k === curMot ? "active" : ""}" data-hf-mot="${escapeHtml(k)}">${escapeHtml(v.label)}</button>`
          )
          .join("");
        pillsBox.querySelectorAll(".hf-motion-pill").forEach((btn) => {
          btn.addEventListener("click", () => {
            $("hf-mot-selected").value = btn.dataset.hfMot;
            pillsBox.querySelectorAll(".hf-motion-pill").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
          });
        });
      }

      renderHiggsfieldCharacters(data.characters || []);
      renderHiggsfieldVoiceSelects();
    } catch (e) {
      console.warn("No se pudo cargar catálogo Higgsfield:", e);
    }
  }

  function renderHiggsfieldVoiceSelects() {
    const voices = state.voices || [];
    if (!voices.length) return;
    const opts = voices.map((v) => `<option value="${escapeHtml(v.id)}">${escapeHtml(v.label)}</option>`).join("");
    ["hf-dir-voice1", "hf-dir-voice2", "hf-char-voice"].forEach((id, idx) => {
      const el = $(id);
      if (!el) return;
      const prev = el.value;
      el.innerHTML = opts;
      if (prev && voices.some((v) => v.id === prev)) {
        el.value = prev;
      } else if (idx === 1 && voices.some((v) => v.id === "es-ES-ElviraNeural")) {
        el.value = "es-ES-ElviraNeural";
      }
    });
  }

  function renderHiggsfieldCharacters(chars) {
    state.hfCharacters = chars || [];
    const listBox = $("hf-characters-list");
    if (listBox) {
      if (!state.hfCharacters.length) {
        listBox.innerHTML = `<div class="muted" style="font-size:12px;">Aún no hay personajes guardados en el Casting Vault.</div>`;
      } else {
        listBox.innerHTML = state.hfCharacters
          .map(
            (c) => `
          <div class="hf-char-card">
            <div class="row-gap">
              ${c.media_url ? `<img src="${escapeHtml(c.media_url)}" class="hf-char-thumb" alt="" />` : `<div class="hf-char-thumb"></div>`}
              <div>
                <strong>${escapeHtml(c.name)}</strong>
                <div class="muted" style="font-size:11.5px;">${escapeHtml(c.appearance_prompt || "Sin rasgos extra")} · 🎙️ ${escapeHtml(c.voice_id || "")}</div>
              </div>
            </div>
            <button type="button" class="btn btn-ghost btn-sm" data-del-hfchar="${escapeHtml(c.id)}">🗑</button>
          </div>`
          )
          .join("");
        listBox.querySelectorAll("[data-del-hfchar]").forEach((btn) => {
          btn.addEventListener("click", async () => {
            try {
              const r = await api(`/api/higgsfield/characters/${encodeURIComponent(btn.dataset.delHfchar)}`, { method: "DELETE" });
              renderHiggsfieldCharacters(r.characters || []);
              toast("Personaje eliminado del Casting Vault.");
            } catch (e) {
              toast(e.message, "error");
            }
          });
        });
      }
    }

    const baseOpts = state.hfCharacters
      .map((c) => `<option value="${escapeHtml(c.id)}">👤 ${escapeHtml(c.name)} (${escapeHtml(c.voice_id || "")})</option>`)
      .join("");
    [
      ["hf-dir-char1", "— Sin ficha asignada —"],
      ["hf-dir-char2", "— Sin segundo personaje —"],
      ["hf-cin-char", "— Ninguno —"],
    ].forEach(([sid, emptyLabel]) => {
      const el = $(sid);
      if (!el) return;
      const prev = el.value;
      el.innerHTML = `<option value="">${emptyLabel}</option>` + baseOpts;
      if (prev) el.value = prev;
    });
  }

  function renderHiggsfieldAssetSelects() {
    if (!state.assets) return;
    const imgs = [
      ...(state.assets.images || []),
      ...(state.assets.uploads || []).filter((u) => u.type === "image"),
    ];
    const vids = [
      ...(state.assets.videos || []),
      ...(state.assets.montages || []),
      ...(state.assets.uploads || []).filter((u) => u.type === "video"),
    ];

    const imgOpts = imgs.map((i) => `<option value="${escapeHtml(i.rel_path)}">🖼️ ${escapeHtml(i.filename)}</option>`).join("");
    if ($("hf-dir-base-img")) {
      const p = $("hf-dir-base-img").value;
      $("hf-dir-base-img").innerHTML = `<option value="">✨ Generar fotograma Cinema Studio nuevo</option>` + imgOpts;
      if (p) $("hf-dir-base-img").value = p;
    }
    if ($("hf-mot-img")) {
      const p = $("hf-mot-img").value;
      $("hf-mot-img").innerHTML = imgOpts || `<option value="">(Sube o genera una imagen primero)</option>`;
      if (p) $("hf-mot-img").value = p;
    }
    if ($("hf-mot-end-img")) {
      const p = $("hf-mot-end-img").value;
      $("hf-mot-end-img").innerHTML = `<option value="">— Libre (solo fotograma inicial) —</option>` + imgOpts;
      if (p) $("hf-mot-end-img").value = p;
    }
    if ($("hf-char-anchor")) {
      const p = $("hf-char-anchor").value;
      $("hf-char-anchor").innerHTML = `<option value="">— Sin foto ancla (solo rasgos) —</option>` + imgOpts;
      if (p) $("hf-char-anchor").value = p;
    }
    if ($("hf-clip-video")) {
      const p = $("hf-clip-video").value;
      $("hf-clip-video").innerHTML =
        vids.map((v) => `<option value="${escapeHtml(v.rel_path)}">🎬 ${escapeHtml(v.filename)} (${v.duration_seconds || "?"}s)</option>`).join("") ||
        `<option value="">(Genera o sube un vídeo MP4 primero)</option>`;
      if (p) $("hf-clip-video").value = p;
    }
  }

  function showHiggsfieldStageResult(meta) {
    state.lastHfResult = meta;
    if ($("hf-empty-stage")) $("hf-empty-stage").hidden = true;
    const imgEl = $("hf-result-img");
    const vidEl = $("hf-result-video");
    const isVid = meta.type === "video" || (meta.filename && meta.filename.endsWith(".mp4"));
    if (isVid) {
      if (imgEl) imgEl.hidden = true;
      if (vidEl) {
        vidEl.hidden = false;
        vidEl.src = `${meta.media_url}?t=${Date.now()}`;
      }
    } else {
      if (vidEl) {
        vidEl.pause();
        vidEl.hidden = true;
      }
      if (imgEl) {
        imgEl.hidden = false;
        imgEl.src = `${meta.media_url}?t=${Date.now()}`;
      }
    }
    if ($("hf-result-actions")) $("hf-result-actions").hidden = false;
    if ($("btn-hf-download")) {
      $("btn-hf-download").href = meta.media_url;
      $("btn-hf-download").download = meta.filename || "higgsfield_out";
    }
    if ($("hf-preview-meta")) {
      $("hf-preview-meta").textContent = `${meta.filename || ""} · ${meta.width || ""}×${meta.height || ""} ${meta.duration_seconds ? `· ${meta.duration_seconds}s` : ""}`;
    }
    const pbox = $("hf-compiled-prompt-box");
    if (pbox) {
      if (meta.compiled_cinema_prompt) {
        pbox.hidden = false;
        pbox.innerHTML = `<span><strong>Óptica aplicada:</strong> ${escapeHtml(meta.compiled_cinema_prompt)}</span>`;
      } else {
        pbox.hidden = true;
      }
    }
  }

  function bindHiggsfieldEvents() {
    if ($("btn-mode-normal")) {
      $("btn-mode-normal").addEventListener("click", () => setWorkspaceMode("normal"));
    }
    if ($("btn-mode-higgsfield")) {
      $("btn-mode-higgsfield").addEventListener("click", () => setWorkspaceMode("higgsfield"));
    }

    document.querySelectorAll(".hf-subtab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (state.workspaceMode !== "higgsfield") setWorkspaceMode("higgsfield");
        document.querySelectorAll(".hf-subtab-btn").forEach((b) => b.classList.toggle("active", b === btn));
        document.querySelectorAll(".hf-subpanel").forEach((p) => {
          p.hidden = p.id !== btn.dataset.hfSub;
        });
        document.querySelectorAll(".view-panel").forEach((p) => {
          p.classList.toggle("active", p.id === "tab-higgsfield");
        });
      });
    });

    if ($("btn-hf-run-director")) {
      $("btn-hf-run-director").addEventListener("click", async () => {
        try {
          const cids = [$("hf-dir-char1").value, $("hf-dir-char2").value].filter(Boolean);
          const dlines = [];
          if ($("hf-dir-line1").value.trim()) {
            dlines.push({
              character_id: $("hf-dir-char1").value || null,
              voice_id: $("hf-dir-voice1").value,
              target_face: cids.length > 1 ? "leftmost" : "largest",
              text: $("hf-dir-line1").value.trim(),
            });
          }
          if ($("hf-dir-line2").value.trim()) {
            dlines.push({
              character_id: $("hf-dir-char2").value || null,
              voice_id: $("hf-dir-voice2").value,
              target_face: "rightmost",
              text: $("hf-dir-line2").value.trim(),
            });
          }
          const r = await api("/api/higgsfield/scene-director", {
            method: "POST",
            body: JSON.stringify({
              scene_prompt: $("hf-dir-prompt").value,
              camera_id: $("hf-dir-camera").value,
              lens_id: $("hf-dir-lens").value,
              focal_mm: $("hf-dir-focal").value,
              aperture: $("hf-dir-aperture").value,
              lighting_id: $("hf-dir-lighting").value,
              motion_id: $("hf-dir-motion").value,
              video_engine_mode: $("hf-dir-engine").value,
              aspect_ratio: $("hf-dir-aspect").value,
              duration_seconds: Number($("hf-dir-dur").value || 4.5),
              base_image_rel: $("hf-dir-base-img").value || null,
              character_ids: cids,
              dialogue_lines: dlines,
              ambience_preset: $("hf-dir-ambience").value,
              apply_lipsync: $("hf-dir-lipsync").checked,
              image_lora_ids: $("hf-dir-img-lora") && $("hf-dir-img-lora").value ? [$("hf-dir-img-lora").value] : [],
              video_lora_ids: $("hf-dir-vid-lora") && $("hf-dir-vid-lora").value ? [$("hf-dir-vid-lora").value] : [],
            }),
          });
          watchJob(r.job);
          toast("🎬 Auto-Director Higgsfield rodando escena...", "info");
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    if ($("btn-hf-shoot-cinema")) {
      $("btn-hf-shoot-cinema").addEventListener("click", async () => {
        try {
          const cid = $("hf-cin-char").value;
          const r = await api("/api/higgsfield/cinema-shot", {
            method: "POST",
            body: JSON.stringify({
              prompt: $("hf-cin-prompt").value,
              camera_id: $("hf-cin-camera").value,
              lens_id: $("hf-cin-lens").value,
              focal_mm: $("hf-cin-focal").value,
              aperture: $("hf-cin-aperture").value,
              lighting_id: $("hf-cin-lighting").value,
              aspect_ratio: $("hf-cin-aspect").value,
              character_ids: cid ? [cid] : [],
              image_lora_ids: $("hf-cin-img-lora") && $("hf-cin-img-lora").value ? [$("hf-cin-img-lora").value] : [],
              seed: Number($("hf-cin-seed").value || 0),
            }),
          });
          watchJob(r.job);
          toast("📸 Disparando fotograma Cinema Studio...", "info");
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    if ($("btn-hf-run-motion")) {
      $("btn-hf-run-motion").addEventListener("click", async () => {
        try {
          const r = await api("/api/higgsfield/motion-clip", {
            method: "POST",
            body: JSON.stringify({
              image_rel: $("hf-mot-img").value,
              end_image_rel: $("hf-mot-end-img").value || null,
              prompt: $("hf-mot-prompt").value,
              motion_id: $("hf-mot-selected").value,
              engine_mode: $("hf-mot-engine").value,
              aspect_ratio: $("hf-mot-aspect").value,
              duration_seconds: Number($("hf-mot-dur").value || 4.5),
              video_lora_ids: $("hf-mot-vid-lora") && $("hf-mot-vid-lora").value ? [$("hf-mot-vid-lora").value] : [],
            }),
          });
          watchJob(r.job);
          toast("🌪️ Animando cámara Higgsfield...", "info");
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    if ($("btn-hf-save-char")) {
      $("btn-hf-save-char").addEventListener("click", async () => {
        try {
          const traits = {
            gender_age: $("hf-trait-age").value,
            hair: $("hf-trait-hair").value,
            outfit: $("hf-trait-outfit").value,
          };
          const r = await api("/api/higgsfield/characters", {
            method: "POST",
            body: JSON.stringify({
              name: $("hf-char-name").value,
              traits,
              anchor_image_rel: $("hf-char-anchor").value,
              voice_id: $("hf-char-voice").value,
              default_face_side: $("hf-char-side").value,
            }),
          });
          $("hf-char-name").value = "";
          renderHiggsfieldCharacters(r.characters || []);
          toast("👤 Personaje guardado en Casting Vault.", "success");
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    if ($("btn-hf-smart-clip")) {
      $("btn-hf-smart-clip").addEventListener("click", async () => {
        try {
          const r = await api("/api/higgsfield/smart-clip", {
            method: "POST",
            body: JSON.stringify({
              video_rel: $("hf-clip-video").value,
              target_aspect: $("hf-clip-aspect").value,
              start_sec: Number($("hf-clip-start").value || 0),
              end_sec: Number($("hf-clip-end").value || 0),
            }),
          });
          watchJob(r.job);
          toast("✂️ Reencuadrando clip inteligente...", "info");
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    if ($("btn-hf-send-to-montage")) {
      $("btn-hf-send-to-montage").addEventListener("click", () => {
        if (!state.lastHfResult) return;
        state.timeline.push({
          rel_path: state.lastHfResult.rel_path,
          filename: state.lastHfResult.filename,
          type: state.lastHfResult.type || "video",
          duration_seconds: state.lastHfResult.duration_seconds || 4.0,
          motion: "zoom_in",
          audio_rel: "",
        });
        renderTimeline();
        toast("Añadido a la línea de tiempo de Montaje Final.", "success");
      });
    }
  }

  // ==================== ARRANQUE ====================
  async function init() {
    setupMaskCanvas();
    bindEvents();
    bindHiggsfieldEvents();
    await Promise.all([loadCatalog(), loadVoices(), loadAssets()]);
    await loadHiggsfieldCatalog();
    if ($("select-image-model")) applyImageModelDefaults($("select-image-model").value);
    setInterval(() => {
      if (document.visibilityState !== "hidden" && $("tab-catalog").classList.contains("active")) refreshDiscoveryStatus();
    }, 30000);
    if ($("select-video-model") && $("select-video-model").value) {
      applyVideoModelDefaults($("select-video-model").value);
    }
    syncVideoSubmodeUI();
    startAdaptivePolling();
  }

  document.addEventListener("DOMContentLoaded", () => {
  const s = $("btn-ghc-start"); if (s) s.addEventListener("click", startGitHubChannel);
  const t = $("btn-ghc-stop"); if (t) t.addEventListener("click", stopGitHubChannel);
  const c = $("btn-ghc-copy");
  if (c) c.addEventListener("click", () => {
    const box = $("ghc-handoff");
    if (box && box.value) {
      navigator.clipboard?.writeText(box.value);
      toast("Datos del canal copiados: pégalos en el chat del agente", "success");
    }
  });
});
document.addEventListener("DOMContentLoaded", init);
})();

// ── Canal GitHub para agentes sin túnel (arena.ai) ───────────────────────────
async function refreshGitHubChannel() {
  const status = $("ghc-status");
  const startBtn = $("btn-ghc-start");
  const stopBtn = $("btn-ghc-stop");
  const copyBtn = $("btn-ghc-copy");
  const box = $("ghc-handoff");
  if (!status) return;
  try {
    const ch = await api("/api/mcp/github-channel", { headers: { "X-Studio-UI": "1" } });
    if (ch.running) {
      status.innerHTML = `🟢 <b>Activo</b> · ${ch.polls || 0} sondeos · buzón ${ch.gist_id ? "privado (gist)" : ch.backend}` +
        (ch.last_error ? ` · <span style="color:#f88">${ch.last_error}</span>` : "");
      if (startBtn) startBtn.hidden = true;
      if (stopBtn) stopBtn.hidden = false;
      if (copyBtn) copyBtn.hidden = false;
      if (box && ch.agent) {
        box.hidden = false;
        box.value = JSON.stringify(ch.agent, null, 2);
      }
    } else {
      status.innerHTML = ch.token_present
        ? "⚪ Apagado · ya hay un token guardado: basta pulsar «Encender»."
        : "⚪ Apagado · pega un token de GitHub con permiso <code>gist</code> y pulsa «Encender».";
      if (startBtn) startBtn.hidden = false;
      if (stopBtn) stopBtn.hidden = true;
      if (copyBtn) copyBtn.hidden = true;
      if (box) box.hidden = true;
    }
  } catch (_e) {
    status.textContent = "Canal no disponible en esta versión.";
  }
}

async function startGitHubChannel() {
  const status = $("ghc-status");
  const tokenEl = $("ghc-token");
  const startBtn = $("btn-ghc-start");
  if (status) status.textContent = "Encendiendo… (creando el buzón privado)";
  if (startBtn) startBtn.disabled = true;
  try {
    const out = await api("/api/mcp/github-channel/start", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Studio-UI": "1" },
      body: JSON.stringify({ gh_token: tokenEl ? tokenEl.value.trim() : "" }),
    });
    if (tokenEl) tokenEl.value = "";
    if (!out.ok) {
      if (status) status.innerHTML = `⚠️ ${out.error || out.error_code || "no se pudo encender"}` +
        (out.hint ? `<br><span style="opacity:.8">${out.hint}</span>` : "");
    } else {
      toast("Canal para arena.ai encendido", "success");
    }
  } catch (e) {
    if (status) status.textContent = `⚠️ ${e.message}`;
  } finally {
    if (startBtn) startBtn.disabled = false;
    refreshGitHubChannel();
  }
}

async function stopGitHubChannel() {
  try {
    await api("/api/mcp/github-channel/stop", { method: "POST", headers: { "X-Studio-UI": "1" } });
    toast("Canal detenido", "info");
  } catch (_e) {}
  refreshGitHubChannel();
}
