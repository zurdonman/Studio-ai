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
    hfAuth: { has_token: false, masked_token: null, username: null },
    sessionHfToken: "",
    assets: { uploads: [], images: [], videos: [], audio: [], montages: [] },
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

  function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tabId);
    });
    document.querySelectorAll(".view-panel").forEach((p) => {
      p.classList.toggle("active", p.id === tabId);
    });
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
      renderJobs(snap.jobs);
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
    $("nav-loaded-count").textContent = `${loadedSlots.length} en VRAM`;

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
    const qwenDef = state.catalog.find((m) => m.id === "qwen-image-edit-2511-pack");
    const imgDot = $("img-engine-dot");
    const btnQwen = $("btn-prepare-qwen");
    if (imgSlot) {
      imgDot.className = "status-dot loaded";
      $("img-engine-state").textContent = `Cargado en RAM/VRAM (${imgSlot.name})`;
      if (!btnQwen.disabled) btnQwen.textContent = "✅ Qwen + Lightning Listo en VRAM";
    } else if (qwenDef && qwenDef.downloaded) {
      imgDot.className = "status-dot downloaded";
      $("img-engine-state").textContent = "Descargado en disco · Sin cargar en VRAM";
      if (!btnQwen.disabled) btnQwen.textContent = "⚡ Cargar Qwen + Lightning en VRAM";
    } else {
      imgDot.className = "status-dot";
      $("img-engine-state").textContent = "No descargado todavía (~16.6 GB)";
      if (!btnQwen.disabled) btnQwen.textContent = "⬇⚡ Descargar y Cargar Qwen + Lightning (1 clic)";
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
    if (job.kind === "image" && res.media_url) {
      showImageResult(res);
      toast("Imagen lista y verificada.", "success");
    } else if (job.kind === "video" && res.media_url) {
      showVideoResult(res);
      toast("Clip de vídeo MP4 listo.", "success");
    } else if (job.kind === "tts" && res.media_url) {
      toast(`Locución lista (${res.duration_seconds}s).`, "success");
    } else if (job.kind === "montage" && res.media_url) {
      showMontageResult(res);
      toast(`Montaje MP4 completado (${res.duration_seconds}s).`, "success");
    } else if (job.kind === "download") {
      toast("Descarga completada y verificada en disco.", "success");
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
    syncGpuPillsUI();
    renderCatalog();
    populateVideoModelSelect();
    populateTtsEngineSelect();
    if (state.lastHardware) {
      updateEngineBars(data.loaded_slots || []);
    }
  }

  function populateTtsEngineSelect() {
    const sel = $("tts-engine-select");
    if (!sel) return;
    const prev = sel.value;
    const ttsModels = state.catalog.filter((m) => m.category === "tts");
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

  function populateVideoLoraSelect(modelId) {
    const loraBox = $("vid-lora-box");
    const loraSel = $("vid-lora-select");
    const scaleWrap = $("vid-lora-scale-wrap");
    const dlBtn = $("btn-dl-vid-lora");
    if (!loraBox || !loraSel) return;

    const compatLoras = state.catalog.filter(
      (m) =>
        m.category === "lora" &&
        m.lora_target === "video" &&
        (!Array.isArray(m.compatible_models) || m.compatible_models.includes(modelId))
    );

    if (compatLoras.length === 0) {
      loraBox.hidden = true;
      loraSel.innerHTML = `<option value="">Sin LoRA acelerador</option>`;
      loraSel.value = "";
      if (scaleWrap) scaleWrap.hidden = true;
      if (dlBtn) dlBtn.hidden = true;
      return;
    }

    const prev = loraSel.value;
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
  }

  function resetVideoStepsToRecommended() {
    const mid = $("select-video-model") ? $("select-video-model").value : "";
    const m = state.catalog.find((x) => x.id === mid);
    const spec = (m && m.official_spec) || {};
    const lid = $("vid-lora-select") ? $("vid-lora-select").value : "";
    const loraObj = lid ? state.catalog.find((x) => x.id === lid) : null;

    const recSteps = loraObj
      ? Number(loraObj.recommended_steps || 6)
      : Number(spec.default_steps || (m && m.default_steps) || 45);
    const recCfg = loraObj
      ? Number(loraObj.recommended_cfg || 1.0)
      : Number(spec.default_cfg || (m && m.default_cfg) || 5.0);

    if ($("vid-steps")) $("vid-steps").value = recSteps;
    if ($("vid-steps-slider")) $("vid-steps-slider").value = recSteps;
    if ($("vid-cfg")) $("vid-cfg").value = recCfg;
    if (loraObj && $("vid-lora-scale")) {
      $("vid-lora-scale").value = loraObj.default_lora_scale || 1.0;
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
    if (slider) slider.value = Math.max(4, Math.min(55, steps));

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
      if (steps <= 10) {
        const estMin = Math.max(0.6, Math.round(steps * (mid.includes("5b") ? 0.2 : 0.08) * 10) / 10);
        noteEl.className = "vid-steps-note turbo";
        noteEl.textContent = `⚡ LoRA Acelerador activo (${steps} pasos · CFG ${cfg}): genera el clip ~12× más rápido (~${estMin} min en GPU L4) sin deformar.`;
      } else {
        noteEl.className = "vid-steps-note warn";
        noteEl.textContent = `⚠️ Con el LoRA Acelerador activo lo ideal son 4–8 pasos y CFG 1.0 (más pasos pueden sobresaturar). Pulsa «↺ Resetear pasos».`;
      }
      return;
    }

    if (defSteps <= 10) {
      noteEl.className = "vid-steps-note turbo";
      noteEl.textContent = `⚡ Modelo ya destilado de fábrica (${steps} pasos · CFG ${cfg}): velocidad máxima nativa (~1 min en L4).`;
      return;
    }

    if (steps < minSafe) {
      noteEl.className = "vid-steps-note warn";
      noteEl.textContent = `⚠️ Menos de ${minSafe} pasos sin LoRA puede deformar piernas/brazos o dejar borrones. Usa ≥${fastSafe} pasos, pulsa «↺ Resetear pasos» o elige arriba el ⚡ LoRA Turbo (6 pasos).`;
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
    let vModels = state.catalog.filter((m) => m.category === "video");
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
      if (state.catOnlyCompat && m.compatibility && !m.compatibility.compatible) return false;
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
        const comp = m.compatibility || {};
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
        if (m.category === "lora") {
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
          btns += `<span class="mc-tag ok">Siempre activo sin ocupar RAM/VRAM</span>`;
        }

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
              ${matrixHtml}
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

  function evaluateClientGpuSticker(estVramMin, category) {
    const prof = state.activeGpuProfile || { gpu_name: "L4 (22.5 GB)", vram_gb: 22.5, has_gpu: true };
    const vram = Number(prof.vram_gb || 0);
    const gname = prof.gpu_name || "GPU";
    const recV = estVramMin * 1.45;
    if (category === "lora") {
      if (!prof.has_gpu || vram < 12) {
        return { tier: "incompatible", compatible: false, sticker: "🔴 Requiere GPU ≥14 GB", label: `🔴 Requiere GPU ≥14 GB (${gname})` };
      }
      return { tier: "excellent", compatible: true, sticker: "🟢 Excelente", label: `🟢 Excelente en ${gname}` };
    }
    if (!prof.has_gpu || vram <= 0.1) {
      return { tier: "incompatible", compatible: false, sticker: "🔴 Requiere GPU", label: `🔴 Requiere GPU (~${Math.round(estVramMin)} GB VRAM)` };
    }
    if (vram >= recV) {
      return { tier: "excellent", compatible: true, sticker: "🟢 Excelente", label: `🟢 Excelente en ${gname}` };
    }
    if (vram >= estVramMin * 1.12) {
      return { tier: "normal", compatible: true, sticker: "🟡 Normal", label: `🟡 Normal en ${gname} (con CPU Offload)` };
    }
    if (vram >= estVramMin * 0.95) {
      return { tier: "slow", compatible: true, sticker: "🟠 Lento", label: `🟠 Lento en ${gname}` };
    }
    return { tier: "incompatible", compatible: false, sticker: "🔴 No cabe", label: `🔴 No cabe en ${gname} (pide ~${Math.round(estVramMin)} GB)` };
  }

  async function browserDirectHfSearch(query, category, signal) {
    const taskMap = {
      video: ["image-to-video", "text-to-video"],
      image: ["image-to-image", "text-to-image"],
      lora: [null],
      tts: ["text-to-speech"],
    };
    const tasks = taskMap[category] || ["image-to-video"];
    const headers = {};
    if (state.sessionHfToken) {
      headers["Authorization"] = `Bearer ${state.sessionHfToken}`;
    }

    const raw = [];
    for (const t of tasks) {
      const params = new URLSearchParams();
      const sq = query || (category === "lora" ? "Qwen-Image-Edit lora" : "");
      if (sq) params.set("search", sq);
      if (t) params.set("pipeline_tag", t);
      if (category === "tts") params.set("library", "transformers");
      else if (category !== "lora") params.set("library", "diffusers");
      params.set("sort", "downloads");
      params.set("direction", "-1");
      params.set("limit", "14");
      ["safetensors", "gated", "tags", "downloads", "likes", "pipeline_tag"].forEach((ex) =>
        params.append("expand[]", ex)
      );
      const r = await fetch(`https://huggingface.co/api/models?${params.toString()}`, {
        headers,
        signal,
      });
      if (r.ok) {
        const arr = await r.json();
        if (Array.isArray(arr)) raw.push(...arr);
      }
    }

    const seen = new Set();
    const out = [];
    for (const m of raw) {
      const repoId = m.id || "";
      if (!repoId || seen.has(repoId.toLowerCase())) continue;
      seen.add(repoId.toLowerCase());

      const tags = (m.tags || []).map((x) => String(x).toLowerCase());
      if ((category === "video" || category === "image") && tags.includes("gguf") && !tags.includes("diffusers")) {
        continue;
      }

      const st = m.safetensors || null;
      let paramsTotal = 0;
      let bytesEst = 0;
      if (st && st.parameters) {
        paramsTotal = Number(st.total || 0);
        for (const [dk, cnt] of Object.entries(st.parameters)) {
          const c = Number(cnt || 0);
          const up = String(dk).toUpperCase();
          if (up.includes("F32")) bytesEst += c * 4;
          else if (up.includes("BF16") || up.includes("F16")) bytesEst += c * 2;
          else bytesEst += c * 1;
        }
      }

      const lowId = repoId.toLowerCase();
      let estDiskGb = 14.0;
      let estVramMin = 12.0;
      if (bytesEst > 0) {
        estDiskGb = Math.max(0.2, Math.round((bytesEst / 1073741824) * 10) / 10);
        estVramMin = Math.max(
          1.0,
          Math.round(((paramsTotal * 2.0) / 1073741824 + (category === "lora" ? 0.3 : 3.0)) * 10) / 10
        );
        if (lowId.includes("4bit") || lowId.includes("nf4")) {
          estVramMin = Math.max(4.0, Math.round(((paramsTotal * 0.65) / 1073741824 + 3.0) * 10) / 10);
        }
      } else if (category === "lora" || lowId.includes("lora")) {
        estDiskGb = 0.6;
        estVramMin = 0.4;
      } else if (lowId.includes("14b")) {
        estDiskGb = 32.0;
        estVramMin = 22.0;
      } else if (lowId.includes("19b") || lowId.includes("27b")) {
        estDiskGb = 48.0;
        estVramMin = 26.0;
      } else if (lowId.includes("5b")) {
        estDiskGb = 18.0;
        estVramMin = 13.5;
      } else if (lowId.includes("1.3b") || lowId.includes("2b")) {
        estDiskGb = 12.0;
        estVramMin = 8.5;
      } else if (category === "tts") {
        estDiskGb = 1.2;
        estVramMin = 1.5;
      }

      const compat = evaluateClientGpuSticker(estVramMin, category);
      if (state.catOnlyCompat && !compat.compatible) continue;

      const isGated = Boolean(m.gated);
      out.push({
        repo_id: repoId,
        name: repoId.split("/").pop(),
        author: repoId.split("/")[0],
        category,
        downloads: Number(m.downloads || 0),
        likes: Number(m.likes || 0),
        estimated_gb: estDiskGb,
        min_vram_gb: estVramMin,
        compatibility: compat,
        gated: isGated,
        gated_locked: isGated && !(state.hfAuth.has_token || Boolean(state.sessionHfToken)),
      });
    }
    return out.slice(0, 14);
  }

  let _hfDebounceTimer = null;
  let _hfAbortCtrl = null;

  function scheduleRealtimeHfSearch() {
    if (_hfDebounceTimer) clearTimeout(_hfDebounceTimer);
    _hfDebounceTimer = setTimeout(() => {
      runHfLiveSearch();
    }, 280);
  }

  async function runHfLiveSearch(customQuery = null, customCat = null) {
    const qInput = $("hf-live-query");
    const cSelect = $("hf-live-cat");
    const box = $("hf-live-results");
    const btn = $("btn-hf-live-search");
    const ind = $("hf-live-indicator");
    if (!box) return;

    if (customQuery !== null && qInput) qInput.value = customQuery;
    if (customCat !== null && cSelect) cSelect.value = customCat;

    const q = (qInput && qInput.value.trim()) || "";
    const cat = (cSelect && cSelect.value) || "video";

    if (_hfAbortCtrl) {
      _hfAbortCtrl.abort();
    }
    const ctrl = new AbortController();
    _hfAbortCtrl = ctrl;

    if (ind) {
      ind.classList.add("searching");
      ind.textContent = "⏳ Buscando en tiempo real...";
    }
    if (btn) btn.disabled = true;

    try {
      const resp = await fetch(
        `/api/catalog/hf-search?query=${encodeURIComponent(q)}&category=${encodeURIComponent(cat)}&gpu_profile=${encodeURIComponent(state.gpuProfile)}&only_compatible=${state.catOnlyCompat}`,
        { signal: ctrl.signal }
      );
      const data = await resp.json();
      let items = data.items || [];

      // Si el contenedor actual no sale directo a huggingface.co, consultar desde el navegador en tiempo real
      if (data.online === false) {
        try {
          const browserItems = await browserDirectHfSearch(q, cat, ctrl.signal);
          if (browserItems.length > 0) {
            items = browserItems;
          }
        } catch (_be) {
          // Mantiene resultados locales verificados si tampoco hubiera red en el navegador
        }
      }

      if (ctrl.signal.aborted) return;
      state.hfSearchResults = items;

      if (ind) {
        ind.classList.remove("searching");
        ind.textContent = `⚡ ${items.length} resultado(s) en vivo`;
      }

      if (items.length === 0) {
        box.innerHTML = `<div class="muted">Sin coincidencias para «${escapeHtml(q)}» con el filtro de GPU actual. Prueba otro término o cambia la GPU arriba.</div>`;
        return;
      }

      box.innerHTML = items
        .map((it, idx) => {
          const comp = it.compatibility || {};
          const already = state.catalog.some((m) =>
            (m.artifacts || []).some((a) => a.repo_id.toLowerCase() === it.repo_id.toLowerCase())
          );
          const gatedHtml = it.gated
            ? it.gated_locked
              ? `<span class="gated-pill locked" title="Requiere aceptar licencia en HF y activar tu Token HF">🔒 Pide Token HF</span>`
              : `<span class="gated-pill unlocked">🔓 Token HF listo</span>`
            : `<span class="gated-pill">🔓 Público</span>`;

          let actionHtml = "";
          if (already) {
            actionHtml = `<span class="mc-tag ok">✓ Ya en tu catálogo</span>`;
          } else if (it.gated_locked) {
            actionHtml = `<button type="button" class="btn btn-secondary btn-sm" data-hf-unlock="1">🔑 Poner Token HF para usar</button>`;
          } else {
            actionHtml = `
              <div class="row-gap" style="flex-wrap:wrap;">
                <button type="button" class="btn btn-secondary btn-sm" data-hf-import="${idx}">➕ Añadir</button>
                <button type="button" class="btn btn-primary btn-sm" data-hf-import-load="${idx}">⬇⚡ Añadir y Cargar</button>
              </div>
            `;
          }

          return `
            <div class="hf-live-card">
              <div>
                <div style="display:flex;justify-content:space-between;gap:6px;align-items:flex-start;">
                  <strong style="font-size:12.5px;word-break:break-word;">${escapeHtml(it.repo_id)}</strong>
                  <span class="muted" style="white-space:nowrap;">⬇ ${(it.downloads || 0).toLocaleString()}</span>
                </div>
                <div style="margin-top:6px;display:flex;gap:5px;flex-wrap:wrap;align-items:center;">
                  <span class="gpu-sticker ${escapeHtml(comp.tier || "normal")}">${escapeHtml(comp.label || comp.sticker || "")}</span>
                  ${gatedHtml}
                </div>
                <div class="muted" style="margin-top:5px;">
                  Peso aprox.: ~${it.estimated_gb} GB · VRAM mín.: ~${it.min_vram_gb} GB
                </div>
              </div>
              <div>${actionHtml}</div>
            </div>
          `;
        })
        .join("");

      box.querySelectorAll("[data-hf-unlock]").forEach((b) => {
        b.addEventListener("click", () => {
          $("hf-token-modal").hidden = false;
        });
      });

      const importItem = async (idx, andLoad) => {
        const item = state.hfSearchResults[idx];
        if (!item) return;
        try {
          const res = await api("/api/catalog/custom", {
            method: "POST",
            body: JSON.stringify({
              name: item.name || item.repo_id,
              repo_id: item.repo_id,
              category: item.category,
              estimated_gb: item.estimated_gb,
              min_vram_gb: item.min_vram_gb,
            }),
          });
          await loadCatalog();
          if (andLoad && res.model && res.model.id) {
            toast(`Descargando y cargando «${item.repo_id}»...`, "info");
            await loadModelIntoMemory(res.model.id, true);
          } else {
            toast(`«${item.repo_id}» añadido a tu catálogo.`, "success");
          }
          runHfLiveSearch();
        } catch (e) {
          toast(e.message, "error");
        }
      };

      box.querySelectorAll("[data-hf-import]").forEach((b) => {
        b.addEventListener("click", () => importItem(Number(b.dataset.hfImport), false));
      });
      box.querySelectorAll("[data-hf-import-load]").forEach((b) => {
        b.addEventListener("click", () => importItem(Number(b.dataset.hfImportLoad), true));
      });
    } catch (e) {
      if (e && e.name === "AbortError") return;
      if (ind) {
        ind.classList.remove("searching");
        ind.textContent = "⚡ En vivo al escribir";
      }
      box.innerHTML = `<div class="muted">Error consultando Hugging Face (${escapeHtml(e.message)}).</div>`;
    } finally {
      if (btn) btn.disabled = false;
    }
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
    state.assets = await api("/api/assets");
    renderImageGallery();
    renderVideoSelectsAndGallery();
    renderAudioList();
    renderMontagePickers();
    renderMontageGallery();
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
      const payload = {
        prompt: $("img-prompt").value,
        negative_prompt: $("img-neg-prompt").value,
        base_image_rel: state.baseImage ? state.baseImage.rel_path : null,
        reference_rels: refRels,
        mask_image_rel: maskRel,
        width: Number($("img-width").value || 1024),
        height: Number($("img-height").value || 1024),
        steps: Number($("img-steps").value || 4),
        guidance_scale: Number($("img-cfg").value || 4.0),
        seed: Number($("img-seed").value || 0),
        use_lightning_lora: $("chk-lightning").checked,
        use_nsfw_lora: $("chk-nsfw-lora").checked,
        nsfw_lora_weight: 1.0,
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
        endSel.value = allImgs[1].rel_path;
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
    $("btn-prepare-qwen").addEventListener("click", () =>
      loadModelIntoMemory("qwen-image-edit-2511-pack", true)
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
          toast(
            `LoRA «${loraObj.name.split("(")[0].trim()}» seleccionado: ajustados automáticamente ${loraObj.recommended_steps || 6} pasos y CFG ${loraObj.recommended_cfg || 1.0}.`,
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
      const f = ev.target.files[0];
      if (!f) return;
      try {
        const asset = await uploadFile(f);
        $("vid-source-image").value = asset.rel_path;
        updateVideoSourceThumb();
        toast(`Imagen inicial lista: ${asset.filename}`, "success");
      } catch (e) {
        toast(e.message, "error");
      }
      ev.target.value = "";
    });

    if ($("input-vid-upload-end-img")) {
      $("input-vid-upload-end-img").addEventListener("change", async (ev) => {
        const f = ev.target.files[0];
        if (!f) return;
        try {
          const asset = await uploadFile(f);
          if ($("vid-end-image")) $("vid-end-image").value = asset.rel_path;
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

        const res = await api("/api/video/generate", {
          method: "POST",
          body: JSON.stringify({
            mode: effectiveMode,
            video_submode: effectiveSubmode,
            model_id: chosenVidModel || null,
            video_lora_id: effectiveMode === "ai" ? chosenVidLora : null,
            video_lora_scale: chosenVidLoraScale,
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
          state.sessionHfToken = val;
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
          state.sessionHfToken = "";
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
      $("btn-hf-live-search").addEventListener("click", () => runHfLiveSearch());
    }
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
        toast("Repositorio de Hugging Face añadido al catálogo.", "success");
        await loadCatalog();
      } catch (e) {
        toast(e.message, "error");
      }
    });
  }

  // ==================== ARRANQUE ====================
  async function init() {
    setupMaskCanvas();
    bindEvents();
    await Promise.all([loadCatalog(), loadVoices(), loadAssets()]);
    if ($("select-video-model") && $("select-video-model").value) {
      applyVideoModelDefaults($("select-video-model").value);
    }
    syncVideoSubmodeUI();
    startAdaptivePolling();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
