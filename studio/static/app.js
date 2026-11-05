/* Studio IA — Lógica de Interfaz Web */
(function () {
  "use strict";

  const state = {
    catalog: [],
    catalogFilter: "all",
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
      $("img-engine-state").textContent = "No descargado todavía (~12.2 GB)";
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
        (topRunning.title && /wan|ltx|vídeo|video/i.test(topRunning.title));
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
        btnVidPrep.textContent = "⚡ Preparar Modelo de Vídeo";
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
  async function loadCatalog() {
    const data = await api("/api/catalog");
    state.catalog = data.models || [];
    renderCatalog();
    populateVideoModelSelect();
    if (state.lastHardware) {
      updateEngineBars(data.loaded_slots || []);
    }
  }

  function populateVideoModelSelect() {
    const sel = $("select-video-model");
    const prev = sel.value;
    const vModels = state.catalog.filter((m) => m.category === "video");
    sel.innerHTML = vModels
      .map((m) => {
        const st = m.loaded_in_memory ? "🟢 En VRAM" : m.downloaded ? "🟡 En disco" : `⬇ ${m.estimated_gb} GB`;
        return `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)} [${st}]</option>`;
      })
      .join("");
    if (prev && vModels.some((m) => m.id === prev)) {
      sel.value = prev;
    }
  }

  function renderCatalog() {
    const grid = $("catalog-grid");
    const filtered = state.catalog.filter((m) => {
      if (state.catalogFilter === "all") return true;
      if (state.catalogFilter === "image") return m.category === "image" || m.category === "lora";
      return m.category === state.catalogFilter;
    });

    grid.innerHTML = filtered
      .map((m) => {
        const compatClass = m.compatibility.status === "ok" ? "ok" : "warn";
        const diskLabel = m.builtin_no_download
          ? "Integrado (0 GB)"
          : m.downloaded
          ? `En disco: ${m.actual_disk_gb || m.estimated_gb} GB`
          : `Peso est.: ~${m.estimated_gb} GB (Libre: ${m.free_disk_gb} GB)`;

        let btns = "";
        if (!m.builtin_no_download) {
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
                <span class="mc-tag ${compatClass}">${escapeHtml(m.compatibility.label)}</span>
              </div>
              <h4 class="mc-title">${escapeHtml(m.name)}</h4>
              <p class="mc-desc">${escapeHtml(m.description || "")}</p>
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
      const cvs = $("mask-canvas");
      cvs.width = imgEl.naturalWidth || 1024;
      cvs.height = imgEl.naturalHeight || 1024;
      state.mask.history = [];
      state.mask.dirty = false;
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
            <button class="mini-btn" data-act="vid" data-idx="${idx}">Vídeo</button>
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
          $("vid-source-image").value = asset.rel_path;
          updateVideoSourceThumb();
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
    imgSel.innerHTML =
      `<option value="">-- Selecciona una imagen --</option>` +
      allImgs
        .map((im) => `<option value="${escapeHtml(im.rel_path)}">${escapeHtml(im.filename)}</option>`)
        .join("");
    if (prevImg && allImgs.some((i) => i.rel_path === prevImg)) {
      imgSel.value = prevImg;
    } else if (allImgs.length > 0 && !imgSel.value) {
      imgSel.value = allImgs[0].rel_path;
    }
    updateVideoSourceThumb();

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
    const rel = $("vid-source-image").value;
    const box = $("vid-source-preview-box");
    if (!rel) {
      box.innerHTML = `<span class="muted">Selecciona o sube una imagen</span>`;
      return;
    }
    box.innerHTML = `<img src="/media/${escapeHtml(rel)}" alt="Origen vídeo" />`;
  }

  function showVideoResult(meta) {
    state.lastVideoResult = meta;
    $("vid-result-empty").hidden = true;
    $("vid-result-content").hidden = false;
    const player = $("vid-result-player");
    player.src = `${meta.media_url}?t=${Date.now()}`;
    $("btn-vid-download").href = meta.media_url;
    $("btn-vid-download").download = meta.filename;
    $("vid-provenance").innerHTML = `
      <span><strong>Motor:</strong> ${escapeHtml(meta.model_name || meta.model_id)}</span>
      <span><strong>Duración:</strong> ${meta.duration_seconds}s</span>
      <span><strong>Resolución:</strong> ${meta.width}×${meta.height} @ ${meta.fps}fps</span>
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
    $("btn-prepare-video").addEventListener("click", () => {
      const mid = $("select-video-model").value || "wan22-ti2v-5b";
      loadModelIntoMemory(mid, true);
    });
    $("btn-unload-video").addEventListener("click", () => unloadFromMemory("video"));
    $("vid-source-image").addEventListener("change", updateVideoSourceThumb);

    $("input-vid-upload-img").addEventListener("change", async (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      try {
        const asset = await uploadFile(f);
        $("vid-source-image").value = asset.rel_path;
        updateVideoSourceThumb();
      } catch (e) {
        toast(e.message, "error");
      }
      ev.target.value = "";
    });

    document.querySelectorAll("[data-vmode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.videoMode = btn.dataset.vmode;
        $("vmode-ai").classList.toggle("active", state.videoMode === "ai");
        $("vmode-ffmpeg").classList.toggle("active", state.videoMode === "ffmpeg_motion");
        $("vid-ai-fields").hidden = state.videoMode !== "ai";
        $("vid-ai-steps-row").hidden = state.videoMode !== "ai";
        $("vid-ffmpeg-fields").hidden = state.videoMode !== "ffmpeg_motion";
        if (state.videoMode === "ffmpeg_motion") {
          $("vid-width").value = 1280;
          $("vid-height").value = 720;
          $("vid-fps").value = 24;
        } else {
          $("vid-width").value = 832;
          $("vid-height").value = 480;
          $("vid-fps").value = 16;
        }
      });
    });

    $("btn-run-video").addEventListener("click", async () => {
      try {
        const res = await api("/api/video/generate", {
          method: "POST",
          body: JSON.stringify({
            mode: state.videoMode,
            prompt: $("vid-prompt").value,
            image_rel: $("vid-source-image").value || null,
            audio_rel: $("vid-source-audio").value || null,
            width: Number($("vid-width").value || 832),
            height: Number($("vid-height").value || 480),
            duration_sec: Number($("vid-duration").value || 4),
            fps: Number($("vid-fps").value || 16),
            steps: Number($("vid-steps").value || 20),
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

    // Pestaña 5: Catálogo y Añadir HF custom
    document.querySelectorAll(".filter-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        state.catalogFilter = chip.dataset.cat;
        document.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        renderCatalog();
      });
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
    startAdaptivePolling();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
