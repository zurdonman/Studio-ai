/* Exercises the real UI with deterministic API responses; no model/GPU downloads. */
const fs = require("fs");
const {JSDOM, VirtualConsole} = require("jsdom");
const wait = (ms) => new Promise(r => setTimeout(r, ms));
const model = (id, fit, extra = {}) => ({
  id, name: id, category: "video", engine: "wan_t2v", video_modes: ["t2v", "i2v"],
  estimated_gb: 2, default_steps: 25, default_cfg: 3, default_fps: 24, default_duration_sec: 5,
  official_spec: {landscape: [640, 384], portrait: [384, 640], default_steps: 25, default_cfg: 3, default_fps: 24, default_duration_sec: 5},
  compatibility: {compatible: fit, label: fit ? "estimado" : "no apto", tier: fit ? "normal" : "incompatible"}, ...extra
});
(async () => {
  const errors = [], requests = [];
  let statusEnabled = false;
  const runningJob = (id, title, kind, step, progress) => ({id, title, kind, current_step: step, progress, status: "running", elapsed_seconds: 12, speed_text: "1.2 s/it", logs: []});
  const imageJob = runningJob("image", "Imagen Qwen", "image", "Render imagen", 22);
  const ggufJob = runningJob("gguf-test", "LTX 2.3 GGUF Q4 · audio y vídeo (experimental)", "video", "Difusión: paso 4/8", 50);
  const vc = new VirtualConsole();
  vc.on("jsdomError", e => errors.push(e.message));
  const dom = new JSDOM(fs.readFileSync("studio/static/index.html", "utf8"), {
    url: "http://studio.test/", runScripts: "outside-only", pretendToBeVisual: true, virtualConsole: vc,
  });
  const w = dom.window;
  w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){}, drawImage(){}, getImageData(){return {data:[]}}, putImageData(){}});
  w.confirm = () => true;
  w.fetch = async (url, opts={}) => {
    const u = new URL(url, "http://studio.test");
    requests.push({path:u.pathname, method:opts.method || "GET", body:opts.body});
    let data = {};
    if (u.pathname === "/api/catalog") {
      const profile = u.searchParams.get("gpu_profile");
      if (profile === "L4") await wait(60);
      if (profile === "T4") await wait(5);
      data = {models: [model("heavy", profile === "auto", {downloaded:true,loaded_in_memory:true}),
                       model("fits", profile !== "T4"), model("unknown", false),
                       model("ltx-2.3-distilled-gguf-q4-experimental", false, {name:"LTX 2.3 GGUF Q4", engine:"ltx_gguf", ltx_distilled:true, execution_validation:{status:"observed",label:"✅ Generación I2V probada en L4",scope:"20 s con caché; no garantiza esta sesión"}}),
                       model("ltx-2.5-distilled-gguf-ud-q2-experimental", false, {name:"LTX 2.5 GGUF UD-Q2", engine:"ltx_gguf", ltx_distilled:true})],
              active_gpu_profile:{id:profile,short:profile}, loaded_slots:[]};
    } else if (u.pathname === "/api/catalog/hf-search") {
      const compatOnly = u.searchParams.get("only_compatible") === "true";
      await wait(20);
      data = {items: (compatOnly ? ["fits"] : ["heavy", "fits"]).map(id => ({repo_id: id, category:"video", compatibility:{compatible:id === "fits"}, support:{status:"reviewed_recipe"}, recipes:[{model_id:id}]})), source_mode:"cache"};
    } else if (u.pathname === "/api/assets") {
      data = {uploads:[], images:[], videos:[], audio:[], montages:[]};
    } else if (u.pathname === "/api/tts/voices") data = {voices:[]};
    else if (u.pathname === "/api/higgsfield/catalog") data = {cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    else if (u.pathname === "/api/status") {
      if (!statusEnabled) throw new Error("poll not needed");
      data = {hardware:{gpu:{has_gpu:false}, ram:{used_gb:1,total_gb:4,percent:25}, disk:{free_gb:9,total_gb:10,percent:10}, credits:{remaining_credits:0,estimated_hours_left:null,hourly_rate:0,session_spent:0}, ephemeral_disk_notice:""}, memory:{loaded_slots:[]}, jobs:[imageJob,ggufJob], active_jobs_count:2, hf_auth:{}, runtime:{}};
    }
    else if (u.pathname === "/api/video/ltx-gguf") { statusEnabled = true; data = {job:{id:"gguf-test"}}; }
    return {ok:true, json: async()=>data};
  };
  w.eval(fs.readFileSync("studio/static/app.js", "utf8"));
  await wait(120);
  const $ = id => w.document.getElementById(id);
  const ids = () => Array.from($("select-video-model").options, o => o.value);
  const pill = id => w.document.querySelector(`#vid-gpu-pills [data-gpu="${id}"]`);
  if (!ids().includes("heavy") || !ids().includes("fits")) throw Error("Initial unfiltered catalogue missing entries");
  w.document.querySelector('[data-tab="tab-catalog"]').click();
  await wait(50);
  if (!$("hf-live-results").textContent.includes("heavy")) throw Error("Initial discovery results missing");
  $("select-video-model").value = "heavy";
  $("vid-steps").value = "8";
  pill("L4").click();
  if ($("hf-live-results").textContent.includes("heavy")) throw Error("Stale discovery results remained visible after filter change");
  if (!$("btn-prepare-video").disabled || !$("btn-run-video").disabled) throw Error("Controls not disabled while loading");
  await wait(100);
  if (!$("chk-vid-only-compat").checked || !$("chk-cat-only-compat").checked) throw Error("L4 did not enable filters");
  if (ids().join() !== "fits") throw Error(`L4 options incorrect: ${ids()}`);
  if (Number($("vid-steps").value) !== 25) throw Error("New selection did not reset official defaults");
  if (w.document.querySelectorAll(".vm-card").length !== 1 || $("catalog-grid").textContent.includes("heavy")) throw Error("Filtered cards still visible");
  if ($("hf-live-results").textContent.includes("heavy") || !$("hf-live-results").textContent.includes("fits")) throw Error("Discovery did not honor L4 filter");
  pill("T4").click();
  await wait(30);
  if (ids().length || !$("video-model-cards").textContent.includes("No hay modelos")) throw Error("Empty filter silently returned models");
  if (!$("btn-prepare-video").disabled || !$("btn-run-video").disabled) throw Error("Empty filter allows actions");
  const ggufPanel = $("ltx-gguf-panel");
  if (!ggufPanel || ggufPanel.hidden || ggufPanel.closest("details") ||
      !(ggufPanel.compareDocumentPosition($("vid-gpu-pills")) & w.Node.DOCUMENT_POSITION_FOLLOWING))
    throw Error("GGUF experiment not prominently visible above GPU filter");
  if (ids().includes("ltx-2.3-distilled-gguf-q4-experimental")) throw Error("Unknown GGUF advertised as fitting the GPU filter");
  $("btn-prepare-video").dispatchEvent(new w.Event("click"));
  $("btn-run-video").dispatchEvent(new w.Event("click"));
  await wait(20);
  if (requests.some(r => r.method === "POST" && ["/api/memory/load", "/api/video/generate"].includes(r.path))) throw Error("Empty selection submitted a default model");
  $("chk-vid-only-compat").click();
  if (ids().length !== 5) throw Error("Unchecking filter did not restore catalogue including GGUF");
  const ggufId = "ltx-2.3-distilled-gguf-q4-experimental";
  const card = w.document.querySelector(`[data-vmid="${ggufId}"]`);
  if (!card) throw Error("GGUF card missing");
  if (!card.textContent.includes("✅ Generación I2V probada en L4") || !$("ltx-gguf-panel").textContent.includes("✅ LTX")) throw Error("Scoped GGUF validation badge missing");
  card.click();
  if ($("select-video-model").value !== ggufId || !$("btn-prepare-video").textContent.includes("GGUF")) throw Error("GGUF selection or prepare label missing");
  $("btn-prepare-video").click();
  await wait(20);
  if (!requests.some(r => r.method === "POST" && r.path === "/api/video/ltx-gguf/prepare")) throw Error("GGUF prepare used wrong endpoint");
  $("vid-prompt").value = "";
  $("btn-run-video").click();
  if ($("vid-gguf-feedback").hidden || !$("vid-gguf-feedback").textContent.includes("prompt")) throw Error("Missing prompt should remain visible beside generate");
  if (requests.some(r => r.method === "POST" && r.path === "/api/video/ltx-gguf")) throw Error("Missing prompt submitted GGUF request");
  $("vsub-t2v").click();
  if ($("select-video-model").value !== ggufId) throw Error("Switching supported submode changed GGUF selection");
  $("vid-prompt").value = "A person waves";
  $("vid-width").value = "512"; $("vid-height").value = "320";
  $("vid-duration").value = "20"; $("vid-fps").value = "24";
  $("vid-gguf-spanish").checked = true; $("vid-gguf-strength").value = ".7";
  if ($("vid-duration").max !== "20" || $("vid-gguf-guidance").hidden) throw Error("GGUF experimental options unavailable");
  if (!$("vid-steps").disabled || !$("vid-cfg").disabled) throw Error("GGUF fixed settings not disabled");
  $("btn-run-video").click();
  await wait(20);
  const ggufRequest = requests.find(r => r.path === "/api/video/ltx-gguf" && r.method === "POST");
  if (!ggufRequest || JSON.parse(ggufRequest.body).prompt !== "A person waves" || JSON.parse(ggufRequest.body).duration_sec !== 20 || JSON.parse(ggufRequest.body).width !== 512 || JSON.parse(ggufRequest.body).speech_language !== "es" || JSON.parse(ggufRequest.body).image_strength !== .7) throw Error("GGUF generation used wrong endpoint or parameters");
  if ($("vid-gguf-feedback").hidden || !$("vid-gguf-feedback").textContent.includes("enviada")) throw Error("GGUF submission confirmation missing");
  if ($("vid-gguf-live-progress").hidden || $("vid-gguf-live-step").textContent !== "Difusión: paso 4/8" || $("vid-gguf-live-bar").style.width !== "50%") throw Error("GGUF concurrent progress not displayed");
  if ($("img-inline-progress").hidden || !$("img-iep-step").textContent.includes("Render imagen")) throw Error("Concurrent image progress hidden");
  $("btn-cancel-gguf").click();
  await wait(20);
  if (!requests.some(r => r.path === "/api/jobs/gguf-test/cancel" && r.method === "POST")) throw Error("GGUF cancel targeted wrong job");
  const q2 = "ltx-2.5-distilled-gguf-ud-q2-experimental";
  const q2card = w.document.querySelector(`[data-vmid="${q2}"]`);
  if (!q2card || q2card.textContent.includes("✅ Generación")) throw Error("New LTX 2.5 should be visible but not marked tested");
  q2card.click();
  $("btn-run-video").click();
  await wait(30);
  const q2request = requests.filter(r => r.path === "/api/video/ltx-gguf" && r.method === "POST").at(-1);
  if (!q2request || JSON.parse(q2request.body).model_id !== q2) throw Error("LTX 2.5 was not routed to GGUF API");
  $("btn-prepare-video").click();
  await wait(20);
  const q2prep = requests.filter(r => r.path === "/api/video/ltx-gguf/prepare").at(-1);
  if (!q2prep) throw Error("LTX 2.5 prepare missing");
  pill("L4").click(); pill("T4").click();
  await wait(110);
  if (ids().length !== 0 || !w.document.querySelector('#vid-gpu-pills [data-gpu="T4"]').classList.contains("active")) throw Error("Stale profile response overrode newer selection");
  if (errors.length) throw Error(`DOM errors: ${errors.join('; ')}`);
  console.log("L4 filter, downloaded/loaded exclusion, empty state, actions and profile races: OK");
  w.close();
})().catch(e => { console.error(e); process.exitCode = 1; });
