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
                       model("fits", profile !== "T4"), model("unknown", false)],
              active_gpu_profile:{id:profile,short:profile}, loaded_slots:[]};
    } else if (u.pathname === "/api/catalog/hf-search") {
      const compatOnly = u.searchParams.get("only_compatible") === "true";
      await wait(20);
      data = {items: (compatOnly ? ["fits"] : ["heavy", "fits"]).map(id => ({repo_id: id, category:"video", compatibility:{compatible:id === "fits"}, support:{status:"reviewed_recipe"}, recipes:[{model_id:id}]})), source_mode:"cache"};
    } else if (u.pathname === "/api/assets") {
      data = {uploads:[], images:[], videos:[], audio:[], montages:[]};
    } else if (u.pathname === "/api/tts/voices") data = {voices:[]};
    else if (u.pathname === "/api/higgsfield/catalog") data = {cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    else if (u.pathname === "/api/status") { throw new Error("poll not needed"); }
    else if (u.pathname === "/api/video/ltx-gguf") data = {job:{id:"gguf-test"}};
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
  $("vsub-t2v").click();
  $("vid-prompt").value = "A person waves";
  $("btn-run-ltx-gguf").click();
  await wait(20);
  const ggufRequest = requests.find(r => r.path === "/api/video/ltx-gguf" && r.method === "POST");
  if (!ggufRequest || JSON.parse(ggufRequest.body).prompt !== "A person waves") throw Error("GGUF action not usable when GPU filter is empty");
  $("btn-prepare-video").dispatchEvent(new w.Event("click"));
  $("btn-run-video").dispatchEvent(new w.Event("click"));
  await wait(20);
  if (requests.some(r => r.method === "POST" && ["/api/memory/load", "/api/video/generate"].includes(r.path))) throw Error("Empty selection submitted a default model");
  $("chk-vid-only-compat").click();
  if (ids().length !== 3) throw Error("Unchecking filter did not restore catalogue");
  pill("L4").click(); pill("T4").click();
  await wait(110);
  if (ids().length !== 0 || !w.document.querySelector('#vid-gpu-pills [data-gpu="T4"]').classList.contains("active")) throw Error("Stale profile response overrode newer selection");
  if (errors.length) throw Error(`DOM errors: ${errors.join('; ')}`);
  console.log("L4 filter, downloaded/loaded exclusion, empty state, actions and profile races: OK");
  w.close();
})().catch(e => { console.error(e); process.exitCode = 1; });
