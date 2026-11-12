/* Simulate a completion during an old assets request; no GPU or model required. */
const fs = require('fs');
const {JSDOM, VirtualConsole} = require('jsdom');
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(test, description) {
  for (let i = 0; i < 100; i++) { if (test()) return; await wait(30); }
  throw Error('Timeout: ' + description);
}
(async () => {
  const errors = [], requests = [];
  const vc = new VirtualConsole();
  vc.on('jsdomError', e => errors.push(e.message));
  const dom = new JSDOM(fs.readFileSync('studio/static/index.html', 'utf8'), {
    url:'http://studio.test/', runScripts:'outside-only', pretendToBeVisual:true, virtualConsole:vc,
  });
  const w = dom.window, $ = id => w.document.getElementById(id);
  w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){},drawImage(){},getImageData(){return {data:[]}},putImageData(){}});
  w.confirm = () => true;
  const id = 'ltx-2.5-distilled-gguf-ud-q4-gemma-q5-experimental';
  const model = {id, name:'LTX 2.5 GGUF', category:'video', engine:'ltx_gguf', video_modes:['t2v','i2v'],
    estimated_gb:24, ltx_distilled:true, compatibility:{compatible:false},
    official_spec:{landscape:[512,320],portrait:[320,512],default_steps:8,default_cfg:1,default_fps:24,default_duration_sec:5}};
  const asset = {id:'clip', filename:'clip.mp4', rel_path:'outputs/videos/clip.mp4', media_url:'/media/outputs/videos/clip.mp4', type:'video', model_name:'LTX', duration_seconds:5, width:512, height:320, fps:24, steps:8};
  const job = {id:'job1', kind:'video', title:'LTX 2.5 GGUF', status:'running', progress:40, elapsed_seconds:30, current_step:'Generando', logs:[]};
  let running = false, completed = false, hold = false, release, catalogFailure = false, assetsFailure = false;
  w.fetch = async (url, opts={}) => {
    const u = new URL(url, 'http://studio.test'); requests.push(u.pathname);
    let data = {}, ok = true;
    if (u.pathname === '/api/catalog') {
      if (catalogFailure) { ok = false; data = {detail:'catalog unavailable'}; }
      else data = {models:[model], active_gpu_profile:{id:'auto',short:'Auto'}, loaded_slots:[]};
    } else if (u.pathname === '/api/assets') {
      // Take the snapshot *before* waiting, to reproduce a stale in-flight response.
      const snapshot = {uploads:[],images:[],videos:completed ? [asset] : [],audio:[],montages:[]};
      if (hold) { hold = false; await new Promise(r => { release = r; }); }
      if (assetsFailure) { ok = false; data = {detail:'assets unavailable'}; }
      else data = snapshot;
    } else if (u.pathname === '/api/tts/voices') data = {voices:[]};
    else if (u.pathname === '/api/higgsfield/catalog') data = {cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    else if (u.pathname === '/api/status') data = {
      hardware:{gpu:{has_gpu:false},ram:{used_gb:1,total_gb:4,percent:25},disk:{free_gb:9,total_gb:10,percent:10},credits:{remaining_credits:0,estimated_hours_left:null,hourly_rate:0,session_spent:0},ephemeral_disk_notice:''},
      memory:{loaded_slots:[]}, jobs:running ? [{...job, status:completed?'completed':'running', result:completed?asset:null}] : [], active_jobs_count:running&&!completed?1:0, hf_auth:{},runtime:{}};
    else if (u.pathname === '/api/video/ltx-gguf') { running = true; data = {job}; }
    return {ok, status:ok?200:503, json:async()=>data};
  };
  w.eval(fs.readFileSync('studio/static/app.js', 'utf8'));
  await until(() => $('select-video-model').value === id, 'initialization');
  $('vsub-t2v').click(); $('vid-prompt').value = 'Un hombre habla';
  $('btn-run-video').click();
  await until(() => running, 'job submitted');
  await wait(100);
  hold = true; $('btn-refresh-clips').click();
  await until(() => !!release, 'stale assets request held');
  catalogFailure = true;
  completed = true;
  await until(() => !$('vid-result-content').hidden, 'result shown despite pending assets/catalog');
  if (!$('vid-result-player').src.includes(asset.media_url)) throw Error('Wrong preview');
  if ($('btn-vid-download').getAttribute('href') !== asset.media_url) throw Error('Missing direct download');
  $('vid-result-player').dispatchEvent(new w.Event('error'));
  if ($('vid-playback-error').hidden) throw Error('Playback failure not visible');
  release();
  await until(() => $('video-gallery-strip').textContent.includes('clip.mp4'), 'fresh gallery after stale request');
  if (!$('video-gallery-strip').querySelector('a[download]')) throw Error('No gallery download link');
  if (requests.filter(x => x === '/api/assets').length < 3) throw Error('No follow-up asset read');
  assetsFailure = true;
  $('btn-refresh-clips').click(); await wait(60);
  if (!$('toast-container').textContent.includes('No se pudo actualizar')) throw Error('Manual refresh failure not reported');
  assetsFailure = false;
  $('btn-refresh-clips').click(); await wait(100);
  if (!$('toast-container').textContent.includes('Biblioteca de clips actualizada')) throw Error('Manual retry failed');
  if (errors.length) throw Error(errors.join('; '));
  w.close();
  // A new browser tab did not watch the original job. It must still recover
  // the saved clip from /api/assets and offer it in the preview.
  catalogFailure = false; running = false; completed = true;
  const reloaded = new JSDOM(fs.readFileSync('studio/static/index.html', 'utf8'), {
    url:'http://studio.test/', runScripts:'outside-only', pretendToBeVisual:true, virtualConsole:vc,
  });
  const w2 = reloaded.window;
  w2.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){},drawImage(){},getImageData(){return {data:[]}},putImageData(){}});
  w2.fetch = w.fetch; w2.confirm = () => true;
  w2.eval(fs.readFileSync('studio/static/app.js', 'utf8'));
  await until(() => !w2.document.getElementById('vid-result-content').hidden, 'recovered on page reload');
  if (!w2.document.getElementById('vid-result-player').src.includes(asset.media_url)) throw Error('Wrong recovered clip');
  if (errors.length) throw Error(errors.join('; '));
  console.log('Video completion, stale refresh, playback warning, retry and reload: OK');
  w2.close();
})().catch(e => { console.error(e); process.exitCode = 1; });
