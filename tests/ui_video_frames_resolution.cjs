/* Real UI event handlers, fake assets and jobs; no GPU or network needed. */
const fs = require('fs');
const {JSDOM, VirtualConsole} = require('jsdom');
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(test, label) {
  for (let i = 0; i < 100; i++) { if (test()) return; await wait(20); }
  throw Error('Timeout: ' + label);
}
(async () => {
  const errors = [], calls = [], uploads = [];
  const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(e.message));
  const dom = new JSDOM(fs.readFileSync('studio/static/index.html', 'utf8'), {
    url:'http://studio.test/', runScripts:'outside-only', pretendToBeVisual:true, virtualConsole:vc,
  });
  const w = dom.window, $ = id => w.document.getElementById(id);
  w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){},drawImage(){},getImageData(){return {data:[]}},putImageData(){}});
  w.confirm = () => true;
  const id = 'ltx-2.5-distilled-comfy-nvfp4-gemma-q5-experimental';
  const models = [{id, name:'NVFP4', category:'video', engine:'ltx_gguf', video_modes:['i2v','t2v'],
    estimated_gb:24, ltx_distilled:true, compatibility:{compatible:false},
    official_spec:{landscape:[512,320],portrait:[320,512],default_steps:8,default_cfg:1,default_fps:24,default_duration_sec:2}},
    {id:'wan-flf', name:'Frame a Frame', category:'video', engine:'wan_i2v_14b', video_modes:['i2v','flf2v'],
     estimated_gb:8, compatibility:{compatible:true}, official_spec:{landscape:[640,384],portrait:[384,640],default_steps:25,default_cfg:3}}];
  let job = null, completed = false;
  const output = {id:'clip', filename:'clip.mp4', rel_path:'outputs/videos/clip.mp4', media_url:'/media/outputs/videos/clip.mp4', type:'video', width:768, height:512, fps:24, steps:8};
  w.fetch = async (url, opts={}) => {
    const u = new URL(url, 'http://studio.test'); calls.push({path:u.pathname, body:opts.body}); let data = {};
    if (u.pathname === '/api/catalog') data = {models, active_gpu_profile:{id:'auto',short:'Auto'}, loaded_slots:[]};
    else if (u.pathname === '/api/assets') data = {uploads:[...uploads],images:[],videos:completed ? [output]:[],audio:[],montages:[]};
    else if (u.pathname === '/api/assets/upload') {
      const name = opts.body.get('file').name;
      const asset = {type:'image', filename:name, rel_path:'uploads/'+name, media_url:'/media/uploads/'+name};
      uploads.unshift(asset); data = {asset};
    } else if (u.pathname === '/api/video/ltx-gguf') {
      job = {id:'j1',kind:'video',title:'LTX 2.5',status:'running',progress:10,current_step:'Generando',logs:[]}; data = {job};
    } else if (u.pathname === '/api/status') data = {
      hardware:{gpu:{has_gpu:false},ram:{used_gb:1,total_gb:4,percent:25},disk:{free_gb:9,total_gb:10,percent:10},credits:{remaining_credits:0,estimated_hours_left:null,hourly_rate:0,session_spent:0},ephemeral_disk_notice:''},
      memory:{loaded_slots:[]}, jobs:job ? [{...job, status:completed?'completed':'running',result:completed?output:null}] : [], active_jobs_count:job&&!completed?1:0, hf_auth:{},runtime:{}};
    else if (u.pathname === '/api/tts/voices') data = {voices:[]};
    else if (u.pathname === '/api/higgsfield/catalog') data = {cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    return {ok:true, status:200, json:async()=>data};
  };
  w.eval(fs.readFileSync('studio/static/app.js', 'utf8'));
  await until(() => $('select-video-model').value === id, 'model ready');
  if ($('input-vid-upload-img').multiple) throw Error('Start uploader must not silently assign a second image');
  const upload = async (input, name) => {
    const elem = $(input);
    Object.defineProperty(elem, 'files', {configurable:true, value:[new w.File(['x'], name, {type:'image/png'})]});
    elem.dispatchEvent(new w.Event('change'));
    await until(() => uploads.some(x => x.filename === name), 'upload ' + name);
    await until(() => $(input === 'input-vid-upload-img' ? 'vid-source-image' : 'vid-end-image').value === 'uploads/'+name, 'assignment ' + name);
  };
  $('vid-width').value = '768'; $('vid-height').value = '512';
  await upload('input-vid-upload-img','one.png');
  const oldThumb = $('vid-src-thumb-img');
  Object.defineProperty(oldThumb, 'naturalWidth', {value:1920}); Object.defineProperty(oldThumb, 'naturalHeight', {value:1080});
  oldThumb.dispatchEvent(new w.Event('load'));
  if ($('vid-width').value !== '768' || $('vid-height').value !== '512') throw Error('Image load replaced manually set resolution');
  await upload('input-vid-upload-img','two.png');
  oldThumb.dispatchEvent(new w.Event('load'));
  if ($('vid-end-image').value || $('select-video-model').value !== id || !$('vsub-i2v').classList.contains('active')) throw Error('Second start upload changed to end or switched model/mode');
  if ($('vid-width').value !== '768' || $('vid-height').value !== '512') throw Error('Second upload replaced resolution');
  $('vid-prompt').value = 'Una persona camina';
  $('btn-run-video').click(); await until(() => !!job, 'job submitted');
  const body = JSON.parse(calls.filter(x => x.path === '/api/video/ltx-gguf').at(-1).body);
  if (body.image_rel !== 'uploads/two.png' || body.width !== 768 || body.height !== 512) throw Error('Wrong generation input');
  completed = true;
  await until(() => !$('vid-result-content').hidden, 'job completion');
  await until(() => $('video-gallery-strip').textContent.includes('clip.mp4'), 'gallery refresh');
  $('vid-src-thumb-img').dispatchEvent(new w.Event('load'));
  if ($('vid-width').value !== '768' || $('vid-height').value !== '512' || $('vid-end-image').value) throw Error('Completion refresh changed inputs');
  $('vsub-t2v').click(); $('vsub-i2v').click();
  if ($('vid-width').value !== '768' || $('vid-height').value !== '512') throw Error('Mode change reset resolution for same model');
  $('vid-end-image').value = 'uploads/one.png'; $('vid-end-image').dispatchEvent(new w.Event('change'));
  $('btn-refresh-clips').click(); await wait(150);
  if ($('vid-end-image').value !== 'uploads/one.png') throw Error('Explicit end frame lost during refresh');
  await upload('input-vid-upload-img','three.png');
  if ($('vid-end-image').value !== 'uploads/one.png' || $('select-video-model').value !== id) throw Error('Replacing start overwrote explicit end or changed model');
  $('vid-end-image').value = ''; $('vid-end-image').dispatchEvent(new w.Event('change'));
  $('btn-refresh-clips').click(); await wait(150);
  if ($('vid-end-image').value) throw Error('End frame was automatically selected after being cleared');
  await upload('input-vid-quick-end-img','final.png');
  if ($('vid-source-image').value !== 'uploads/three.png' || $('vid-end-image').value !== 'uploads/final.png' || !$('vsub-flf2v').classList.contains('active')) throw Error('Explicit final upload failed');
  if (errors.length) throw Error(errors.join('; '));
  w.close(); console.log('Video resolution and explicit frame selection: OK');
})().catch(e => { console.error(e); process.exit(1); });
