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
    } else if (u.pathname === '/api/video/h3/prepare' || u.pathname === '/api/video/h3/generate') {
      job = {id:'h3job',kind:u.pathname.endsWith('prepare')?'download':'video',title:'H3',status:'running',progress:10,current_step:'Preparando',logs:[]}; data = {job};
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
  $('h3-prepare').click();
  if (calls.some(c => c.path === '/api/video/h3/prepare')) throw Error('Preparation did not require license confirmation');
  $('h3-license').checked = true;
  $('h3-prepare').click();
  await until(() => calls.some(c => c.path === '/api/video/h3/prepare'), 'H3 prepare');
  let body = JSON.parse(calls.find(c => c.path === '/api/video/h3/prepare').body);
  if (!body.license_authorized || body.model_id !== 'h3-fl2va-udq2') throw Error('Bad H3 prepare');
  const uploadH3 = async (input, target, name) => {
    Object.defineProperty($(input), 'files', {configurable:true, value:[new w.File(['x'], name, {type:'image/png'})]});
    $(input).dispatchEvent(new w.Event('change'));
    await until(() => $(target).value === 'uploads/'+name, 'upload ' + name);
  };
  await uploadH3('h3-upload-start', 'h3-start', 'start.png');
  await uploadH3('h3-upload-start', 'h3-start', 'replacement.png');
  if ($('h3-end').value) throw Error('Start upload assigned end');
  await uploadH3('h3-upload-end', 'h3-end', 'end.png');
  $('h3-prompt').value = 'una escena'; $('h3-generate').click();
  await until(() => calls.some(c => c.path === '/api/video/h3/generate'), 'H3 generate');
  body = JSON.parse(calls.find(c => c.path === '/api/video/h3/generate').body);
  if (body.image_rel !== 'uploads/replacement.png' || body.end_image_rel !== 'uploads/end.png' || body.frames !== 22 || !body.offload || !body.license_authorized) throw Error('Wrong H3 inputs');
  if (errors.length) throw Error(errors.join('; '));
  w.close(); console.log('H3 UI prepare, license and frames: OK');
})().catch(e => { console.error(e); process.exit(1); });
