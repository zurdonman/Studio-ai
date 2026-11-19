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
    else if (u.pathname === '/api/assets') data = {uploads:[...uploads],images:[{id:'ref',filename:'ref.png',rel_path:'outputs/images/ref.png',media_url:'/media/outputs/images/ref.png',type:'image'}],videos:[output],audio:[],montages:[]};
    else if (u.pathname === '/api/assets/upload') {
      const name = opts.body.get('file').name;
      const asset = {type:'image', filename:name, rel_path:'uploads/'+name, media_url:'/media/uploads/'+name};
      uploads.unshift(asset); data = {asset};
    } else if (u.pathname === '/api/community/motion') {
      job = {id:'j1',kind:'video',title:'Motion',status:'running',progress:10,current_step:'Preparando',logs:[]}; data = {job};
    } else if (u.pathname === '/api/community/film') {
      job = {id:'j1',kind:'video',title:'FILM',status:'running',progress:10,current_step:'Preparando',logs:[]}; data = {job};
    } else if (u.pathname === '/api/video/ltx-gguf') {
      job = {id:'j1',kind:'video',title:'LTX 2.5',status:'running',progress:10,current_step:'Generando',logs:[]}; data = {job};
    } else if (u.pathname === '/api/status') data = {
      hardware:{gpu:{has_gpu:false},ram:{used_gb:1,total_gb:4,percent:25},disk:{free_gb:9,total_gb:10,percent:10},credits:{remaining_credits:0,estimated_hours_left:null,hourly_rate:0,session_spent:0},ephemeral_disk_notice:''},
      memory:{loaded_slots:[]}, jobs:job ? [{...job, status:completed?'completed':'running',result:completed?output:null}] : [], active_jobs_count:job&&!completed?1:0, hf_auth:{},runtime:{}};
    else if (u.pathname === '/api/3d/catalog') data = {models:[{id:'sf3d',name:'Stable Fast 3D',state:'Experimental',output:'GLB',variants:'F32',license:'licencia',hardware:'6 GB',preprocess:'rembg',url:'https://example.com',configured:false,auto_setup:true,missing:['STUDIO_SF3D_REPO']},{id:'triposr',name:'TripoSR',state:'Experimental',output:'GLB',variants:'checkpoint',license:'MIT',hardware:'6 GB',preprocess:'rembg',url:'https://example.com',configured:true}]};
    else if (u.pathname === '/api/3d/prepare') { job={id:'prep',kind:'scene_setup',status:'running',title:'Preparar',logs:[]}; data={job}; }
    else if (u.pathname === '/api/3d/generate') { job = {id:'j3',kind:'model3d',title:'3D',status:'running',progress:1,current_step:'Generando',logs:[]}; data={job}; }
    else if (u.pathname === '/api/tts/voices') data = {voices:[]};
    else if (u.pathname === '/api/higgsfield/catalog') data = {cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    return {ok:true, status:200, json:async()=>data};
  };
  w.eval(fs.readFileSync('studio/static/app.js', 'utf8'));
  await until(() => $('select-video-model').value === id, 'model ready');
  w.HTMLElement.prototype.scrollIntoView = () => {};
  w.document.querySelector('[data-tab="tab-3d"]').click();
  await until(() => $('model3d-catalog').textContent.includes('Stable Fast 3D') && $('model3d-image').options.length > 1, '3D ready');
  if (!$('tab-3d').classList.contains('active') || !$('model3d-runtime').textContent.includes('automáticamente')) throw Error('missing runtime status');
  if ($('model3d-prepare').disabled) throw Error('preparation should be enabled');
  if (!Array.from($('model3d-variant').options).some(o=>o.value==='onnx')) throw Error('SF3D ONNX missing');
  $('model3d-variant').value='onnx'; $('model3d-variant').dispatchEvent(new w.Event('change'));
  $('model3d-license').checked=true; $('model3d-prepare').click();
  await until(() => calls.some(x => x.path === '/api/3d/prepare'), 'prepare submit');
  const prep=JSON.parse(calls.find(x=>x.path==='/api/3d/prepare').body);
  if(prep.engine!=='sf3d' || prep.variant!=='onnx' || !prep.license_accepted) throw Error('bad prepare payload');
  $('model3d-engine').value = 'triposr'; $('model3d-engine').dispatchEvent(new w.Event('change'));
  if (!$('model3d-resolution').disabled || !$('model3d-runtime').textContent.includes('configuradas')) throw Error('TripoSR state');
  $('model3d-image').value='outputs/images/ref.png'; $('model3d-license').checked=true;
  $('model3d-run').click();
  await until(() => calls.some(x => x.path === '/api/3d/generate'), '3D submit');
  const body = JSON.parse(calls.find(x => x.path === '/api/3d/generate').body);
  if (body.engine !== 'triposr' || body.image_rel !== 'outputs/images/ref.png' || !body.license_accepted || body.texture_resolution !== 1024) throw Error('bad 3D payload');
  await until(() => $('model3d-feedback').textContent.includes('cola'), 'job feedback');
  if(errors.length) throw Error(errors.join('; ')); w.close(); console.log('3D UI catalog, runtime and submission: OK');
})().catch(e => { console.error(e); process.exit(1); });
