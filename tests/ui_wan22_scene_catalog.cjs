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
    } else if (u.pathname === '/api/video/wan22-i2v') {
      job = {id:'wan',kind:'video',title:'Wan',status:'running',progress:1,current_step:'Preparando',logs:[]}; data = {job};
    } else if (u.pathname === '/api/video/ltx-gguf') {
      job = {id:'j1',kind:'video',title:'LTX 2.5',status:'running',progress:10,current_step:'Generando',logs:[]}; data = {job};
    } else if (u.pathname === '/api/status') data = {
      hardware:{gpu:{has_gpu:false},ram:{used_gb:1,total_gb:4,percent:25},disk:{free_gb:9,total_gb:10,percent:10},credits:{remaining_credits:0,estimated_hours_left:null,hourly_rate:0,session_spent:0},ephemeral_disk_notice:''},
      memory:{loaded_slots:[]}, jobs:job ? [{...job, status:completed?'completed':'running',result:completed?output:null}] : [], active_jobs_count:job&&!completed?1:0, hf_auth:{},runtime:{}};
    else if (u.pathname === '/api/3d/catalog') data = {models:[{id:'sf3d',name:'Stable Fast 3D',state:'sin integración',output:'GLB',variants:'F32',license:'licencia',hardware:'variable',preprocess:'rembg',url:'https://example.com'}]};
    else if (u.pathname === '/api/worlds/catalog') data = {models:[{id:'sharp',name:'Apple SHARP',input:'foto',output:'PLY',details:'investigación',url:'https://example.com'}]};
    else if (u.pathname === '/api/tts/voices') data = {voices:[]};
    else if (u.pathname === '/api/higgsfield/catalog') data = {cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    return {ok:true, status:200, json:async()=>data};
  };
  w.eval(fs.readFileSync('studio/static/app.js', 'utf8'));
  await until(() => $('select-video-model').value === id, 'model ready');
  w.document.querySelector('[data-tab="tab-3d"]').click();
  await until(() => $('model3d-catalog').textContent.includes('Stable Fast 3D'), '3D catalogue');
  if (!$('model3d-run') || $('model3d-run').closest('[hidden]')) throw Error('3D generation must be available');
  w.document.querySelector('[data-tab="tab-worlds"]').click();
  await until(() => $('world-catalog').textContent.includes('Apple SHARP'), 'world catalogue');
  if (!$('world-run') || $('world-run').closest('[hidden]')) throw Error('world generation must be available');
  Object.defineProperty($('wan22-upload'), 'files', {configurable:true,value:[new w.File(['x'],'start.png',{type:'image/png'})]});
  $('wan22-upload').dispatchEvent(new w.Event('change'));
  await until(() => $('wan22-image').value === 'uploads/start.png', 'wan upload');
  $('wan22-prompt').value = 'A person walking'; $('wan22-run').click();
  if (calls.some(c => c.path === '/api/video/wan22-i2v')) throw Error('download consent required');
  $('wan22-install').checked=true; $('wan22-quant').value='fp8'; $('wan22-run').click();
  await until(() => calls.some(c => c.path === '/api/video/wan22-i2v'), 'Wan submit');
  const body=JSON.parse(calls.find(c => c.path === '/api/video/wan22-i2v').body);
  if (body.quant !== 'fp8' || body.image_rel !== 'uploads/start.png' || body.steps !== 40 || body.frames !== 49 || !body.install_accepted) throw Error('bad Wan request');
  $('wan22-recipe').value='fun'; $('wan22-recipe').dispatchEvent(new w.Event('change'));
  if (!$('wan22-quant').querySelector('[value="nvfp4"]').disabled || $('wan22-quant').querySelector('[value="q4_0"]').disabled) throw Error('Fun quant availability');
  $('wan22-mode').value='t2v'; $('wan22-mode').dispatchEvent(new w.Event('change'));
  if (!$('wan22-image').disabled || !$('wan22-end').disabled) throw Error('Text mode image state');
  $('wan22-run').click();
  await until(() => calls.filter(c=>c.path==='/api/video/wan22-i2v').length===2, 'Wan text submit');
  let fun=JSON.parse(calls.filter(c=>c.path==='/api/video/wan22-i2v')[1].body);
  if(fun.recipe!=='fun' || fun.mode!=='t2v' || fun.image_rel!==null || fun.end_image_rel!==null) throw Error('bad text request');
  $('wan22-mode').value='flf2v'; $('wan22-mode').dispatchEvent(new w.Event('change'));
  if ($('wan22-image').disabled || $('wan22-end').disabled) throw Error('First/last image state');
  Object.defineProperty($('wan22-end-upload'), 'files', {configurable:true,value:[new w.File(['x'],'end.png',{type:'image/png'})]});
  $('wan22-end-upload').dispatchEvent(new w.Event('change'));
  await until(() => $('wan22-end').value === 'uploads/end.png', 'wan end upload');
  $('wan22-quant').value='q4_0'; $('wan22-run').click();
  await until(() => calls.filter(c=>c.path==='/api/video/wan22-i2v').length===3, 'Wan first last submit');
  fun=JSON.parse(calls.filter(c=>c.path==='/api/video/wan22-i2v')[2].body);
  if(fun.mode!=='flf2v' || fun.quant!=='q4_0' || fun.image_rel!=='uploads/start.png' || fun.end_image_rel!=='uploads/end.png') throw Error('bad first/last request');
  $('wan22-recipe').value='i2v'; $('wan22-recipe').dispatchEvent(new w.Event('change'));
  if($('wan22-mode').value!=='i2v' || $('wan22-quant').value!=='q4') throw Error('I2V incompatible choices reset');
  if (errors.length) throw Error(errors.join('; '));
  w.close(); console.log('Wan I2V UI and experimental 3D menus: OK');
})().catch(e => { console.error(e); process.exit(1); });
