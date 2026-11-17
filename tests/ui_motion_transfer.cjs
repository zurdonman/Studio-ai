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
    else if (u.pathname === '/api/tts/voices') data = {voices:[]};
    else if (u.pathname === '/api/higgsfield/catalog') data = {cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    return {ok:true, status:200, json:async()=>data};
  };
  w.eval(fs.readFileSync('studio/static/app.js', 'utf8'));
  await until(() => $('select-video-model').value === id, 'model ready');
  w.HTMLElement.prototype.scrollIntoView = () => {};
  w.document.querySelector('[data-tab="tab-community"]').click();
  if (!$('tab-community').classList.contains('active')) throw Error('Community tab not opened');
  await until(() => $('motion-image').options.length > 1 && $('motion-video').options.length > 1, 'motion assets');
  $('motion-image').value='outputs/images/ref.png'; $('motion-video').value=output.rel_path; $('motion-prompt').value='Dos personas bailan';
  if ($('motion-quant').value !== 'q4' || $('motion-steps').value !== '10') throw Error('Incorrect Wan defaults');
  const aspect=value => {const radio=w.document.querySelector(`input[name="motion-aspect"][value="${value}"]`); radio.checked=true; radio.dispatchEvent(new w.Event('change'));};
  if ($('motion-width').value!=='512' || $('motion-height').value!=='320' || !$('motion-width').disabled) throw Error('Default aspect incorrect');
  aspect('portrait');
  if ($('motion-width').value!=='288' || $('motion-height').value!=='512') throw Error('Portrait aspect incorrect');
  aspect('landscape');
  if ($('motion-width').value!=='512' || $('motion-height').value!=='288') throw Error('Landscape aspect incorrect');
  aspect('manual');
  if ($('motion-width').disabled || $('motion-height').disabled) throw Error('Manual dimensions disabled');
  $('motion-width').value='384'; $('motion-height').value='640';
  aspect('default');
  $('motion-engine').value='scail'; $('motion-engine').dispatchEvent(new w.Event('change'));
  if ($('motion-sam-options').hidden || $('motion-steps').value !== '30' || $('motion-cfg').value !== '5') throw Error('Incorrect SCAIL options');
  $('motion-run').click(); await wait(100);
  if (calls.some(x=>x.path==='/api/community/motion')) throw Error('SCAIL accepted without license');
  $('motion-sam-license').checked=true; $('motion-quant').value='q3'; $('motion-run').click();
  await until(()=>calls.some(x=>x.path==='/api/community/motion'),'SCAIL submit');
  let body=JSON.parse(calls.filter(x=>x.path==='/api/community/motion').at(-1).body);
  if(body.engine!=='scail' || body.quant!=='q3' || !body.sam_license_accepted || body.image_rel!=='outputs/images/ref.png' || body.video_rel!==output.rel_path) throw Error('Incorrect SCAIL payload');
  $('motion-engine').value='ltx'; $('motion-engine').dispatchEvent(new w.Event('change'));
  if (!$('motion-sam-options').hidden || !$('motion-steps').disabled || $('motion-quant').value !== 'q4') throw Error('Incorrect LTX controls');
  aspect('portrait');
  if ($('motion-width').value!=='320' || $('motion-height').value!=='576') throw Error('LTX alignment incorrect');
  $('motion-run').click(); await until(()=>calls.filter(x=>x.path==='/api/community/motion').length===2,'LTX submit');
  body=JSON.parse(calls.filter(x=>x.path==='/api/community/motion').at(-1).body);
  if(body.engine!=='ltx' || body.quant!=='q4' || body.steps!==8 || body.width!==320 || body.height!==576) throw Error('Incorrect LTX payload');
  $('motion-engine').value='wan'; $('motion-engine').dispatchEvent(new w.Event('change'));
  if ($('motion-duration-wrap').hidden || !$('motion-frames-wrap').hidden) throw Error('Wan duration field missing');
  $('motion-duration').value='30.25'; $('motion-quant').value='int8'; $('motion-run').click(); await until(()=>calls.filter(x=>x.path==='/api/community/motion').length===3,'Wan submit');
  body=JSON.parse(calls.filter(x=>x.path==='/api/community/motion').at(-1).body);
  if(body.engine!=='wan' || body.quant!=='int8' || body.duration_sec!==30.25 || body.width!==288 || body.height!==512) throw Error('Incorrect Wan payload');
  aspect('manual'); $('motion-width').value='384'; $('motion-height').value='640'; $('motion-run').click();
  await until(()=>calls.filter(x=>x.path==='/api/community/motion').length===4,'Wan manual submit');
  body=JSON.parse(calls.filter(x=>x.path==='/api/community/motion').at(-1).body);
  if(body.width!==384 || body.height!==640) throw Error('Manual dimensions missing');
  if(errors.length) throw Error(errors.join('; ')); w.close(); console.log('Motion UI selection, licensing, payloads: OK');
})().catch(e => { console.error(e); process.exit(1); });
