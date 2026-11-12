/* Clone submission, completion and handoff to LTX with simulated API only. */
const fs = require('fs');
const {JSDOM, VirtualConsole} = require('jsdom');
const wait = ms => new Promise(r => setTimeout(r, ms));
async function until(test, why) {
  for (let i=0;i<120;i++) { if (test()) return; await wait(25); }
  throw Error('Timeout: '+why);
}
(async () => {
  const errors=[]; const vc=new VirtualConsole(); vc.on('jsdomError',e=>errors.push(e.message));
  const dom=new JSDOM(fs.readFileSync('studio/static/index.html','utf8'),{url:'http://studio.test/',runScripts:'outside-only',pretendToBeVisual:true,virtualConsole:vc});
  const w=dom.window, $=id=>w.document.getElementById(id);
  w.confirm=()=>true; w.HTMLCanvasElement.prototype.getContext=()=>({clearRect(){},drawImage(){},getImageData(){return {data:[]}},putImageData(){}});
  const model={id:'ltx-2.5-distilled-gguf-ud-q2-gemma-q5-experimental',name:'LTX',category:'video',engine:'ltx_gguf',video_modes:['t2v','i2v'],ltx_distilled:true,estimated_gb:24,compatibility:{compatible:false},official_spec:{landscape:[512,320],portrait:[320,512],default_steps:8,default_cfg:1,default_fps:24,default_duration_sec:2}};
  const ref={filename:'muestra.wav',rel_path:'uploads/muestra.wav',media_url:'/media/uploads/muestra.wav',type:'audio'};
  const output={filename:'voz.wav',rel_path:'outputs/audio/voz.wav',media_url:'/media/outputs/audio/voz.wav',type:'audio',model_id:'qwen3-tts-0.6b-base',duration_seconds:3,text:'Hola'};
  let started=false, completed=false, submitted=null;
  w.fetch=async (url,opts={})=>{
    const path=new URL(url,'http://studio.test/').pathname; let data={};
    if(path==='/api/catalog') data={models:[model],active_gpu_profile:{id:'auto',short:'Auto'},loaded_slots:[]};
    else if(path==='/api/assets') data={uploads:[ref],images:[],videos:[],audio:completed?[output]:[],montages:[]};
    else if(path==='/api/tts/voices') data={voices:[]};
    else if(path==='/api/higgsfield/catalog') data={cameras:{},lenses:{},focal_lengths:{},apertures:{},lighting_moods:{},motions:{},characters:[]};
    else if(path==='/api/tts/clone') { submitted=JSON.parse(opts.body); started=true; data={job:{id:'clone'}}; }
    else if(path==='/api/status') data={hardware:{gpu:{has_gpu:false},ram:{used_gb:1,total_gb:4,percent:25},disk:{free_gb:9,total_gb:10,percent:10},credits:{remaining_credits:0,estimated_hours_left:null,hourly_rate:0,session_spent:0},ephemeral_disk_notice:''},memory:{loaded_slots:[]},jobs:started?[{id:'clone',kind:'tts',title:'Voz de referencia Qwen3-TTS',status:completed?'completed':'running',result:completed?output:null,progress:50,elapsed_seconds:1,logs:[]}]:[],active_jobs_count:started&&!completed?1:0,hf_auth:{},runtime:{}};
    return {ok:true,json:async()=>data};
  };
  w.eval(fs.readFileSync('studio/static/app.js','utf8'));
  await until(()=>$('clone-reference').options.length>1,'reference in library');
  $('clone-reference').value=ref.rel_path; $('clone-text').value='Hola'; $('clone-authorized').checked=true;
  $('btn-clone-voice').click(); await until(()=>started,'clone submitted');
  if(!submitted.authorized || submitted.reference_rel!==ref.rel_path) throw Error('wrong request');
  await wait(100); completed=true;
  await until(()=>$('clone-result').hidden===false,'clone result');
  if($('clone-result-download').getAttribute('href')!==output.media_url) throw Error('download missing');
  $('clone-result-video').click();
  await until(()=>$('vid-source-audio').value===output.rel_path,'audio selected for video');
  if($('vid-gguf-voice-mode').value!=='audio') throw Error('GGUF audio mode not selected');
  if(errors.length) throw Error(errors.join('; '));
  w.close(); console.log('Voice clone UI, result, download and video handoff: OK');
})().catch(e=>{console.error(e);process.exitCode=1});
