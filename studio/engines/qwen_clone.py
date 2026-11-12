"""Experimental isolated Qwen3-TTS voice cloning; no model data in metadata."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from studio.config import AUDIO_OUT_DIR, DATA_ROOT, TEMP_DIR
from studio.engines import ltx_gguf_backend as common
from studio.engines.ffmpeg_utils import probe_media_info
from studio.engines.voice_clone_reference import prepared_reference_audio, validate_clone_request
from studio.hardware import query_gpu_info
from studio.memory_manager import memory_manager, serialized_gpu_operation, checkpoint_current_operation

ENGINE = 'qwen3-tts-0.6b-base'
SOURCE_REV = '022e286b98fbec7e1e916cb940cdf532cd9f488e'
ROOT = DATA_ROOT / 'runtimes' / 'qwen3-tts-clone'
WEIGHTS = DATA_ROOT / 'models' / 'qwen3-tts-0.6b-base'
_verified = None


def _prepare(job: Any) -> Path:
    global _verified
    from studio.engines.qwen_clone_download import FILES
    ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    missing = sum(size for name, (size, _, _) in FILES.items() if not (WEIGHTS/name).is_file())
    if shutil.disk_usage(WEIGHTS).free < missing + 4 * 1024**3:
        raise RuntimeError('QWEN_DISK_LOW: falta espacio para pesos, entorno y margen de seguridad')
    env = common._setup_env()
    py = ROOT / 'venv' / 'bin' / 'python'
    source = ROOT / 'source'
    marker = ROOT / 'ready-v1.json'
    if not source.exists():
        common._command(['git','clone','--no-checkout','https://github.com/QwenLM/Qwen3-TTS.git',str(source)], job, stage='Preparando código Qwen3-TTS', timeout=600, env=env)
        common._command(['git','-C',str(source),'checkout','--detach',SOURCE_REV],job,stage='Fijando versión Qwen3-TTS',timeout=120,env=env)
    if source.is_symlink() or not (source/'.git').is_dir():
        raise RuntimeError('QWEN_SOURCE_CONFLICT: código existente no gestionado')
    head=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=20).stdout.strip()
    status=subprocess.run(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True,timeout=20).stdout.strip()
    if head != SOURCE_REV or status:
        raise RuntimeError('QWEN_SOURCE_CONFLICT: código modificado o revisión diferente; no se sobrescribe')
    if not marker.is_file() or not py.is_file():
        py = common._prepare_venv(job, ROOT)
        common._command([str(py),'-c','import torch, torchaudio; assert torch.cuda.is_available()'],job,
                        stage='Comprobando PyTorch y torchaudio CUDA',timeout=90,env=env)
        # These packages live only in this venv. Torch/torchaudio are inherited
        # from Colab and checked before pip, not replaced by an unpinned build.
        common._command([str(py),'-m','pip','install','transformers==4.57.3','accelerate==1.12.0',
                         'numpy==1.26.4','librosa==0.11.0','soundfile','sox','onnxruntime','einops'],job,
                        stage='Instalando dependencias Qwen aisladas',timeout=2400,env=env)
        check = "import sys; sys.path.insert(0,sys.argv[1]); import torch,torchaudio,transformers; assert transformers.__version__=='4.57.3'; from qwen_tts import Qwen3TTSModel"
        common._command([str(py),'-c',check,str(source)],job,stage='Comprobando Qwen aislado',timeout=120,env=env)
        marker.write_text(json.dumps({'source':SOURCE_REV,'transformers':'4.57.3'}),encoding='utf8')
    def fingerprint():
        try:
            return tuple((p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino) for p in (WEIGHTS / name for name in FILES))
        except OSError:
            return None
    fp=fingerprint()
    if fp is None or fp != _verified:
        from studio.catalog import catalog_manager
        token=catalog_manager.get_hf_token()
        if token: env['HF_TOKEN']=token
        repo=str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH']=repo
        common._command([sys.executable,'-m','studio.engines.qwen_clone_download',str(WEIGHTS)],job,
                        stage='Descargando y verificando Qwen3-TTS',timeout=7200,env=env,cwd=repo)
        _verified=fingerprint()
        if _verified is None: raise RuntimeError('QWEN_WEIGHTS_MISSING: faltan archivos de modelo')
    return py


class QwenSession:
    def __init__(self, py: Path, job: Any):
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='qwen-clone-',dir=TEMP_DIR)
        self.directory=Path(self.temp.name)
        self.log=tempfile.TemporaryFile()
        self.proc=None
        try:
            self.proc=subprocess.Popen([str(py),str(Path(__file__).with_name('qwen_clone_worker.py')),str(ROOT/'source'),str(WEIGHTS),str(self.directory)],
                stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True,start_new_session=True,env=common._runtime_env())
            self._wait(self.directory/'ready.json',job,900,'Cargando modelo Qwen3-TTS')
        except BaseException:
            self.unload(); raise

    def _wait(self, result: Path, job: Any, timeout: int, stage: str):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            job.check_cancelled()
            checkpoint_current_operation(job=job,force=True)
            if result.is_file():
                data=json.loads(result.read_text(encoding='utf8')); result.unlink()
                if not data.get('ok'): raise RuntimeError('QWEN_INFERENCE_FAILED: '+str(data.get('error','Error desconocido')))
                return
            if self.proc.poll() is not None:
                raise RuntimeError('QWEN_WORKER_EXITED: '+common._tail(self.log))
            job.heartbeat(step=stage)
            time.sleep(.5)
        raise TimeoutError('QWEN_TIMEOUT: '+stage)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def generate(self, job, text, transcript, reference, output):
        ident=uuid.uuid4().hex
        req={'id':ident,'text':text,'transcript':transcript,'reference':str(reference),'output':str(output)}
        self.proc.stdin.write(json.dumps(req,ensure_ascii=False)+'\n'); self.proc.stdin.flush()
        self._wait(self.directory/(ident+'.json'),job,900,'Generando voz Qwen3-TTS')

    def unload(self):
        if self.proc is not None:
            common._stop(self.proc); self.proc=None
        self.log.close()
        self.temp.cleanup()


@serialized_gpu_operation(slots=('tts',), managed=True)
def synthesize(*, text: str, reference_path: Path, reference_text: str | None, job: Any) -> dict:
    started=time.monotonic()
    transcript=validate_clone_request(ENGINE,text,reference_text)
    text=text.strip()
    if len(text)>500: raise ValueError('Para este experimento usa hasta 500 caracteres por frase.')
    gpu=query_gpu_info()
    if not gpu.get('has_gpu') or not gpu.get('bf16_supported'):
        raise RuntimeError('QWEN_GPU_UNSUPPORTED: se necesita GPU CUDA compatible con BF16')
    with prepared_reference_audio(reference_path) as reference:
        slot=memory_manager.get_slot('tts')
        session=slot.get('handle') if slot and slot.get('model_id')==ENGINE else None
        if session and not session.alive():
            memory_manager.unload_slot('tts'); session=None
        if session:
            memory_manager.ensure_capacity(ram_gb=1,vram_gb=1,stage='Qwen residente: margen de inferencia',job=job)
            job.log('Reutilizando modelo de voz en caché mientras haya memoria suficiente.')
        else:
            if slot: memory_manager.unload_slot('tts')
            memory_manager.ensure_capacity(ram_gb=8,vram_gb=5,stage='Qwen3-TTS: estimación preventiva no medida',job=job)
            py=_prepare(job)
            checkpoint_current_operation(job=job,force=True)
            memory_manager.ensure_capacity(ram_gb=8,vram_gb=5,stage='Qwen3-TTS: margen antes de cargar pesos',job=job)
            session=QwenSession(py,job)
            try:
                memory_manager.register_loaded('tts',ENGINE,'Qwen3-TTS 0.6B · clonación experimental',session)
            except BaseException:
                session.unload(); raise
        AUDIO_OUT_DIR.mkdir(parents=True,exist_ok=True)
        name=f'voz_qwen_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.wav'
        temporary=session.directory/(uuid.uuid4().hex+'.wav')
        try:
            session.generate(job,text,transcript,reference,temporary)
            info=probe_media_info(temporary)
            if not info.get('has_audio') or info.get('has_video') or not (info.get('duration_seconds') or 0)>0:
                raise RuntimeError('QWEN_OUTPUT_INVALID: audio no verificable')
            output=AUDIO_OUT_DIR/name
            if output.exists(): raise RuntimeError('QWEN_OUTPUT_COLLISION: no se sobrescribirá audio existente')
            shutil.copyfile(temporary,output)
        except BaseException:
            memory_manager.unload_slot('tts')
            raise
        finally:
            temporary.unlink(missing_ok=True)
    meta={'id':output.stem,'filename':name,'rel_path':f'outputs/audio/{name}','media_url':f'/media/outputs/audio/{name}',
          'type':'audio','model_id':ENGINE,'model_name':'Qwen3-TTS 0.6B · experimental','text':text,
          'duration_seconds':info['duration_seconds'],'elapsed_seconds':round(time.monotonic()-started,2),
          'created_at':datetime.now().isoformat(timespec='seconds')}
    (AUDIO_OUT_DIR/(output.stem+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    return meta
