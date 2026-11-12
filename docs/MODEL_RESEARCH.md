# Investigación de modelos ejecutables y descubribles — Studio IA

**Fecha de revisión: 2026-11-05 (UTC).**

## Actualización GGUF

LTX 2.3 destilado GGUF Q4 ya tiene una ejecución I2V observada en Colab L4, incluidos dos clips de 20 s con caché; no se generaliza esa prueba a otros GGUF. Ver [guía de implementación, límites y candidatos](GGUF_PLAYBOOK.md). El resto de este informe describe la investigación y recetas de la fecha indicada, no una lista actualizada de inferencias GPU.

## Resultado y alcance real

Se ha reunido una base reproducible de **73 familias/proyectos documentados** y **243 referencias únicas a repositorios** de modelos/compañeros, más el catálogo de **74 recetas y complementos** de Studio (8 recetas nuevas de imagen). Las referencias proceden de documentación oficial de Diffusers fijada en **v0.42.1**, publicada el **2026-10-27**, y de README oficiales fijados a commits. El inventario machine-readable es [`studio/data/upstream_models.json`](../studio/data/upstream_models.json).

**No es una lista de 243 modelos garantizados ni de todos los modelos existentes.** Algunas referencias son encoders, VAE, tokenizers, adapters, convertidos para otros runtimes o checkpoints anteriores; no son necesariamente pipelines autónomos. Los tags/IDs pueden cambiar. El buscador consulta los metadatos actuales de Hugging Face y registra fecha/revisión; en esta máquina la salida directa a Hugging Face está restringida, de modo que el acceso real al Hub y las inferencias CUDA quedan pendientes de validar en Colab. Las fuentes GitHub sí se han consultado. No se han descargado pesos para investigar.

Un descubrimiento NO se registra automáticamente en el catálogo ejecutable. Se conservan tres estados: **receta revisada implementada**, **candidato por validar** y **backend/formato no soportado**. Los números de memoria son presupuestos de ingeniería para recetas concretas, NO benchmarks ni promesas.

## Hallazgos que cambian la selección del agente

1. **La memoria no se calcula solo por parámetros.** Un denoiser de 0.6B puede acompañarse de un T5/Gemma mucho mayor; un vídeo mantiene activaciones temporales y un VAE 3D. Cuantizar solo el denoiser no reduce los demás componentes.
2. **Receta, clase y formato van juntos.** HunyuanVideo 1.5 usa un contrato/guider diferente; FramePack combina varios repositorios; FLUX.2 no es FLUX.1; Qwen-Image 2.1 no es el paquete Qwen Edit 2511. Un `DiffusionPipeline.from_pretrained` genérico no traduce esas convenciones a la API de Studio.
3. **No extrapolar optimizaciones de otros runtimes.** GGUF/NVFP4/ComfyUI/WanGP/MLX requieren su propio cargador, precisión y offload. La receta específica de LTX 2.3 GGUF se ha implementado; la disponibilidad de otro modelo en esas herramientas no acredita soporte aquí. Las recetas FP8 específicas ya implementadas siguen usando sus compañeros y hooks exactos.
4. **BF16 importa.** Las nuevas recetas BF16 nativas no se autorizan para T4/V100. En T4 hay recetas FP16 de SDXL/PixArt; cambiar automáticamente el dtype de cualquier arquitectura puede provocar NaN, errores de kernel o picos distintos. L4/A100/H100 tampoco eliminan RAM/disco/gates.
5. **Poca VRAM puede exigir mucha RAM y tiempo.** Las cifras mínimas de los autores suelen presuponer offload, encoder cuantizado, resolución baja o group offload específicos. No se han trasladado como mínimos universales a Studio. El ejemplo oficial de PixArt con menos de 8 GiB modifica y descarga el T5; nuestra receta FP16 completa no promete ese mínimo.
6. **LoRA no es una base.** Compatibilidad por arquitectura, versión y submodo explícitos, incluyendo presupuesto de base + adapter. Ni todos los LoRAs son Qwen, ni todos los Wan 14B sirven para VACE/FLF2V.
7. **Licencias independientes.** Se muestran cuando constan, sin inferir libertad comercial por ser público. Por ejemplo, el repositorio oficial diferencia FLUX.2 Klein 4B Apache-2.0 de otras variantes con licencia no comercial; no se aplica una licencia única a toda la familia. Consultar la tarjeta/revisión vigente antes de usar pesos o publicar resultados.

Fuentes de estos hallazgos: [memoria/ejemplos Wan](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/wan.md), [PixArt Sigma](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/pixart_sigma.md), [FramePack](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/framepack.md), [HunyuanVideo 1.5](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/hunyuan_video15.md), [SANA y precisión de compañeros](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/sana.md). Para FLUX.2, ver la fuente oficial fijada en el inventario de abajo. Las conclusiones de soporte Studio se han contrastado con sus cargadores, no con publicidad ni benchmarks de terceros.

## Recetas nuevas de imagen

Todas usan una clase explícita, lectura local de pesos safetensors y CPU offload. No hay importación dinámica de Python remoto, conversión 4-bit implícita ni instalación automática. La creación desde texto no introduce una imagen de referencia gris en pipelines de imagen opcional. Los T2I puros rechazan referencias.

| ID Studio | Pipeline | Precisión/estrategia | VRAM mínima estimada GiB | RAM de planificación GiB | Disco de pesos estimado GiB | Pasos / CFG |
|---|---|---|---:|---:|---:|---|
| `sdxl-base-1.0` | `StableDiffusionXLPipeline` | float16 + CPU offload | 7.0 | 16 | 9.0 | 30 / 5.0 |
| `flux1-schnell` | `FluxPipeline` | bfloat16 + CPU offload | 24.0 | 48 | 36.0 | 4 / 0.0 |
| `flux2-klein-4b` | `Flux2KleinPipeline` | bfloat16 + CPU offload | 14.0 | 32 | 19.0 | 4 / 1.0 |
| `z-image-turbo` | `ZImagePipeline` | bfloat16 + CPU offload | 16.0 | 32 | 23.0 | 9 / 0.0 |
| `pixart-sigma-xl-1024` | `PixArtSigmaPipeline` | float16 + CPU offload | 12.0 | 32 | 24.0 | 20 / 4.5 |
| `sana-600m-1024` | `SanaPipeline` | bfloat16 + CPU offload | 10.0 | 24 | 20.0 | 20 / 4.5 |
| `sana-1600m-1024` | `SanaPipeline` | bfloat16 + CPU offload | 12.0 | 24 | 24.0 | 20 / 4.5 |
| `sana-sprint-600m` | `SanaSprintPipeline` | bfloat16 + CPU offload | 10.0 | 24 | 20.0 | 2 / 0.0 |

Estos límites incluyen presupuestos de compañeros, pero son aproximaciones y no picos medidos. La descarga se planifica de nuevo con tamaños de archivo reales; las estimaciones no autorizan una descarga si falta espacio. SANA conserva encoder BF16 y decodifica VAE en FP32; el transformer 600M usa FP16 y las otras dos recetas BF16. Se usan archivos canónicos (posiblemente FP32), no se presupone que el dtype de ejecución reduzca el disco. PixArt incluye presupuesto del T5 completo, no el ejemplo cuantizado de menos de 8 GiB. SDXL Base usa la variante FP16 sin bajar copias F32 duplicadas. El mínimo anunciado por un proyecto con quant/group-offload distinto NO se presenta como mínimo de este cargador. No se añaden LoRAs de una arquitectura a otra.

**Lumina 2 permanece como candidato**: el ejemplo oficial inyecta un transformer desde `.pth` o GGUF. No se habilita la receta genérica safetensors/carpeta porque no implementa ese contrato; no se ejecuta pickle ni código del repositorio para completar un cargador desconocido.

## Vídeo: lo implementado frente a lo investigado

- **Recetas ya implementadas**: Wan 2.1 T2V/VACE/I2V/FLF2V, combinaciones FP8 específicas, Wan 2.2 TI2V/A14B según modos declarados; LTX-Video 2B/13B destilado, LTX-2 AV; SANA-Video, SkyReels V2 I2V y CogVideoX I2V. Consultar ID exacto, compañeros, frames, resolución y `official_spec`. Que un modelo aparezca aquí no garantiza que una librería antigua contenga su clase.
- **Candidatos de integración**: HunyuanVideo/1.5, FramePack, Mochi, Cosmos, Helios, nuevas generaciones LTX/Kandinsky/Wan Animate/Motif/MiniMax y demás referencias del inventario. Diffusers puede soportarlos, pero Studio no promete automáticamente sus entradas de audio/vídeo, guider, caches, multi-etapa, múltiples denoisers ni callbacks. El manifiesto y los componentes se inspeccionan antes de proponer una receta.
- **No cambiar submodo por conveniencia**: TI2V de una sola imagen no equivale a fijar inicial y final; un encoder de audio de LTX AV no aparece por usar un `.safetensors` parecido.

## Voz, música, visión y LipSync

- **Voz implementada**: Edge (servicio de red), MMS español, Kokoro si se instala su backend y Bark pequeño. El ID de Studio es `mms-tts-spa`, no `mms-tts-es`; Bark no se anuncia como clonador arbitrario por defecto.
- **Investigados, no integrados como Kokoro/Transformers genéricos**: Qwen3-TTS (Base/CustomVoice/VoiceDesign y tokenizer exactos), Chatterbox y variantes, CosyVoice. Los README oficiales fijados se incluyen más abajo; se necesita adaptar sus librerías, selección de voz, streaming/clonación, dependencias, licencias y budgets antes de habilitarlos. No se han extrapolado tiempos ni VRAM de ejemplos de terceros.
- **Audio implementado**: MusicGen pequeño instrumental, AudioLDM2 para SFX/paisajes y DSP CPU. ACE-Step, Stable Audio, LongCat y otras referencias requieren contratos propios; una canción vocal no es ruido/ambiente DSP.
- **Visión implementada**: OpenCV fotométrico y Qwen2.5-VL-3B para auditoría semántica. Qwen3-VL, encoders de imagen y modelos de profundidad presentes en las fuentes no se convierten automáticamente en inspectores compatibles.
- **LipSync implementado**: paquete EDTalk ONNX + YuNet/XSeg/GFPGAN. El inventario de un modelo nuevo de avatar no basta para reemplazar ese contrato.

## Cómo se mantiene viva la investigación

El JSON de fuentes es la investigación reproducible de esta revisión. El **índice vivo es independiente**, guarda metadatos públicos con revisión/fecha, tiene búsqueda paginada, refresco manual o periódico, caché TTL y enfriamiento por errores/429. La búsqueda no instala modelos y la incorporación de un pipeline nuevo requiere una actualización de código revisada. Ver [manual detallado MCP/oráculo](MCP_MODEL_ORACLE.md) para esquemas y procedimientos exactos.

Se descartan las heurísticas anteriores de «14B en el nombre = cabe», las estadísticas de popularidad ficticias en modo offline y la importación automática de cualquier TTS como Kokoro. Si no hay cifras fiables, se devuelve `unknown`. No se promete cobertura exhaustiva: nuevos modelos aparecen cada día y el rastreo tiene límites deliberados.

## Inventario de fuentes oficiales fijadas

Las columnas indican **referencias de los documentos**, no aprobación para ejecutar cada repo. Un repositorio puede contener componentes o formatos de otro runtime; incluso una referencia oficial puede quedar obsoleta o tener un ID mal escrito. Comprobar metadatos actuales y revisión antes de descargar. Las familias sin IDs literales siguen siendo fuentes útiles, pero no se ha inventado un ID para rellenarlas.

| Contexto | Familia/proyecto y fuente | Repositorios referenciados |
|---|---|---|
| image | [flux](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/flux.md) | `Kijai/flux-fp8`<br>`LiheYoung/depth-anything-large-hf`<br>`black-forest-labs/FLUX.1-Canny-dev`<br>`black-forest-labs/FLUX.1-Canny-dev-lora`<br>`black-forest-labs/FLUX.1-Depth-dev`<br>`black-forest-labs/FLUX.1-Depth-dev-lora`<br>`black-forest-labs/FLUX.1-Fill-dev`<br>`black-forest-labs/FLUX.1-Kontext-dev`<br>`black-forest-labs/FLUX.1-Redux-dev`<br>`black-forest-labs/FLUX.1-dev`<br>`black-forest-labs/FLUX.1-schnell` |
| image | [flux2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/flux2.md) | Sin ID autor/repo literal confirmado en ese documento |
| image | [flux3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/flux3.md) | `black-forest-labs/flux-3-image-klein`<br>`black-forest-labs/flux-3-image-klein-base`<br>`black-forest-labs/flux-3-image-klein-pup` |
| image | [z_image](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/z_image.md) | `Tongyi-MAI/Z-Image-Turbo` |
| image | [qwenimage](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/qwenimage.md) | `Qwen/Qwen-Image`<br>`Qwen/Qwen-Image-Edit`<br>`Qwen/Qwen-Image-Edit-2509` |
| image | [qwenimage21](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/qwenimage21.md) | `Comfy-Org/Qwen-Image-2.1`<br>`Qwen/Qwen-Image-2.1`<br>`Qwen/Qwen-Image-2.1-Turbo` |
| image | [sana](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/sana.md) | `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`<br>`Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers`<br>`Efficient-Large-Model/Sana_1600M_1024px_diffusers`<br>`Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers`<br>`Efficient-Large-Model/Sana_1600M_512px_diffusers`<br>`Efficient-Large-Model/Sana_600M_1024px_diffusers`<br>`Efficient-Large-Model/Sana_600M_512px_diffusers` |
| image | [sana_sprint](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/sana_sprint.md) | `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`<br>`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` |
| image | [pixart](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/pixart.md) | `PixArt-alpha/PixArt-XL-2-1024-MS` |
| image | [pixart_sigma](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/pixart_sigma.md) | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` |
| image | [lumina2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/lumina2.md) | `Alpha-VLLM/Lumina-Image-2.0`<br>`calcuis/lumina-gguf` |
| image | [hunyuandit](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/hunyuandit.md) | `Tencent-Hunyuan/HunyuanDiT`<br>`Tencent-Hunyuan/HunyuanDiT-Diffusers` |
| image | [hunyuanimage21](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/hunyuanimage21.md) | `hunyuanvideo-community/HunyuanImage-2.1-Diffusers`<br>`hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers`<br>`hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers` |
| image | [omnigen](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/omnigen.md) | `Shitao/OmniGen-v1`<br>`Shitao/OmniGen-v1-diffusers` |
| image | [ovis_image](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/ovis_image.md) | `AIDC-AI/Ovis-Image-7B` |
| image | [hidream](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/hidream.md) | `HiDream-ai/HiDream-I1-Dev`<br>`HiDream-ai/HiDream-I1-Fast`<br>`HiDream-ai/HiDream-I1-Full` |
| image | [glm_image](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/glm_image.md) | `zai-org/GLM-4-9B-0414`<br>`zai-org/GLM-Image` |
| image | [longcat_image](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/longcat_image.md) | `meituan-longcat/LongCat-Image`<br>`meituan-longcat/LongCat-Image-Dev`<br>`meituan-longcat/LongCat-Image-Edit` |
| image | [chroma](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/chroma.md) | `lodestones/Chroma`<br>`lodestones/Chroma1-Base`<br>`lodestones/Chroma1-HD` |
| image | [cogview3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/cogview3.md) | Sin ID autor/repo literal confirmado en ese documento |
| image | [cogview4](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/cogview4.md) | Sin ID autor/repo literal confirmado en ese documento |
| image | [kolors](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/kolors.md) | `Kwai-Kolors/Kolors-IP-Adapter-Plus`<br>`Kwai-Kolors/Kolors-diffusers`<br>`openai/clip-vit-large-patch14-336` |
| image | [stable_cascade](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_cascade.md) | `stabilityai/stable-cascade`<br>`stabilityai/stable-cascade-prior` |
| image | [kandinsky](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/kandinsky.md) | `kandinsky-community/kandinsky-2-1`<br>`kandinsky-community/kandinsky-2-1-inpaint`<br>`kandinsky-community/kandinsky-2-1-prior`<br>`kandinsky-community/kandinsky-2-2-controlnet-depth`<br>`kandinsky-community/kandinsky-2-2-decoder`<br>`kandinsky-community/kandinsky-2-2-decoder-inpaint`<br>`kandinsky-community/kandinsky-2-2-prior`<br>`kandinsky-community/kandinsky-3` |
| image | [kandinsky3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/kandinsky3.md) | `google/flan-ul2` |
| image | [kandinsky_v22](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/kandinsky_v22.md) | Sin ID autor/repo literal confirmado en ese documento |
| image | [kandinsky5_image](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/kandinsky5_image.md) | `kandinsky-community/kandinsky-3`<br>`kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers`<br>`kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers`<br>`kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers`<br>`kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers` |
| video | [kandinsky6](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/kandinsky6.md) | `kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers`<br>`kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`<br>`kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers`<br>`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`<br>`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`<br>`kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers`<br>`kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers`<br>`kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` |
| image | [bria_3_2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/bria_3_2.md) | `briaai/BRIA-3.2` |
| image | [bria_fibo](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/bria_fibo.md) | `briaai/FIBO`<br>`briaai/FIBO-VLM-prompt-to-JSON`<br>`briaai/FIBO-gemini-prompt-to-JSON` |
| image | [bria_fibo2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/bria_fibo2.md) | `Qwen/Qwen3-VL-4B-Instruct`<br>`briaai/FIBO`<br>`briaai/Fibo-2.0-dev`<br>`briaai/Fibo-2.0-edit-turbo` |
| image | [ideogram4](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/ideogram4.md) | `diffusers/qwen3-vl-8b-instruct-lm-head`<br>`ideogram-ai/ideogram-4-nf4`<br>`ideogram-ai/ideogram-v4` |
| image | [text2img](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_diffusion/text2img.md) | Sin ID autor/repo literal confirmado en ese documento |
| image | [stable_diffusion_xl](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_xl.md) | `stabilityai/stable-diffusion-xl-base-1.0`<br>`stabilityai/stable-diffusion-xl-refiner-1.0` |
| image | [stable_diffusion_3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_3.md) | `madebyollin/taesd3`<br>`stabilityai/stable-diffusion-3-medium`<br>`stabilityai/stable-diffusion-3-medium-diffusers`<br>`stabilityai/stable-diffusion-3.5-large`<br>`stabilityai/stable-diffusion-3.5-large-turbo` |
| image | [sdxl_turbo](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_diffusion/sdxl_turbo.md) | `madebyollin/sdxl-vae-fp16-fix`<br>`stabilityai/sdxl-turbo` |
| video | [wan](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/wan.md) | `Comfy-Org/Wan_2.1_ComfyUI_repackaged`<br>`Kijai/WanVideo_comfy`<br>`Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`<br>`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`<br>`Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`<br>`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`<br>`Wan-AI/Wan2.1-T2V-14B-Diffusers`<br>`Wan-AI/Wan2.1-VACE-1.3B-diffusers`<br>`Wan-AI/Wan2.1-VACE-14B-diffusers`<br>`Wan-AI/Wan2.2-Animate-14B-Diffusers`<br>`Wan-AI/Wan2.2-I2V-A14B-Diffusers`<br>`Wan-AI/Wan2.2-T2V-A14B-Diffusers`<br>`Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| video | [ltx_video](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/ltx_video.md) | `Lightricks/LTX-Video`<br>`Lightricks/LTX-Video-0.9.5`<br>`Lightricks/LTX-Video-0.9.7-dev`<br>`Lightricks/LTX-Video-0.9.7-distilled`<br>`Lightricks/LTX-Video-0.9.8-13B-distilled`<br>`Lightricks/LTX-Video-Cakeify-LoRA`<br>`Lightricks/ltxv-spatial-upscaler-0.9.7`<br>`a-r-r-o-w/LTX-0.9.8-Latent-Upsampler`<br>`city96/LTX-Video-gguf` |
| video | [ltx2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/ltx2.md) | `Lightricks/LTX-2`<br>`Lightricks/LTX-2.5-22b-LoRA-Slow-Motion-Control`<br>`Lightricks/LTX-2.5-Diffusers`<br>`google/gemma-3-12b-it-qat-q4_0-unquantized`<br>`google/gemma-4-E2B-it` |
| video | [sana_video](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/sana_video.md) | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` |
| video | [skyreels_v2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/skyreels_v2.md) | `Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers`<br>`Skywork/SkyReels-V2-DF-14B-540P-Diffusers`<br>`Skywork/SkyReels-V2-DF-14B-720P-Diffusers`<br>`Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers`<br>`Skywork/SkyReels-V2-I2V-14B-540P-Diffusers`<br>`Skywork/SkyReels-V2-I2V-14B-720P-Diffusers`<br>`Skywork/SkyReels-V2-T2V-14B-540P-Diffusers`<br>`Skywork/SkyReels-V2-T2V-14B-720P-Diffusers` |
| video | [cogvideox](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/cogvideox.md) | `THUDM/CogVideoX-2b`<br>`THUDM/CogVideoX-5b` |
| video | [hunyuan_video](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/hunyuan_video.md) | `hunyuanvideo-community/HunyuanVideo`<br>`lucataco/hunyuan-steamboat-willie-10` |
| video | [hunyuan_video15](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/hunyuan_video15.md) | Sin ID autor/repo literal confirmado en ese documento |
| video | [framepack](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/framepack.md) | `hunyuanvideo-community/HunyuanVideo`<br>`lllyasviel/FramePackI2V_HY`<br>`lllyasviel/FramePack_F1_I2V_HY_20250503`<br>`lllyasviel/flux_redux_bfl` |
| video | [mochi](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/mochi.md) | `Comfy-Org/mochi_preview_repackaged`<br>`genmo/mochi-1-preview` |
| video | [cosmos](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/cosmos.md) | Sin ID autor/repo literal confirmado en ese documento |
| video | [cosmos3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/cosmos3.md) | `nvidia/Cosmos3-Nano`<br>`nvidia/Cosmos3-Super` |
| video | [allegro](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/allegro.md) | `rhymes-ai/Allegro` |
| video | [animatediff](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/animatediff.md) | `Lightricks/LongAnimateDiff`<br>`SG161222/Realistic_Vision_V5.1_noVAE`<br>`emilianJR/epiCRealism`<br>`guoyww/animatediff-motion-adapter-sdxl-beta`<br>`guoyww/animatediff-motion-adapter-v1-5-2`<br>`guoyww/animatediff-motion-adapter-v1-5-3`<br>`guoyww/animatediff-sparsectrl-rgb`<br>`guoyww/animatediff-sparsectrl-scribble`<br>`lllyasviel/Annotators`<br>`lllyasviel/ControlNet-v1-1`<br>`lllyasviel/sd-controlnet-depth`<br>`lllyasviel/sd-controlnet-openpose`<br>`stabilityai/sd-vae-ft-mse`<br>`wangfuyun/AnimateLCM` |
| video | [easyanimate](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/easyanimate.md) | `alibaba-pai/EasyAnimateV5.1-12b-zh`<br>`alibaba-pai/EasyAnimateV5.1-12b-zh-Control`<br>`alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera`<br>`alibaba-pai/EasyAnimateV5.1-12b-zh-InP` |
| video | [consisid](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/consisid.md) | `BestWishYsh/ConsisID-preview` |
| video | [kandinsky5_video](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/kandinsky5_video.md) | `kandinsky-community/kandinsky-3`<br>`kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers` |
| video | [wan_animate_2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/wan_animate_2.md) | `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` |
| video | [svd](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_diffusion/svd.md) | `stabilityai/stable-video-diffusion-img2vid`<br>`stabilityai/stable-video-diffusion-img2vid-xt` |
| video | [helios](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/helios.md) | `BestWishYsh/Helios-Base`<br>`BestWishYsh/Helios-Distilled`<br>`BestWishYsh/Helios-Mid` |
| video | [motif_video](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/motif_video.md) | `Motif-Technologies/Motif-Video-2B` |
| video | [minimax_h3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/minimax_h3.md) | `MiniMaxAI/MiniMax-H3` |
| audio | [audioldm2](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/audioldm2.md) | `anhnct/audioldm2_gigaspeech`<br>`anhnct/audioldm2_ljspeech`<br>`cvssp/audioldm2`<br>`cvssp/audioldm2-large`<br>`cvssp/audioldm2-music` |
| audio | [stable_audio](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_audio.md) | `stabilityai/stable-audio-open-1.0` |
| audio | [stable_audio_3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/stable_audio_3.md) | Sin ID autor/repo literal confirmado en ese documento |
| audio | [ace_step](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/ace_step.md) | `ACE-Step/acestep-v15-base`<br>`ACE-Step/acestep-v15-sft`<br>`ACE-Step/acestep-v15-xl-turbo-diffusers` |
| audio | [longcat_audio_dit](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/longcat_audio_dit.md) | `ruixiangma/LongCat-AudioDiT-1B-Diffusers` |
| audio | [minimax_music3](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/minimax_music3.md) | `MiniMaxAI/MiniMax-Music3` |
| vision | [marigold](https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/marigold.md) | `SG161222/RealVisXL_V4.0`<br>`diffusers/controlnet-depth-sdxl-1.0`<br>`madebyollin/taesd`<br>`prs-eth/marigold-depth-v1-1`<br>`prs-eth/marigold-iid-appearance-v1-1`<br>`prs-eth/marigold-iid-lighting-v1-1`<br>`prs-eth/marigold-normals-v0-1`<br>`prs-eth/marigold-normals-v1-1` |
| tts | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/README.md) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`<br>`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`<br>`Qwen/Qwen3-TTS-12Hz-1.7B-Base`<br>`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`<br>`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`<br>`Qwen/Qwen3-TTS-Tokenizer-12Hz` |
| tts | [chatterbox](https://github.com/resemble-ai/chatterbox/blob/ab55923c27757c90cfcc407baa57dc25a7d25217/README.md) | `ResembleAI/Chatterbox-Multilingual-es-es`<br>`ResembleAI/Chatterbox-Multilingual-es-mx-latam`<br>`ResembleAI/Chatterbox-Multilingual-hi`<br>`ResembleAI/Chatterbox-Multilingual-pt-br`<br>`ResembleAI/Chatterbox-Multilingual-pt-pt`<br>`ResembleAI/Chatterbox-Multilingual-zh-cmn`<br>`ResembleAI/chatterbox-nano` |
| tts | [kokoro](https://github.com/hexgrad/kokoro/blob/dfb907a02bba8152ca444717ca5d78747ccb4bec/README.md) | `hexgrad/Kokoro-82M` |
| tts | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/README.md) | `FunAudioLLM/CosyVoice-300M`<br>`FunAudioLLM/CosyVoice2-0.5B`<br>`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| audio | [ACE-Step-1.5](https://github.com/ACE-Step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/README.md) | `ACE-Step/Ace-Step1.5`<br>`ACE-Step/acestep-v15-base`<br>`ACE-Step/acestep-v15-sft`<br>`ACE-Step/acestep-v15-xl-base`<br>`ACE-Step/acestep-v15-xl-sft`<br>`ACE-Step/acestep-v15-xl-turbo` |
| vision | [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md) | `Qwen/QVQ-72B-Preview`<br>`Qwen/Qwen2-VL-72B-Instruct`<br>`Qwen/Qwen2-VL-72B-Instruct-AWQ`<br>`Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4`<br>`Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8`<br>`Qwen/Qwen2.5-VL-32B-Instruct`<br>`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`<br>`Qwen/Qwen2.5-VL-72B-Instruct-AWQ`<br>`Qwen/Qwen2.5-VL-7B-Instruct-AWQ`<br>`Qwen/Qwen3-VL-235B-A22B-Instruct`<br>`Qwen/Qwen3-VL-235B-A22B-Thinking`<br>`Qwen/Qwen3-VL-2B-Instruct`<br>`Qwen/Qwen3-VL-2B-Thinking`<br>`Qwen/Qwen3-VL-30B-A3B-Instruct`<br>`Qwen/Qwen3-VL-30B-A3B-Thinking`<br>`Qwen/Qwen3-VL-32B-Instruct`<br>`Qwen/Qwen3-VL-32B-Thinking`<br>`Qwen/Qwen3-VL-4B-Instruct`<br>`Qwen/Qwen3-VL-4B-Thinking`<br>`Qwen/Qwen3-VL-8B-Instruct`<br>`Qwen/Qwen3-VL-8B-Thinking` |
| video | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/README.md) | `Qwen/Qwen3-235B-A22B-Thinking-2507`<br>`Qwen/Qwen3-VL-235B-A22B-Instruct`<br>`tencent/HunyuanVideo-1.5` |
| image | [flux2](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/README.md) | `black-forest-labs/FLUX.1-dev`<br>`black-forest-labs/FLUX.2-dev`<br>`black-forest-labs/FLUX.2-klein-4B`<br>`black-forest-labs/FLUX.2-klein-9B`<br>`black-forest-labs/FLUX.2-klein-9b-kv`<br>`black-forest-labs/FLUX.2-klein-base-4B`<br>`black-forest-labs/FLUX.2-klein-base-9B`<br>`mistralai/Mistral-Small-3.2-24B-Instruct-2506` |

## Catálogo ejecutable y complementos de Studio en esta revisión

Esta tabla identifica la receta, no prueba que quepa en la GPU actual. RAM/VRAM/disco y dependencias se evalúan en tiempo real; los LoRAs requieren además la base exacta y los componentes asociados. Los mínimos VRAM históricos del catálogo son estimaciones de receta y pueden ser insuficientes si se aumentan resolución, frames o adaptadores. No se reutilizan como un cálculo universal para modelos descubiertos.

| ID | Tipo | Motor | VRAM base estimada GiB | Pesos estimados GiB |
|---|---|---|---:|---:|
| `qwen-image-edit-2511-pack` | image | `qwen_image_edit` | 14.0 | 16.6 |
| `qwen-image-2512-pack` | image | `qwen_image_t2i` | 14.0 | 16.6 |
| `sdxl-turbo-fp16` | image | `sdxl_turbo` | 5.5 | 6.5 |
| `qwen-lora-nsfw` | lora | `qwen_lora` | 0.2 | 0.35 |
| `qwen-lora-lightning-8steps` | lora | `qwen_lora` | 0.3 | 0.8 |
| `qwen-lora-multi-angles-3d` | lora | `qwen_lora` | 0.2 | 0.3 |
| `qwen-lora-next-scene` | lora | `qwen_lora` | 0.2 | 0.3 |
| `qwen-lora-anylight-env` | lora | `qwen_lora` | 0.2 | 0.56 |
| `qwen-lora-relight-scene` | lora | `qwen_lora` | 0.2 | 0.29 |
| `qwen-lora-studio-delight` | lora | `qwen_lora` | 0.2 | 0.24 |
| `qwen-lora-skin-realism` | lora | `qwen_lora` | 0.2 | 0.3 |
| `qwen-lora-ultra-portrait` | lora | `qwen_lora` | 0.2 | 0.24 |
| `qwen-lora-upscale-unblur` | lora | `qwen_lora` | 0.2 | 0.3 |
| `qwen-lora-flatlog-cinema` | lora | `qwen_lora` | 0.2 | 0.24 |
| `qwen-lora-anything2real` | lora | `qwen_lora` | 0.3 | 0.58 |
| `qwen-lora-anime-v2` | lora | `qwen_lora` | 0.2 | 0.24 |
| `qwen-t2i-lora-snapshot-real` | lora | `qwen_lora` | 0.2 | 0.29 |
| `sdxl-lora-high-detail` | lora | `sdxl_lora` | 0.2 | 0.23 |
| `sdxl-lora-cinematic-light` | lora | `sdxl_lora` | 0.1 | 0.05 |
| `lora-wan22-ti2v-5b-turbo` | lora | `video_lora` | 0.4 | 0.33 |
| `lora-wan22-ti2v-5b-fastwan` | lora | `video_lora` | 0.6 | 0.66 |
| `lora-wan21-14b-fusionx-phantom` | lora | `video_lora` | 0.4 | 0.31 |
| `lora-wan21-14b-fusionx-i2v` | lora | `video_lora` | 0.4 | 0.37 |
| `lora-wan21-14b-lightx2v-i2v` | lora | `video_lora` | 0.4 | 0.37 |
| `lora-wan21-14b-lightx2v-v2` | lora | `video_lora` | 0.4 | 0.32 |
| `lora-wan21-1.3b-bidirect2` | lora | `video_lora` | 0.2 | 0.09 |
| `lora-wan21-1.3b-rcm-turbo` | lora | `video_lora` | 0.3 | 0.19 |
| `lora-wan21-14b-unianimate-char` | lora | `video_lora` | 0.5 | 1.23 |
| `lora-wan21-14b-moviigen-cinema` | lora | `video_lora` | 0.3 | 0.32 |
| `lora-wan21-14b-detailz` | lora | `video_lora` | 0.3 | 0.31 |
| `lora-wan21-14b-snapshot-real` | lora | `video_lora` | 0.3 | 0.31 |
| `lora-wan21-1.3b-classic-anim` | lora | `video_lora` | 0.1 | 0.05 |
| `lora-wan22-a14b-svi-continuity` | lora | `video_lora` | 0.6 | 1.23 |
| `lora-wan22-a14b-lightx2v-4step` | lora | `video_lora` | 0.5 | 0.63 |
| `lora-ltx13b-detailer-ic` | lora | `video_lora` | 0.6 | 1.36 |
| `lora-ltx2-detailer-ic` | lora | `video_lora` | 0.8 | 2.62 |
| `lora-ltx2-cam-static` | lora | `video_lora` | 0.3 | 0.33 |
| `lora-ltx2-cam-dolly-in` | lora | `video_lora` | 0.3 | 0.33 |
| `wan22-ti2v-5b` | video | `wan_ti2v` | 13.5 | 31.9 |
| `wan22-ti2v-5b-turbo` | video | `wan_ti2v` | 12.5 | 22.8 |
| `wan21-vace-14b-fp8` | video | `wan_vace_fp8` | 15.8 | 29.1 |
| `wan21-flf2v-14b-fp8` | video | `wan_i2v_14b_fp8` | 16.2 | 30.7 |
| `wan21-fun-inp-14b-fp8` | video | `wan_i2v_14b_fp8` | 16.0 | 30.7 |
| `ltx-video-13b-distilled` | video | `ltx_video_distilled` | 14.5 | 45.8 |
| `wan21-vace-1.3b` | video | `wan_vace` | 8.0 | 17.7 |
| `wan21-fun-1.3b-inp` | video | `wan_i2v_14b` | 8.5 | 29.5 |
| `skyreels-v2-i2v-1.3b` | video | `skyreels_v2` | 9.5 | 29.8 |
| `ltx-video-2b` | video | `ltx_video` | 9.5 | 23.6 |
| `sana-video-2b-480p` | video | `sana_video` | 7.0 | 13.0 |
| `wan21-t2v-1.3b` | video | `wan_t2v` | 8.0 | 27.0 |
| `cogvideox-5b-i2v` | video | `cogvideox_i2v` | 13.0 | 20.6 |
| `ltx-2-19b-av` | video | `ltx2_av` | 21.5 | 58.0 |
| `wan21-i2v-14b-480p` | video | `wan_i2v_14b` | 21.5 | 68.0 |
| `wan21-flf2v-14b-720p` | video | `wan_i2v_14b` | 24.0 | 88.0 |
| `wan22-i2v-a14b` | video | `wan_i2v_14b` | 36.0 | 117.7 |
| `ffmpeg-motion-cpu` | video | `ffmpeg_motion` | 0.0 | 0.0 |
| `edge-tts-cloud` | tts | `edge_tts` | 0.0 | 0.0 |
| `mms-tts-spa` | tts | `mms_tts` | 1.0 | 0.14 |
| `kokoro-82m-tts` | tts | `kokoro_tts` | 0.5 | 0.35 |
| `bark-small-tts` | tts | `bark_tts` | 2.0 | 1.7 |
| `qwen25-vl-3b-coherence` | vision | `qwen25_vl_coherence` | 3.2 | 6.4 |
| `opencv-fast-coherence` | vision | `opencv_fast` | 0.0 | 0.0 |
| `edtalk-256-lipsync-hd` | lipsync | `edtalk_onnx_hd` | 1.5 | 0.65 |
| `musicgen-small-bgm` | audio | `musicgen` | 2.3 | 2.0 |
| `audioldm2-sfx` | audio | `audioldm2` | 3.5 | 3.8 |
| `procedural-ambience-cpu` | audio | `procedural_dsp` | 0.0 | 0.0 |
| `sdxl-base-1.0` | image | `sdxl_base` | 7.0 | 9.0 |
| `flux1-schnell` | image | `flux_schnell` | 24.0 | 36.0 |
| `flux2-klein-4b` | image | `flux2_klein` | 14.0 | 19.0 |
| `z-image-turbo` | image | `z_image_turbo` | 16.0 | 23.0 |
| `pixart-sigma-xl-1024` | image | `pixart_sigma` | 12.0 | 24.0 |
| `sana-600m-1024` | image | `sana_image` | 10.0 | 20.0 |
| `sana-1600m-1024` | image | `sana_image` | 12.0 | 24.0 |
| `sana-sprint-600m` | image | `sana_sprint` | 10.0 | 20.0 |

## Fuentes del protocolo de descubrimiento

- [Hugging Face Hub `HfApi.list_models` / `model_info`](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py): API pública, expand, filtros, metadatos y paginación. `expand` no se combina con `full`/`cardData`/`fetch_config`; algunas propiedades faltan y no se sustituyen por datos ficticios.
- [Tarjetas de modelos](https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/package_reference/cards.md): base, tarea y licencia son metadatos declarados, no validación de ejecución.
- [Diffusers release fijada](https://github.com/huggingface/diffusers/releases/tag/v0.42.1): soporte upstream a la fecha de la investigación; el runtime local puede ser anterior. Una actualización de metadatos no actualiza Diffusers.

### Validación aún necesaria

Inferencia real de cada receta nueva en L4/Colab (incluido Python 3.13/CUDA), uso pico de VRAM/RAM con offload, descarga/variantes precisas, dependencias CUDA/ABI, acceso a repos gated y calidad de salidas. Los tests deterministas de proveedor, cache, formatos, oráculo y API no sustituyen esa validación. Hasta medir, no dar cifras de segundos por imagen/clip como garantía ni etiquetar una receta como GPU probada.
