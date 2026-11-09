# Manual de modelos y oráculo MCP — Studio IA

Revisión: **2026-11-05**. Este manual describe el código de Studio, no una certificación de inferencia. Las recetas nuevas se han contrastado con documentación oficial y sus contratos se prueban sin GPU; **no se han ejecutado benchmarks en T4/L4/A100/H100 en esta revisión**.

## 1. Qué significa «compatible»

Se comprueban e informan ejes separados:

1. **Soporte de ejecución**: `reviewed_recipe` = existe una receta implementada con clase, entradas y componentes definidos; `candidate` = referencia/descubrimiento pendiente de receta; `unsupported` = formato, código remoto o backend que este cargador no admite. Un pipeline presente en Diffusers no basta para soportarlo en Studio.
2. **Capacidad estimada**: VRAM **total de la primera GPU**, RAM del sistema y disco libre. `fits_estimate`, `insufficient` o `unknown` para cada recurso. No se suman las memorias de varias GPU: no existe distribución multi-GPU automática.
3. **Disponibilidad actual**: VRAM libre y RAM disponible pueden ser menores que la capacidad. Descargar de RAM/VRAM el pipeline anterior puede liberar espacio; eliminar sus archivos NO es necesario. Volver a medir tras liberar memoria.
4. **Runtime**: paquetes, versiones mínimas, clases exportadas por la instalación local y FFmpeg (incluido el binario de imageio-ffmpeg). `ffprobe` se informa, pero no se exige si se usa el fallback existente. La inspección de paquetes no importa frameworks; la telemetría de hardware reutiliza la detección habitual. Buscar no carga pesos ni reserva VRAM de inferencia. Una comprobación estática correcta no prueba ABI CUDA, kernels ni calidad de inferencia.
5. **Acceso/licencia**: `public_metadata` solo describe visibilidad de metadatos. Un modelo gated requiere aceptar las condiciones y comprobar acceso a los pesos con la cuenta adecuada. **Tener un token no demuestra que la cuenta tenga permiso**. Una licencia desconocida permanece desconocida.

`compatibility.compatible=true` significa **receta revisada con encaje estimado de capacidad**, NO «funciona garantizado». `runnable_now` además exige dependencias comprobadas, pesos locales completos (en las recetas nuevas se comprueban componentes/shards y variante, no solo un `model_index.json`), memoria actual conocida/suficiente y hardware real; una simulación nunca es ejecutable. Son prerrequisitos estáticos, no una prueba de inferencia (`readiness_level=static_prerequisites_only_not_inference_validation`). El descubridor no verifica por sí solo el permiso de descarga.

`gpu_tested=false`, `confidence=engineering_estimate` y `limits` hacen explícita la incertidumbre. Los números son presupuestos aproximados en GiB para valores predeterminados: ni picos medidos ni garantías de tiempo/calidad. La descarga real vuelve a comprobar el inventario de archivos y el espacio antes de bajar pesos.

## 2. Primera acción del agente

```json
{"name":"studio_get_state_and_manual","arguments":{"include_recent_assets":true}}
```

Leer `hardware.gpu`, `hardware.ram`, `hardware.disk`, `loaded_in_vram`, `downloaded_on_disk`, `runtime`, `model_discovery` y `manual_index`. Si necesitas el manual completo sin soporte MCP de recursos:

```json
{"name":"studio_get_model_guide","arguments":{"guide":"operations"}}
```

Recursos MCP equivalentes:

- `studio://manual/model-oracle`: este manual.
- `studio://manual/model-research`: investigación, familias, fuentes y límites.
- `studio://models/discovery-status`: estado ligero del índice/updater.
- `studio://models/recipes`: recetas del catálogo y diagnósticos sobre hardware real.
- Los recursos `quickstart`, `video-recipes`, `character-consistency` y `lipsync-and-sound` siguen disponibles. Las tablas estáticas son orientativas: **manda el diagnóstico actual del oráculo**, no una cifra histórica de segundos.

## 3. Buscar modelos sin descargar pesos

```json
{"name":"studio_search_models","arguments":{"query":"Wan","category":"video","only_compatible":true,"gpu_profile":"auto","limit":12,"offset":0}}
```

Categorías: `all`, `image`, `video`, `lora`, `tts`, `audio`, `vision`, `lipsync`. Orden: `downloads` o `lastModified`. `limit`: 1–50. Usar el `next_offset` devuelto para otra página; no inventar cursores ni IDs. La instantánea mantiene el orden de páginas ya mostradas y añade páginas remotas al final; refrescar/TTL reinicia el orden. El orden agregado de categorías no es un ranking mundial exhaustivo. `refresh=true` pide metadatos nuevos respetando la espera por fallos/429.

La búsqueda combina:

- recetas locales ya implementadas;
- referencias a modelos/componentes en documentación oficial, con fuente y revisión;
- metadatos públicos recientes de Hugging Face (API paginada).

`only_compatible=true` **excluye candidatos sin receta y estimaciones desconocidas**, aunque sean pequeños o muy populares. La vista filtrada pagina las recetas aptas ya conocidas, no recorre candidatos remotos indefinidamente. `all` consulta sin filtro de tarea; una tarea que no se puede clasificar se devuelve como `category=unknown`, nunca se convierte en imagen por defecto. Para investigar novedades, usar `false` y leer `support`, `recipes`, `compatibility`, `access`, `source_urls`, `revision` y `metadata_checked_at`.

### Leer la frescura correctamente

- `source_mode=network`: se consultó el proveedor en esta petición.
- `cache`: consulta previamente obtenida; mostrar su fecha, NO llamarla «en vivo».
- `offline`: error, red bloqueada o enfriamiento; se conservan los datos anteriores y las referencias locales.
- `refreshing`: la misma consulta o las dos plazas de red ya están en curso; no lanzar otra copia. La web reconsulta el índice sin duplicar la descarga de metadatos.
- `stale=true`: no hay consulta válida reciente (TTL de 15 minutos).

La fecha de investigación NO es una consulta actual a Hugging Face. Descargas/likes desconocidos se muestran como desconocidos, nunca se fabrican. Una respuesta parcial/error no vacía el índice ni destruye los pesos existentes. La búsqueda por texto se actualiza al escribir, con debounce; no es una suscripción exhaustiva a todos los modelos del mundo.

### Inspeccionar antes de decidir

```json
{"name":"studio_inspect_model","arguments":{"repo_id":"black-forest-labs/FLUX.2-klein-4B"}}
```

La inspección consulta inventario, revisión, metadatos de licencia/base y pequeños JSON (`model_index.json`, `adapter_config.json`, `config.json`, `transformer/config.json`) fijados al mismo commit público. No lee ni ejecuta Python o instrucciones del README, no activa `trust_remote_code`, no baja safetensors/GGUF/pickle. Los errores de archivos gated se muestran sin afirmar que el acceso esté concedido.

Para una receta de Studio, usar su **model_id**, no la URL/repo HF:

```json
{"name":"studio_inspect_model","arguments":{"model_id":"wan21-vace-14b-fp8"}}
```

Devuelve definición, artefactos/compañeros, especificación, dependencias y evaluación actual. `repo_id` y `model_id` son alternativas: pasar exactamente uno. Un archivo de transformer FP8 conocido puede ser solo `component_of` una receta multirrepositorio, no un pipeline autónomo.

## 4. Actualización manual y automática

```json
{"name":"studio_refresh_model_index","arguments":{"category":"all","refresh_now":true}}
```

Devuelve un **job de metadatos**. Esperar ese `job_id` con `studio_wait_for_job`. Consultar estado ligero con:

```json
{"name":"studio_model_index_status","arguments":{}}
```

Configuración persistente, sin iniciar una actualización inmediata:

```json
{"name":"studio_refresh_model_index","arguments":{"auto_refresh":true,"interval_minutes":360,"refresh_now":false}}
```

- Por defecto: automática cada 6 horas; primer ciclo no bloquea el arranque. Se puede desactivar en la web/MCP o inicialmente con `STUDIO_MODEL_AUTO_REFRESH=0`.
- Intervalo configurable: 15–1440 minutos. Solo funciona mientras el servidor está en marcha; apagar Colab detiene el actualizador.
- Cada ciclo consulta un conjunto acotado de categorías recientes y hasta dos búsquedas utilizadas, no todo Hugging Face. Solo hay dos consultas concurrentes; cada consulta tiene timeout de conexión/lectura, presupuesto temporal y límites de bytes. Topes: 2500 repositorios, 80 consultas, 240 resultados remotos por consulta; límite adicional de tamaño del índice. `query_limit_reached` indica cobertura truncada.
- La caché está en `STUDIO_DATA_DIR/state/model_discovery.json`, con reemplazo atómico, límites y recuperación ante corrupción. No contiene tokens ni resultados privados. Persistencia significa dentro de ese directorio: el disco temporal de Colab sigue siendo efímero.
- Errores de red y 429 activan espera exponencial/`Retry-After`; no se insiste por cada tecla ni se lanzan refrescos idénticos simultáneos.
- **Actualizar metadatos NO actualiza librerías, recetas, código, pesos ni licencias.** Integrar una arquitectura nueva requiere revisión de código y pruebas. No auto-instalar paquetes ni convertir candidatos en ejecutables.

## 5. Usar el oráculo, no elegir solo por parámetros

```json
{"name":"studio_recommend_pipeline","arguments":{"task":"video_flf2v","priority":"balanced_fast"}}
```

Tareas: `video_flf2v`, `video_i2v`, `video_t2v`, `consistent_keyframes`, `coherence_check`, `lipsync_talking_scene`, `music_and_ambience`, `image_create`, `image_edit`, `tts`, `montage`.

Prioridades: `max_quality_14b`, `balanced_fast`, `ultra_fast_draft`, `zero_vram_cpu`. Son preferencias de flujo, no benchmarks comparativos. El oráculo:

1. Solo considera IDs de recetas revisadas y el modo exacto solicitado.
2. Comprueba presupuesto VRAM/RAM/disco, precisión y runtime con el hardware **real**, no con las pestañas de simulación.
3. Prioriza prerrequisitos ya listos y reutilizar pesos locales entre recetas aptas de alcance equivalente; no propone limpiar/bajar otra variante por un timeout. OpenCV no gana a una auditoría semántica solo por no tener pesos. DSP no se presenta como canción/música IA; el alcance reducido se explica en `caveats`.
4. Devuelve `assessment`, `alternatives` con motivos de descarte, `recommended_tool`, `recommended_arguments`, `required_inputs`, `next_actions` y limitaciones.
5. Puede devolver `no_compatible_recipe` y `recommended_tool=null`. Es una respuesta correcta; no inventar un cargador para rellenarla. Si el agente omite `model_id` al generar vídeo y el oráculo no tiene encaje, se devuelve un error guiado, **sin fallback oculto a Wan 1.3B**.

`status=needs_preparation` puede significar dependencias, pesos, presión de memoria o falta de espacio para salidas. Completar primero esos pasos. Los argumentos recomendados son una **plantilla**, no una orden ejecutable sin entradas: rellenar prompts y rutas reales de `required_inputs`, obtenidas de la galería/subida. `required_one_of` especifica alternativas, por ejemplo audio único o turnos de diálogo en LipSync. Las plantillas incluyen `wait=false` y no sustituyen las entradas reales. No enviar literalmente `FRAME_1`, `ANCHOR`, `...` ni rutas de ejemplo.

### Hardware y precisión

| Entorno | Qué comprobar / orientación, no garantía |
|---|---|
| CPU | FFmpeg (montaje/movimiento), DSP, OpenCV, MMS-TTS y Edge (servicio de red). No equivalen a T2I/T2V generativo GPU. Bark/Kokoro dependen de su instalación y RAM. |
| T4 ~15 GiB | Priorizar recetas FP16 ligeras como SDXL Turbo/Base; vídeo depende de su receta, resolución, offload y RAM. Las recetas nuevas marcadas `native_bf16` NO se autorizan para T4/V100. |
| L4 ~22 GiB reales | Qwen 4-bit, recetas Wan FP8, LTX destilado y nuevas imágenes pueden ser candidatos con receta según RAM/disco; un LTX-2/encoder grande puede descartarse por RAM aunque la VRAM parezca suficiente. |
| A100 40/80, H100 | Más VRAM no elimina límites de RAM, disco, gate o versión/clase instalada. Modelos MoE/doble denoiser y encoders muy grandes siguen requiriendo su presupuesto completo. |

Los perfiles de la web son **simulaciones solo de GPU**; RAM/disco siguen siendo los reales de la sesión. No aparecen modelos en una GPU simulada que el agente pueda ejecutar en la CPU actual. Solo se evalúa la GPU 0, no la suma de varias.

### No confundir cuantizaciones

- 4-bit del denoiser no implica 4-bit del encoder, VAE, activaciones, atención ni cachés.
- FP8 almacenado no implica que toda la ejecución sea FP8; puede necesitar casting BF16/FP16, hooks y offload específicos.
- GGUF/NVFP4/MLX/ComfyUI/WanGP no son intercambiables con esta receta Diffusers. Solo se admite el archivo concreto cuando existe integración explícita; no importar un nombre similar y esperar que funcione.
- El tamaño de parámetros safetensors/inventario puede contar varias variantes o solo un componente. **No se transforma automáticamente en una promesa de VRAM.** Estimación desconocida = desconocida.

## 6. LoRAs: siempre modelo base y submodo exactos

Leer `compatible_models`, `parent_model`, `lora_target`, `incompatible_submodes` y `official_spec`. El presupuesto incluye **base + LoRA + compañeros**, no solo los 100–500 MB del adaptador. El mínimo VRAM del registro LoRA es **incremental** (`vram_budget_scope`); el diagnóstico suma la base exacta, RAM incremental y los pesos que faltan, exige PEFT y no considera listo el conjunto por tener solo un marcador del adaptador. `assessed_base_model_id` y `base_options` explican qué base se evaluó. Un LoRA Wan 14B no se asigna a Wan 1.3B, VACE, Qwen o LTX por tener palabras parecidas.

En `flf2v` están prohibidos los modelos de una sola imagen y los LoRAs causales declarados incompatibles. Qwen 4-bit: cargar los adaptadores **antes** de CPU offload; para una secuencia, precargar los futuros con escala `0.0` y después activar solo los necesarios. No inyectar adaptadores sobre tensores meta/offloaded, no usar `to_empty()` para tapar pesos ausentes.

Las importaciones manuales se guardan como `validation_state=candidate`. Se conservan sus archivos/definiciones, pero no se inventa `qwen_lora`, `kokoro_tts`, `video_modes` ni compatibilidad a partir del nombre. Descarga/carga y enums MCP de ejecución excluyen esos candidatos hasta que exista receta revisada.

## 7. Parámetros, entradas y límites de cada motor

- **Imagen**: `studio_generate_or_edit_image`, `use_official_defaults=true`. Cada familia tiene pasos/CFG propios: no usar los 4 pasos Lightning de Qwen en SDXL Base, PixArt o SANA. Lumina 2 sigue como candidato: falta el cargador explícito de su transformer `.pth`/GGUF; no se habilita un loader genérico. FLUX Schnell usa CFG 0; Z-Image Turbo usa 9 pasos (8 NFE); Klein destilado tiene su propia receta. No enviar imagen gris a un pipeline opcional de referencia cuando se pretende T2I. Los T2I puros rechazan referencias, no las ignoran silenciosamente.
- **Edición**: pasar `base_image_rel` y, si procede, `reference_rels`. La máscara de Studio hace composición posterior en los motores sin inpainting nativo: no afirmar que eso entrena o añade un ControlNet/inpainting especializado.
- **Vídeo**: `t2v` no necesita imagen; `i2v` necesita inicial; `flf2v` necesita inicial y final. Respetar `video_modes`, buckets y regla temporal `4k+1`/`8k+1`. El tiempo final puede diferir de `duration_sec` al ajustar frames/fps; medir antes del montaje.
- **Voz**: IDs locales reales `mms-tts-spa`, `kokoro-82m-tts`, `bark-small-tts`; `edge-tts-cloud` es servicio de red, no modelo descargable gratuito garantizado. No atribuir clonación de voz a un ID que no la implemente. Comprobar idioma, pronunciación y licencia por separado. Kokoro requiere su backend opcional y `espeak-ng`; utiliza el KModel y la voz española Dora locales de la receta, no un fallback Transformers ni los IDs de Edge. MMS/Bark son capaces de CPU, pero su backend elige GPU cuando hay CUDA; por eso se presupuesta su VRAM en una sesión GPU y no se presentan como opción de VRAM cero en ese caso. En CPU se evalúa su presupuesto RAM sin exigir GPU.
- **Audio**: DSP/ambiente, MusicGen instrumental y AudioLDM2 Foley son tareas distintas. No convertir todas las novedades musicales en `musicgen` por el tag `text-to-audio`.
- **Coherencia y LipSync**: medir y auditar; no prometer identidad 100%, boca perfecta ni detección infalible de caras/oclusiones.

## 8. Ejecución de trabajos y recuperación

1. Para escenas coherentes/cortos, llamar primero a `studio_get_workflow_preset`.
2. Todos los keyframes → auditorías/reparaciones → todos los vídeos → sonido/LipSync/montaje. No intercalar cargas de imagen/vídeo por cada plano.
3. Una operación explícita de generación puede preparar **solo su modelo elegido**. La búsqueda, el índice, la inspección y el oráculo nunca bajan pesos.
4. Para trabajos largos, `wait=false` y luego:

```json
{"name":"studio_wait_for_job","arguments":{"job_id":"ID_REAL_DEVUELTO","wait_seconds":80}}
```

`wait_seconds` admite 1–88 (NO `max_wait_seconds`). `running`, `queued`, heartbeat lento o timeout HTTP no son fallo del modelo. No duplicar preparación ni invocar otra variante en paralelo. Un ID de job se obtiene de una respuesta real.

5. Error de librería/PEFT/bitsandbytes/tensor meta: parar, conservar pesos y exportar log con `studio_export_session_log`. Actualizar/reiniciar el proceso correcto; no desconectar Colab ni borrar `/content/studio_data` como primer remedio.
6. OOM: revisar memoria real, liberar slots, reducir resolución/frames **dentro de la receta**, o elegir otra receta con autorización. No asumir que cualquier combinación de 4-bit, sequential/model/group offload y casting es válida.
7. Espacio insuficiente: exportar salidas, revisar sugerencias y **pedir permiso** antes de `delete_from_disk`. No borrar cachés/pesos de trabajo sin autorización.

### 8.1. Caída del proceso o túnel, no confundir con un fallo del modelo

Ver [procedimiento de rescate de Colab](COLAB_RECOVERY.md). Wan 14B FP8 usa dtype FP8 explícito y carga meta/CPU: no expandir a FP32 con dtype `None` antes del casting. Se vuelve a medir RAM después de liberar slots y antes de descargar/cargar; `HOST_RAM_BUDGET_EXCEEDED` conserva los pesos y no autoriza una descarga alternativa.

Un 502 o 530/1033 **no confirma OOM ni que la VM haya desaparecido**. El padre puede cerrar el túnel después de que muera solo el servidor. Revisar `state/diagnostics` y la celda de rescate del notebook: código/señal de salida, stderr, cloudflared, RSS/RAM y eventos cgroup. El ZIP de `GET /api/logs/diagnostics` oculta tokens y no incluye pesos/multimedia. Si la VM se elimina, su disco efímero también: descargar antes.

Actualizar también el puente `studio_mcp.py`. Una conexión interrumpida conserva `job_id`, devuelve `unknown_not_failed` y `do_not_resubmit=true`: reanudar consultando el mismo ID. Solo se reintentan lecturas; nunca repetir a ciegas un POST cuya respuesta se perdió.

El inspector usa archivos locales y la API real del catálogo. No hace un segundo intento BF16 tras fallar 4-bit. Si `hybrid_auto` no pudo ejecutar el VLM, `all_consistent=false`, `identity_verified=false`, `ready_for_video=false`: revisar `vlm_fallback_reason`. En `fast_cv` explícito, PASS/`all_consistent` solo se refieren a fotometría con el `effective_comparison_threshold` publicado; **no prueban identidad ni autorizan por sí solos el preset semántico de vídeo**.

## 9. API web equivalente y pruebas reproducibles

- `GET /api/catalog/hf-search?query=Wan&category=video&gpu_profile=auto&only_compatible=true&limit=12&offset=0`
- `GET /api/models/discovery/status`
- `POST /api/models/discovery/settings` con `{"auto_refresh":true,"interval_minutes":360}`
- `POST /api/models/discovery/refresh` con `{"category":"all"}` → job, no pesos.
- `GET /api/models/discovery/inspect?repo_id=Tongyi-MAI/Z-Image-Turbo`
- `GET /api/models/wan21-vace-14b-fp8/assessment`
- `GET /api/models/recommend?task=image_create&priority=balanced_fast` (hardware real)
- `GET /api/models/manual?guide=operations` o `guide=research` (Markdown, lista cerrada de documentos)

La interfaz no recurre al navegador para enviar tokens a Hugging Face ni sustituye un resultado offline por una estimación sin validar. Todos los diagnósticos proceden del mismo evaluador servidor/oráculo. Los tests usan proveedores y hardware simulados deterministas; una prueba sin CUDA no demuestra rendimiento real en Colab.


## 10. Inventario de recetas nuevas y comprobación en Colab

Hay **8 recetas nuevas de imagen**, además de las anteriores, con `gpu_tested=false`: SDXL Base FP16; FLUX.1 Schnell; FLUX.2 Klein 4B; Z-Image Turbo; PixArt Sigma; SANA 600M/1600M y SANA Sprint 600M. [Tabla de presupuestos y fuentes](MODEL_RESEARCH.md#recetas-nuevas-de-imagen). Diffusers debe contener la clase exacta: actualizar paquetes puede requerir reiniciar el proceso Python, no el almacenamiento de modelos.

SANA 600M usa transformer FP16, encoder BF16 y VAE FP32; SANA 1600M/Sprint usan transformer BF16. Los archivos canónicos pueden almacenarse FP32, aunque la ejecución use BF16/FP16: las estimaciones de disco no se reducen por cambiar el dtype de carga. PixArt cuenta el T5 completo. No afirmar que esta receta equivale al ejemplo upstream de menos de 8 GiB con T5 cuantizado y eliminación de componentes.

**Prueba mínima de una receta nueva en una Colab concreta**:

1. Registrar GPU real/VRAM libre, RAM disponible, disco y versiones. Confirmar `compatible`, leer `reasons`, `runtime`, precisión, licencia y acceso. No usar la simulación para declarar una prueba real.
2. Consultar/inspeccionar la receta y confirmar el presupuesto de descarga. Preparar únicamente ese ID con autorización; no descargar otro en caso de timeout. Conservar logs y pesos ya existentes.
3. Generar una imagen con defaults oficiales y semilla registrada. Medir pico de VRAM/RAM, duración, salida válida y comprobar que la composición/edición no se ha confundido con una capacidad no implementada.
4. Anotar ID, revisión de cada componente, parámetros, versiones, memoria y resultado. Solo entonces marcar el perfil concreto como probado; una ejecución en A100 no prueba T4/L4. Repetir con referencias/LoRAs solo si su contrato lo permite.

La batería automatizada de CPU usa hardware/proveedores simulados y verifica seguridad, clasificación, frescura, caché, paginación, contratos de carga y API/MCP. No descarga pesos ni certifica CUDA/ABI ni calidad. Para ejecutarla: `python -m pytest -q`, `node --check studio/static/app.js` y `python -m compileall -q app.py studio tests`.
