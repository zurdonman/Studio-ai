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

Al seleccionar un perfil GPU en la interfaz, se activa el filtro estricto en Vídeo y Catálogo: muestra solo recetas con `compatible=true`, aunque las demás ya estén descargadas o cargadas. El perfil L4 simula solo la GPU; emplea la RAM total y el disco libre de la sesión real. Desmarcar el filtro permite ver las demás recetas sin declararlas aptas. Un resultado vacío permanece vacío; no se carga un modelo alternativo automáticamente. «VRAM con encaje estimado · revisar RAM/disco» no significa que la GPU haya sido probada ni que el conjunto cumpla el filtro; las etiquetas distinguen el presupuesto de VRAM de los demás recursos. Las variantes LTX FP8 almacenan el transformer en FP8 después de una carga inicial BF16, por lo que requieren RAM suficiente aunque la VRAM estimada encaje. Antes de generar, comprobar memoria libre, dependencias, licencia, resolución y presupuesto combinado de los LoRAs.

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

### Prueba acelerada I2V: búsqueda, ejecución y aprobación

Una búsqueda vacía de `studio_search_models`, especialmente con `only_compatible=true`, no prueba que falte un LoRA: el índice agrupa por repositorio y el filtro elimina encajes no estimados compatibles. Consulta también `studio_list_models` por categoría `lora` e inspecciona el ID exacto. Ejemplo de candidato, no de ejecución GPU validada:

```json
{"name":"studio_inspect_model","arguments":{"model_id":"lora-wan22-ti2v-5b-fastwan"}}
{"name":"studio_inspect_model","arguments":{"model_id":"wan22-ti2v-5b"}}
{"name":"studio_recommend_pipeline","arguments":{"task":"video_i2v","priority":"balanced_fast","require_fast":true}}
```

FastWan es un archivo del repositorio `Kijai/WanVideo_comfy`, no su nombre de repositorio. La receta de catálogo lo declara para `wan22-ti2v-5b` en I2V, no FLF2V. Comprueba hardware real, base + LoRA, dependencias, acceso y presupuesto; `compatible`/`reviewed_recipe` son estimaciones, no prueba de descarga, carga, velocidad o calidad. No prepares ni descargues alternativas automáticamente tras un fallo.

Si el usuario pide aceleración, usa `require_fast=true` tanto al recomendar como en `studio_generate_video` (o activa la preferencia `use_fast_loras`). Para esta pareja pasa también `video_lora_id="lora-wan22-ti2v-5b-fastwan"` y `video_submode="i2v"`, una ruta de imagen real y `use_official_defaults=true`. `require_fast` admite modelos destilados sin LoRA, pero si el usuario exige un LoRA, no lo sustituyas por un destilado sin su autorización. No desactives la preferencia para saltarte el requisito. Si el LoRA falta, es incompatible o falla la carga, para e informa; no generes con la base sola.

El cargador de `wan_ti2v` separa tensores directos `diff`/`diff_b` de las matrices LoRA, comprueba destinos y formas, los aplica con la escala y restaura los originales al retirar o cambiar el adaptador. Ante `WAN_LORA_INCOMPLETE`, `WAN_LORA_UNSUPPORTED_KEY`, `WAN_LORA_DELTA_*` o `WAN_LORA_RESTORE_FAILED`, detente y conserva los logs; no descartes tensores, no fuerces un reintento y no continúes con la base. La implementación tiene pruebas sintéticas con Diffusers/PEFT en CPU: **no es una prueba del archivo completo ni de inferencia en L4**.

Si se pidió aprobación previa, ejecutar un solo clip de prueba. Esperar el mismo job, pasar los logs al usuario y comprobar en el resultado `active_lora_id`, `steps`, duración y tiempos realmente disponibles; no inferir valores ausentes. Esperar aprobación antes de los clips restantes, voces y montaje. No hay garantía de mejor anatomía, movimiento o tiempo; no confundir valores recomendados de pasos con mediciones reales.

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
6. La memoria se gestiona automáticamente: consultar `memory_guardian` antes de actuar. Ante OOM o parada preventiva, revisar memoria real y eventos, reducir resolución/frames **dentro de la receta**, o elegir otra receta con autorización. No asumir que cualquier combinación de 4-bit, sequential/model/group offload y casting es válida.
7. Espacio insuficiente: exportar salidas, revisar sugerencias y **pedir permiso** antes de `delete_from_disk`. No borrar cachés/pesos de trabajo sin autorización.

### 8.1. Caída del proceso o túnel, no confundir con un fallo del modelo

Ver [procedimiento de rescate de Colab](COLAB_RECOVERY.md). Wan 14B FP8 usa dtype FP8 explícito y carga meta/CPU: no expandir a FP32 con dtype `None` antes del casting. Se vuelve a medir RAM después de liberar slots y antes de descargar/cargar; `MEMORY_BUDGET_EXCEEDED` (antes `HOST_RAM_BUDGET_EXCEEDED`) conserva los pesos y no autoriza una descarga alternativa.

Un 502 o 530/1033 **no confirma OOM ni que la VM haya desaparecido**. El padre puede cerrar el túnel después de que muera solo el servidor. Revisar `state/diagnostics` y la celda de rescate del notebook: código/señal de salida, stderr, cloudflared, RSS/RAM y eventos cgroup. El ZIP de `GET /api/logs/diagnostics` oculta tokens y no incluye pesos/multimedia. Si la VM se elimina, su disco efímero también: descargar antes.

Actualizar también el puente `studio_mcp.py`. Una conexión interrumpida conserva `job_id`, devuelve `unknown_not_failed` y `do_not_resubmit=true`: reanudar consultando el mismo ID. Solo se reintentan lecturas; nunca repetir a ciegas un POST cuya respuesta se perdió.

El inspector usa archivos locales y la API real del catálogo. No hace un segundo intento BF16 tras fallar 4-bit. Si `hybrid_auto` no pudo ejecutar el VLM, `all_consistent=false`, `identity_verified=false`, `ready_for_video=false`: revisar `vlm_fallback_reason`. En `fast_cv` explícito, PASS/`all_consistent` solo se refieren a fotometría con el `effective_comparison_threshold` publicado; **no prueban identidad ni autorizan por sí solos el preset semántico de vídeo**.

### 8.2. Memoria automática, también sin agente conectado

Studio inicia un vigilante de RAM/VRAM con el servidor (1 s por defecto), usa memoria efectiva cgroup y reserva margen adicional (RAM: 10 % o 1 GiB; VRAM: 8 % o 0,5 GiB). Limpia caché y expulsa por LRU solo modelos inactivos, volviendo a medir. Todos los motores —también TTS, música y ONNX— comparten turno y protección de slots activos. Si el turno está ocupado, el vigilante espera a un punto seguro del motor; no borra un pipeline vivo desde otro hilo.

La admisión suma **base + componentes + todos los LoRA residentes + reserva**, incluso adaptadores a escala cero. Un preset publica `selected_video_stack.memory_assessment`; `ok=true` es compilación, no validación GPU. Una fase con `do_not_execute=true` no se ejecuta ni se sustituye automáticamente. Al reutilizar una base, la admisión incremental no cuenta de nuevo sus pesos.

Consultar `studio_get_state_and_manual.memory_guardian`, `GET /api/memory/status` y `job.memory_protection`. `MEMORY_BUDGET_EXCEEDED`/`MEMORY_BUDGET_UNKNOWN` paran la admisión; `MEMORY_SAFETY_STOP` para cooperativamente una operación sin margen. No repetir automáticamente el mismo job ni borrar archivos. Los eventos locales acotados se incluyen en el ZIP de diagnóstico. [Política completa y límites](MEMORY_MANAGEMENT.md): una asignación nativa individual puede adelantarse al muestreo; no es una garantía de cero OOM ni una prueba GPU.

### 8.3. Preferencias del usuario: una sola pregunta, nunca adivinada

El agente no debe renegociar el encargo ni preguntar en cada paso. `studio_get_workflow_preset` devuelve `production_preferences.pending_questions`; si hay alguna pendiente, se pregunta **una vez** con `studio_ask_user(preference_key=...)`, que ya trae la opción recomendada, y la respuesta se guarda con `studio_answer_preference(key=..., value=...)` en `<data>/mcp_preferences.json` (solo preferencias, nunca pesos ni credenciales). Las tareas siguientes reutilizan lo guardado.

| Clave | Recomendado | Efecto |
|---|---|---|
| `use_fast_loras` | `true` | Exige LoRA de velocidad compatible con modelo/submodo o modelo destilado; no garantiza tiempos. Si el usuario exige LoRA, no sustituirlo por un modelo destilado sin permiso. |
| `clip_duration_sec` | `5.0` | Tramo por clip; 30 s = 6 clips, 60 s = 12. |
| `aspect_ratio` | `16:9` | Bucket nativo de la receta; `9:16` y `1:1` si el usuario los pide. |
| `voice_pair` | Álvaro / Elvira | Dos voces distintas para alternar turnos y LipSync por personaje. |

**Caso que falló en producción:** el agente quería LoRAs rápidos, pero llamó a `studio_generate_video` con los adaptadores creativos y `video_lora_id` vacío. No era un error del motor: sin el LoRA destilado los pasos vuelven a los completos. Ahora esa llamada se rechaza con `SPEED_LORA_MISSING` y la acción exacta (incluir el `video_lora_id` rápido). Si el usuario prefiere calidad máxima, se guarda `use_fast_loras=false` y el plan ya no propone el LoRA rápido.

### 8.4. Fallo, reintento acotado y log

Antes de reintentar, `studio_get_work_state` devuelve preferencias, último trabajo (`status`, `error`, `progress`) y el estado real del modelo de vídeo. Códigos que exigen cambiar algo antes de repetir:

- `SPEED_LORA_MISSING`: repetir con `video_lora_id` rápido.
- `ADAPTER_STACK_INCOMPLETE`: el stack ya se ha retirado limpiamente; repetir el mismo stack una vez.
- `MODEL_LOAD_ERROR`, `RUNTIME_PATCHES_STALE`, `QWEN_VL_4BIT_STATE_LOST`: entorno/proceso, no la escena. Reiniciar el servidor conservando pesos; el aviso también aparece en la interfaz (`⚙ Compatibilidad`) y en `GET /api/runtime/status`.
- `MODEL_RESOURCE_CONSTRAINT`, `MEMORY_BUDGET_EXCEEDED`: leer `memory_guardian`; no borrar pesos.
- `DURATION_EXCEEDS_SINGLE_CLIP`: no es un fallo del motor. Un clip real llega a ~10 s (161 fotogramas @ 16 fps); la respuesta trae `max_seconds_per_clip`, `recommended_clips` y `recommended_clip_duration_sec`, y el agente reparte la duración por tramos y monta con `studio_render_montage`.

### 8.5. Duración honesta y escenario real

Pedir 30 s en una sola llamada no puede devolver 30 s: el motor recorta a 161 fotogramas y el resultado sería más corto sin avisar. Por eso el rechazo es explícito y cuantificado, y el preset ya compila **6 clips de 5 s + 7 keyframes** para 30 s (12 + 13 para 60 s). El montaje final se mide con `probe_media_info`; ninguna cifra de duración se comunica sin medirla.

El escenario también se respeta: el plan detecta el entorno pedido (`parque`, `playa`, `interior`, o `calle` por defecto) y describe árboles/setos/banco/fuente, arena/pasarela/olas o pasillo/muebles/lámparas en lugar de forzar acera y escaparates a toda escena. Los prompts visuales de imagen y vídeo llevan ese entorno; las instrucciones meta («el vídeo debe durar 30 segundos») se eliminan del prompt visual y viven solo en los parámetros del plan.

### 8.6. Auditoría de una instancia desplegada

`studio_get_state_and_manual` publica `build` (`commit`, `commit_short`, `commit_date`, `branch`, `dirty`, `source`), leído de solo lectura del checkout. Eso permite distinguir un despliegue nuevo de uno viejo sin adivinar por el número de herramientas.

`python3 tools/mcp_audit.py --url <túnel> --expect-commit <sha>` verifica, sin generar vídeo ni tocar pesos: handshake JSON-RPC, contrato de herramientas, commit real, preferencias, plan de 30 s (6 clips / 7 keyframes y entorno pedido), guardas `DURATION_EXCEEDS_SINGLE_CLIP` / `SPEED_LORA_MISSING`, contenido de `/mcp/guia` y que ninguna respuesta de herramienta reexponga el token. Devuelve un informe markdown y código de salida distinto de cero si algo falla.

Máximo dos reintentos del mismo trabajo; al segundo, `studio_export_session_log(format="md")` y un informe breve. Un trabajo `running` con progreso no es un fallo: se espera el mismo `job_id`.

## 9. API web equivalente y pruebas reproducibles

- `GET /api/catalog/hf-search?query=Wan&category=video&gpu_profile=auto&only_compatible=true&limit=12&offset=0`
- `GET /api/memory/status` (vigilancia, presión, reserva y modelos protegidos)
- `GET /api/runtime/status` (si el proceso tuvo que recargar frameworks por los parches de compatibilidad, y `build` con el commit real)
- `GET /api/preferences` / `POST /api/preferences` (las mismas preferencias que guarda el agente)
- `GET /api/mcp/info` devuelve el token **solo** a la UI del propio host (`X-Studio-UI: 1` + `Origin` coincidente) o a quien ya lo presenta; un tercero recibe `token_exposed: false` y los comandos ocultos. El token se regenera en cada arranque.
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

### Prueba experimental LTX 2.3 GGUF en la pestaña Vídeo

«LTX 2.3 GGUF Q4» aparece en el selector y las tarjetas de Vídeo al desmarcar el filtro de encaje estimado. Utiliza ComfyUI y ComfyUI-GGUF aislados a commits fijos. La preparación instala el entorno y verifica cinco archivos por revisión y SHA-256 (~24,1 GiB). Descargas y hashes muestran bytes medidos; la carga sin señal de progreso se muestra como indeterminada, y la difusión indica pasos reales cuando Comfy los comunica. El motor se mantiene activo y conserva su caché entre clips mientras haya memoria suficiente; el gestor lo libera al pulsar «Sacar de RAM/VRAM», por presión de memoria o al finalizar el servidor. Comfy puede mover o liberar componentes por su propia política; no se garantiza que todos permanezcan en VRAM.

Admite prompt, prompt negativo, semilla, primer fotograma opcional, resolución (múltiplos de 32, mínimo 256 y área máxima 768×512), duración experimental de 1 a 20 s y 8–30 fps. El número de frames se redondea al 8n+1 más cercano, por lo que la duración puede variar ligeramente (10 s a 24 fps = 241 frames; 20 s a 24 fps = 481). Por encima de 10 s no se ha validado en GPU: el consumo crece, el control de memoria puede rechazar la solicitud y sigue existiendo riesgo de fallo. Bajar resolución o fps puede reducir el cálculo, pero no asegura que quepa. El flujo destilado fija 8 pasos y CFG 1; no admite LoRA, fotograma final ni referencia de voz. Genera audio y vídeo conjuntamente, pero no garantiza identidad de voz entre clips. La opción experimental de diálogo en español añade una instrucción al prompt, no es un selector de idioma ni un motor de voz. La fidelidad opcional del primer fotograma se pasa al nodo I2V; reducirla puede modificar la apariencia y no garantiza movimiento. Comprueba las licencias LTX y Gemma.

Un registro de Colab L4 muestra dos clips A/V de 241 frames generados correctamente y reutilización de caché en el segundo. No se han medido sus picos ni validado clips de más de 10 s u otras resoluciones en L4. Las precomprobaciones (32 GiB RAM y 14 GiB VRAM libres más reserva, con margen adicional preventivo para mayor área/duración) son estimaciones, no garantías ni medidas de pico; la vigilancia puede detener la tarea, pero una reserva nativa puede agotar memoria antes. No se borran pesos si falla. Si `venv` quedó incompleto por `ensurepip`, la preparación intenta repararlo sin borrar archivos y registra el error si no puede.

Fuentes: https://huggingface.co/unsloth/LTX-2.3-GGUF ; https://github.com/city96/ComfyUI-GGUF ; https://github.com/farazshaikh/LTX-2.3-Workflows (se adaptó el flujo público GGUF de una pasada, sin nodos externos de mejora ni segunda etapa).
