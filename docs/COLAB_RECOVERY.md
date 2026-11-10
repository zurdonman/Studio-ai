# Colab: recuperar un fallo de carga sin perder los pesos

## Lo que demuestra el registro recibido

- Las dos primeras generaciones de imagen terminaron. El adjunto del agente también registra fotogramas posteriores con Lightning + Next Scene + Skin Realism + AnyLight completados. **Ese registro no muestra una caída al activar esos LoRAs.**
- La pérdida de conexión se produce tras descargar Wan VACE 14B FP8 (27,23 GiB según ese inventario) y empezar su carga. Se recibió un 502 al consultar el trabajo y después un 530/1033 del túnel.
- **502 no equivale a CUDA OOM; 530/1033 no demuestra por sí solo que se haya destruido la VM.** También puede morir solo el servidor y el lanzador cerrar después el túnel. El lanzador anterior no imprimía el código de salida y terminaba diciendo «detenido limpiamente», incluso ante una salida anormal.
- El adjunto menciona `CatalogManager.local_artifact_dir`: ese método no existía. El auditor no llegó al VLM y devolvió métricas OpenCV. Un PASS de color/luz con umbral relajado **no verifica rostros ni identidad**, aunque su respuesta anterior lo sugiriera.

## Correcciones incluidas

1. **Wan FP8**: carga local en CPU con `torch_dtype=float8_e4m3fn` y `low_cpu_mem_usage=True`. Nunca se deja el dtype a `None` ni se reintenta el 14B entero en BF16/FP32. Se conserva el tratamiento de escalas, embeddings posicionales FLF2V y layerwise casting; se restauran desde el checkpoint los tensores originalmente de mayor precisión de los módulos excluidos del casting.

   El cargador upstream usa el dtype de los parámetros meta cuando no se fija uno, normalmente FP32. Un transformer de unos 14.000 millones de parámetros puede ocupar aproximadamente **52 GiB solo en FP32**, sin contar el checkpoint, encoder, VAE o adaptadores; la sesión informaba 52,96 GiB de RAM total. Es una ruta de agotamiento de RAM identificada en el código, **no una confirmación retrospectiva de OOM sin exit code/telemetría**.

   Referencias de implementación revisadas: [single_file_model.py](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/loaders/single_file_model.py) y [model_loading_utils.py](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/model_loading_utils.py). No es un benchmark de esta GPU.

2. **Gestión automática de RAM/VRAM**: se vigila continuamente la memoria, se liberan modelos inactivos sin tocar el disco y se vuelve a medir antes de descargar/cargar. La admisión suma base + todos los adaptadores + reserva; si no hay presupuesto, se devuelve `MEMORY_BUDGET_EXCEEDED` antes de reservar los tensores. Los puntos seguros pueden detener cooperativamente una carga/inferencia con `MEMORY_SAFETY_STOP`. Es una protección preventiva, no un límite del sistema operativo ni una garantía de cero OOM. [Política y límites](MEMORY_MANAGEMENT.md).
3. **Inspector semántico**: usa la API real del catálogo y solo archivos locales. Un fallo de cuantización no provoca otra carga implícita en BF16. El fallback OpenCV queda explícitamente limitado; si se pidió auditoría semántica y no se pudo ejecutar, `all_consistent=false`, `identity_verified=false` y `ready_for_video=false`. En `fast_cv` explícito, `all_consistent` sigue siendo solo una evaluación fotométrica y no habilita la puerta semántica del preset. Se publican el alcance y el umbral efectivo.
4. **Registros independientes del servidor**: el proceso padre captura stdout/stderr, avisos nativos, salida de cloudflared, RSS/pico RSS, RAM del sistema y eventos de memoria cgroup v2 si están disponibles. Registra código de salida/señal, y no llama «limpia» a una caída. `PYTHONFAULTHANDLER=1` permite recoger trazas de fallos nativos cuando Python puede emitirlas; **SIGKILL no se puede capturar**.
5. **Puente MCP**: los fallos transitorios de lectura se reintentan de forma acotada sobre el mismo `job_id`. Nunca se reenvía automáticamente el POST de generación. Si se pierde el enlace, conserva el ID y devuelve `unknown_not_failed`, `do_not_resubmit=true` y una llamada de reanudación. Si el POST perdió su respuesta, su resultado es desconocido: consultar trabajos antes de volver a enviar.

## Fallos de adaptadores y del inspector 4-bit (segunda sesión de producción)

Sesión distinta de la primera caída: aquí el modelo de vídeo **sí** cargó (Wan VACE 14B FP8, 0,52 GB en reposo, RAM 5,39/52,96 GiB) y Wan llegó a prepararse. Los errores son de carga de adaptadores y de cuantización, no de RAM.

Síntomas observados en una sesión real de L4 y su causa en el código, no en la escena:

1. **`ValueError: 'state_dict' should be empty at this point ... '.0.bias', 'ing_pose.0.weight'`** al acoplar FusionX + UniAnimate + MoviiGen. El checkpoint trae claves que el conversor de Diffusers no consume (un prefijo truncado por el export y pares sin nombre de módulo). Antes se abortaba **todo el stack**; ahora esas claves se informan y se descartan, y el resto del LoRA se aplica. No se borra ni se reescribe ningún peso en disco.
2. **`Adapter name vid_speed_lora already in use`** en el reintento. Tras el fallo anterior quedaba un adaptador vivo que el handle ya no describía. Ahora la carga es transaccional: si un adaptador falla, se retiran todos y el modelo base queda intacto; `studio_get_work_state` muestra `adapters_partial_failure`.
3. **`AssertionError` en `bitsandbytes/nn/modules.py` (`module.weight.shape[1] == 1`)** al ejecutar el inspector Qwen2.5-VL 4-bit. Los parches de compatibilidad se aplicaban después de crear las clases 4-bit, así que la reconstrucción del parámetro no tenía guarda. Ahora los parches se instalan antes de construir el modelo, la instancia parcial se recarga desde los archivos locales y, si el proceso arrastra clases mezcladas, el resultado es `RUNTIME_PATCHES_STALE` (reiniciar servidor) en lugar de un `AssertionError` dentro de la inferencia. La auditoría queda limitada a fotometría (`identity_verified=false`) y nunca autoriza vídeo.

Los tres casos están cubiertos por pruebas de CPU. **No son una validación GPU**: no se han ejecutado en L4 dentro de este entorno.

## A. Rescatar lo que queda en la sesión anterior

**No pulses «Liberar GPU», «Desconectar y eliminar entorno» ni borres `/content/studio_data`.** Si el kernel todavía responde, el disco puede seguir intacto aunque la web no abra.

En una celda nueva puedes descargar el log que ya guardaba la versión anterior, sin contactar con la aplicación:

```python
from pathlib import Path
import re
from google.colab import files

source = Path('/content/studio_data/state/session_console.log')
if source.exists():
    text = source.read_text(encoding='utf-8', errors='replace')
    text = re.sub(r'\b(?:st_mcp_[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{8,})\b', '[REDACTED]', text)
    text = re.sub(r'(?i)([?&]token=)[^\s&]+', r'\1[REDACTED]', text)
    destination = Path('/content/studio_rescate.log')
    destination.write_text(text, encoding='utf-8')
    files.download(str(destination))
else:
    print('No está el log en esta VM. Comprueba si Colab creó un entorno nuevo.')
```

Comprueba en el explorador de archivos `/content/studio_data/outputs/images` y `models` antes de regenerar o descargar nada. El ZIP de diagnóstico nuevo **no incluye imágenes/vídeos**; exporta también tus resultados por separado. Si Colab ya destruyó la VM, ni este parche ni reiniciar el proceso pueden recuperar su disco efímero. Solo una copia descargada o en almacenamiento persistente lo preserva.

## B. Actualizar y arrancar sin borrar modelos

1. Detener la celda de arranque, no eliminar el entorno. Con el lanzador anterior puede ser necesario pulsar detener una segunda vez si seguía ocupado.
2. Actualizar el checkout existente:

   ```python
   import subprocess
   subprocess.check_call(['git', '-C', '/content/studio', 'pull', '--ff-only',
                          'origin', 'arena/f0997b49-studio-ai'])
   print(subprocess.check_output(['git', '-C', '/content/studio', 'rev-parse', '--short', 'HEAD'], text=True))
   ```

   No cambiar a otra rama ni usar `reset --hard`, clonar encima, borrar datos o reiniciar toda la VM como remedio automático. Si hay conflicto o cambios locales, detenerse y revisarlos.
3. Ejecutar de nuevo la celda de arranque del notebook actualizado. Recarga el módulo del lanzador y abre el servidor en un proceso nuevo. Los pesos verificados se reutilizan; no hay que redescargar Qwen/Wan por actualizar código.
4. Usar **la URL y guía nuevas** que imprime la celda. Si un token ha quedado expuesto en un log compartido, no reutilizarlo: eliminar un `STUDIO_MCP_TOKEN` fijado manualmente o sustituirlo de forma privada antes del arranque. No publicar credenciales en diagnósticos.
5. Descargar otra vez `studio_mcp.py` desde la guía nueva si se usa un puente local: la corrección de reanudación también está en ese archivo, no solo en el servidor.

## C. Obtener el diagnóstico con el parche instalado

- Web activa: **🩺 Diagnóstico**, o `GET /api/logs/diagnostics`.
- Web/servidor/túnel caídos, kernel todavía activo: usar la **celda de rescate al final del notebook**. Ejecuta un recolector estándar, sin Torch, CUDA ni acceso a red:

  ```python
  import subprocess, sys
  from google.colab import files
  path = subprocess.check_output(
      [sys.executable, '-m', 'studio.diagnostics', '--data-dir', '/content/studio_data'],
      cwd='/content/studio', text=True).strip()
  files.download(path)
  ```

Los archivos de cada arranque están en `state/diagnostics/FECHA_ID/`: `server.log`, `tunnel.log`, `events.jsonl`, `telemetry.jsonl` y `status.json`. Se rota cada flujo a 8 MiB más una copia anterior; el exportador limita a 32 MiB los logs y puede omitir los más antiguos. El ZIP añade versiones, revisión Git, estado GPU al recogerlo e inventario de salidas, **sin pesos ni contenido multimedia**. Los tokens comunes se ocultan tanto al escribir como al exportar. Revisa también cualquier texto privado de tus prompts antes de compartirlo.

- `SIGKILL` / `-9`: posible OOM de RAM o señal externa. Un aumento de `oom_kill` aporta evidencia del **cgroup**, no identifica por sí solo al proceso víctima.
- `SIGSEGV` / `-11`: fallo nativo; revisar traza y versiones.
- Traceback Python / CUDA OOM: revisar mensaje, recursos y parámetros; no convertir todo error en falta de VRAM.
- Túnel terminado pero servidor activo: conservar el trabajo, revisar `tunnel.log` y el enlace actual. No volver a generar por un 502.

Estos logs sobreviven a una caída del proceso **mientras exista la misma VM/disco**. No sobreviven a la destrucción de `/content`. No se reinicia automáticamente el modelo ni se descarga una variante de respaldo.

## D. Reanudar la prueba de vídeo de forma controlada

1. Consultar estado y trabajos: no repetir los fotogramas ya existentes ni el envío pendiente sin verificar su resultado.
2. Repetir la auditoría con `mode='vlm_qwen'`, ancla y referencias correctas. Los fotogramas siguientes deben usar `base_image_rel`/`reference_rels`, no solo el mismo prompt. Un PASS OpenCV no sustituye esto.
3. Preparar **solo el Wan ya elegido** y los adaptadores exactos. La [gestión automática de memoria](MEMORY_MANAGEMENT.md) libera los modelos inactivos y exige margen de RAM/VRAM antes de descargar/cargar. Observar `memory_guardian`, RAM y carga FP8; si se rechaza el presupuesto, no repetir ni descargar alternativas. No comenzar con varios vídeos en paralelo.
4. Generar un único clip `flf2v` con `image_rel`, `end_image_rel`, `duration_sec` y defaults de la receta. `frame_a_rel`, `frame_b_rel`, `clip_duration_sec` y `motion_vectors` no son los nombres del esquema MCP; la cámara/locomoción/paralaje se describen en `prompt`.
5. Para ~30 s se necesitan aproximadamente seis tramos de 5 s y siete keyframes, ajustando duración real por frames/fps y montaje; tres keyframes solo dan dos transiciones. Revisar el número de clips del preset antes de ejecutarlo.

Las regresiones de CPU verifican contratos, redacción, salida del hijo, recuperación y ausencia de reenvíos/descargas. **La inferencia Wan 14B con el cargador corregido sigue pendiente de validación en la L4 real.**

## `/api/mcp/info` filtraba el token MCP (corregido)

**Síntoma:** cualquier persona con la URL pública del túnel (`*.trycloudflare.com`, sin token) podía abrir
`/api/mcp/info` y recibir el token en claro *y* los comandos listos para usarlo. Con ese token se pueden lanzar
herramientas MCP: descargas de pesos, renders de vídeo, exportaciones. Además, `/api/mcp/tunnel-info` permitía
fijar la URL pública sin credenciales, porque Cloudflare llega siempre como `127.0.0.1` y la comprobación
"es local" daba por bueno a cualquiera.

**Corrección:**

- El token solo se entrega si la petición trae `X-Studio-UI: 1` (lo pone el panel) **y** su `Origin`/`Referer`
  coincide con el host público, o si quien pregunta ya presenta el token. Un tercero recibe `token_exposed: false`
  con los comandos ocultos, así que el panel sigue funcionando dentro de la app.
- La IP del cliente ya no se usa como prueba de confianza: detrás de túnel siempre es `127.0.0.1`.
- El token se **regenera en cada arranque** de `launch_colab.py`, de modo que uno filtrado queda inutilizado al reiniciar.
  El token solo aparece en la celda de Colab, que es el canal privado del usuario.

**Verificación:** `tests/test_production_reliability.py::test_public_mcp_info_never_hands_the_token_to_a_stranger`
cubre anónimo, UI del mismo host, `Origin` ajeno (CSRF), petición forzada sin cabecera de UI y cliente que ya
presenta el token. Si tu instancia es anterior a este commit, el token que muestra la celda sigue siendo válido para
cualquiera que haya visto la URL: reinicia para obtener uno nuevo.

## Clip a 20 s/paso y LoRAs rápidos que no se cargan (sesión del 10-N)

Tres causas distintas, las tres en el log `studio_ia_log_20261110_005200.md`.

**1. CPU offload en una GPU sobrada.** Wan VACE 1.3B + su LoRA rondan 2,7 GB y la L4 tenía 21,8 GB
libres, pero se activó `enable_model_cpu_offload()` y el clip se fue a **20,6 s por paso** (739 s para
5 s de vídeo). El umbral comparaba la VRAM **total** con `min_vram_gb` de la *receta* (8 GB), que es el
mínimo exigido, no lo que el modelo ocupa. Ahora se suman los pesos reales del pipeline (`transformer`,
`vae`, `text_encoder`) más una reserva de activaciones y solo se hace offload si no caben; si no cabe,
el aviso dice cuántos GB son y por qué.

**2. El LoRA rápido abortaba entero.** `ValueError: Invalid LoRA checkpoint. Make sure all LoRA param
names contain 'lora' substring.` Diffusers recorre el dict y exige que **toda** clave contenga `lora`;
un `.alpha` huérfano del export vale para satisfacer esa comprobación y aun así tumba el adaptador.
Ahora, ante ese error concreto, se reintenta una vez sin los `.alpha` huérfanos (el alpha ya está plegado
en el `lora_down`). El clip anterior se generó con `active_lora_id: null`: 30 pasos a CFG 5 en vez de los
6-8 pasos destilados.

**3. El mismo clip se lanzó dos veces.** Los jobs 18 y 19 usaban los mismos fotogramas y el mismo prompt:
el segundo esperó en cola 12 minutos para repetir trabajo idéntico. `studio_generate_video` ahora calcula
una clave de deduplicación (modelo, LoRA, prompt, fotogramas, duración, fps y pasos) y devuelve el trabajo
que ya está corriendo en vez de encolar otro.

**4. La interfaz no mostraba lo que generaba el agente.** `pollStatus` refrescaba hardware, trabajos y
compatibilidad, pero nunca volvía a pedir `/api/assets`: las imágenes y los clips generados por MCP no
aparecían en las galerías hasta recargar la página. Ahora, cuando cambia el número o el estado del último
trabajo, se recargan los assets (con guarda para no solapar peticiones).

Y una decisión de capacidad: el 14B FP8 + 3 adaptadores necesita 44,47 GiB de RAM y la sesión tenía
48,43 libres con 5,30 de reserva, así que **no** entró (`MEMORY_BUDGET_EXCEEDED`). Ese rechazo es correcto:
bajar de 14B a 1.3B fue lo que dejó el clip en 739 s.

## Créditos y sesiones olvidadas

**El saldo no se puede leer.** Google Colab no publica el saldo de unidades por API. Lo que muestra la app
es una estimación: un `initial_credits` configurable (100 por defecto) menos `horas × tarifa fija`. Ese
número se guarda en `studio_data/state/credits_state.json` y **arrastra entre sesiones** con
`prior_spent_credits`, así que puede alejarse mucho del real. Ahora la respuesta lo dice explícitamente
(`is_estimate`, `source: estimacion_local_tarifa_fija`, `authoritative_source`) para que nadie lo confunda
con una factura. La fuente de verdad es la página de Colab.

**Apagado automático.** Avisar en la interfaz no sirve si el usuario está durmiendo. Con
`idle_autoshutdown_minutes` (45 por defecto, `0` lo desactiva) el servidor libera la GPU cuando pasan esos
minutos sin ningún trabajo activo, solo en Colab y solo con `active_jobs_count == 0`. Se puede ajustar con
`POST /api/credits/config`:

```json
{"initial_credits": 98.3, "idle_autoshutdown_minutes": 30}
```

Y para cerrar la pestaña sin dejar la GPU encendida, `POST /api/shutdown` (botón «Liberar GPU»).

## Canal de control cuando el agente no alcanza el túnel

El agente de Arena no puede conectarse a `trycloudflare.com` (ni a ningún otro túnel: su egreso solo
permite `api.github.com`, `pypi.org` y `registry.npmjs.org`), así que no puede lanzar trabajos por MCP
contra la instancia. La vía que sí funciona invierte la dirección:

    agente --(API de GitHub)--> buzón privado (gist) <--(sondeo)--> app --(MCP local)--> trabajos

**Se enciende desde la aplicación.** Panel MCP → «🔗 Canal para agente arena.ai» → pegar el token de
GitHub → «Encender canal». No hay comandos que copiar a la celda: la app crea el buzón, lanza el bucle
en un hilo y muestra el estado (sondeos, último verbo, último error) y el botón «Copiar para el agente»,
que entrega **solo el identificador del gist**, nunca el token.

Por qué gist privado y no el repositorio: el repo puede ser privado y no hace falta publicar órdenes ni
resultados. El agente solo necesita `https://api.github.com/gists/<id>`. El backend `repo`
(`channel/ordenes.json` + `state/channel/resultados.json`) sigue disponible para repos públicos con
`--backend repo`.

El canal habla con el servidor local por `http://127.0.0.1:7860/mcp` **a propósito**: importar
`studio.mcp_server` en su propio proceso crearía una segunda cola de trabajos invisible para la
interfaz y duplicaría el modelo en RAM. Todo lo que ejecute aparece en la app.

Límites, deliberados: lista cerrada de verbos (nada de shell, descargas de modelos ni borrado de pesos),
argumentos con credenciales rechazados, cada orden se ejecuta una sola vez por `id`, y el token de GitHub
vive en `studio_data/state/github_channel.json` con permisos 600.
