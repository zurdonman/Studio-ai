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

2. **Guardia de RAM real**: se liberan los slots anteriores sin tocar el disco y se vuelve a medir la RAM antes de descargar/cargar. Si no hay presupuesto, se devuelve `HOST_RAM_BUDGET_EXCEEDED` antes de reservar los tensores. Es una estimación preventiva, no un límite del sistema operativo ni una garantía contra OOM de inferencia o combinaciones Multi-LoRA.
3. **Inspector semántico**: usa la API real del catálogo y solo archivos locales. Un fallo de cuantización no provoca otra carga implícita en BF16. El fallback OpenCV queda explícitamente limitado; si se pidió auditoría semántica y no se pudo ejecutar, `all_consistent=false`, `identity_verified=false` y `ready_for_video=false`. En `fast_cv` explícito, `all_consistent` sigue siendo solo una evaluación fotométrica y no habilita la puerta semántica del preset. Se publican el alcance y el umbral efectivo.
4. **Registros independientes del servidor**: el proceso padre captura stdout/stderr, avisos nativos, salida de cloudflared, RSS/pico RSS, RAM del sistema y eventos de memoria cgroup v2 si están disponibles. Registra código de salida/señal, y no llama «limpia» a una caída. `PYTHONFAULTHANDLER=1` permite recoger trazas de fallos nativos cuando Python puede emitirlas; **SIGKILL no se puede capturar**.
5. **Puente MCP**: los fallos transitorios de lectura se reintentan de forma acotada sobre el mismo `job_id`. Nunca se reenvía automáticamente el POST de generación. Si se pierde el enlace, conserva el ID y devuelve `unknown_not_failed`, `do_not_resubmit=true` y una llamada de reanudación. Si el POST perdió su respuesta, su resultado es desconocido: consultar trabajos antes de volver a enviar.

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
3. Liberar los slots de imagen/visión sin borrar archivos; preparar **solo el Wan ya elegido**. Observar RAM, carga FP8 y telemetría. No comenzar con varios vídeos en paralelo.
4. Generar un único clip `flf2v` con `image_rel`, `end_image_rel`, `duration_sec` y defaults de la receta. `frame_a_rel`, `frame_b_rel`, `clip_duration_sec` y `motion_vectors` no son los nombres del esquema MCP; la cámara/locomoción/paralaje se describen en `prompt`.
5. Para ~30 s se necesitan aproximadamente seis tramos de 5 s y siete keyframes, ajustando duración real por frames/fps y montaje; tres keyframes solo dan dos transiciones. Revisar el número de clips del preset antes de ejecutarlo.

Las regresiones de CPU verifican contratos, redacción, salida del hijo, recuperación y ausencia de reenvíos/descargas. **La inferencia Wan 14B con el cargador corregido sigue pendiente de validación en la L4 real.**
