# ◆ Studio IA

Estudio web personal y libre para **generación/edición de imagen, vídeo, locución (TTS) y montaje final MP4**, diseñado para ejecutarse **dentro de Google Colab** (compatible con cualquier GPU de tu cuenta: **T4, L4, A100, H100** o modo CPU para pruebas y montaje).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zurdonman/Studio-ai/blob/arena/f0997b49-studio-ai/Studio_IA_Colab.ipynb)

---

## Arranque en 1 línea en Google Colab

Elige tu GPU en Colab (`Entorno de ejecución → Cambiar tipo de entorno de ejecución`), pega esta celda y pulsa **Play**:

```python
import os, subprocess; subprocess.run("rm -rf /content/studio && (git clone -b arena/f0997b49-studio-ai https://github.com/zurdonman/Studio-ai.git /content/studio || git clone https://github.com/zurdonman/Studio-ai.git /content/studio)", shell=True, check=True); os.chdir("/content/studio"); import launch_colab; launch_colab.main()
```

> **Por qué se ejecuta desde Python en la celda:** así el botón rojo **«⏻ Liberar GPU»** de la aplicación puede llamar a `google.colab.runtime.unassign()` y apagar la máquina de verdad para no gastar créditos.

---

## Qué incluye

1. **Imagen (Predeterminado: `Qwen-Image-Edit-2511 4-bit` + `LoRA Lightning 4 pasos`):**
   - Descarga y carga conjunta en un solo clic.
   - **Edición simple**, **Multi-Referencia (hasta 3 imágenes)**, **Inpainting pintando máscara con pincel** (con borde suavizado respetando el exterior al 100 %) y creación sobre lienzo limpio.
   - Soporte para **LoRA Extra / NSFW** opcional con interruptor y LoRAs con arquitectura/base revisadas. Las importaciones desconocidas son candidatos, no ejecutables.
   - Procesamiento **FFmpeg BT.709** automático a la entrada y a la salida (`-map_metadata -1`, `yuvj444p`, `bt709`).
   - Registro automático de **procedencia** en cada imagen (modelo, LoRAs, semilla, pasos, tiempo, resolución).

2. **Catálogo Bajo Demanda + Control Separado de Disco y RAM/VRAM:**
   - No se descargan pesos al iniciar. El índice puede refrescar metadatos públicos en segundo plano.
   - **Imagen:** Qwen-Image-Edit 2511 (4-bit) + Lightning 4 pasos ★, LoRA Extra Qwen Edit Plus, + repositorios propios de Hugging Face.
   - **Vídeo:** `Wan 2.2 TI2V-5B` ★, Wan 2.1 VACE/FLF2V FP8, LTX 13B destilado/LTX-2 AV según hardware y receta, `LTX-Video 2B` (rápido para T4/L4) + **Animación FFmpeg en CPU (0 VRAM)**.
   - **Voz (TTS):** `Edge-TTS` instantáneo sin descarga (`Álvaro`, `Elvira`, etc.), MMS español, `Kokoro-82M` (backend opcional) y Bark pequeño. Qwen3-TTS es un candidato investigado, no un motor integrado.
   - **Memoria autogestionada:** vigilancia continua de RAM/VRAM, márgenes preventivos y expulsión LRU de modelos inactivos. Protege inferencias activas y comprueba base + todos los LoRA antes de descargar/cargar; no borra pesos ni descarga alternativas como remedio. [Política, diagnóstico y límites](docs/MEMORY_MANAGEMENT.md).
   - Tres acciones independientes por modelo:
     - **Descargar de HF** con porcentaje real `0–100 %` y velocidad `MB/s` (comprueba antes si cabe en disco; si falta espacio te dice cuántos GB faltan y qué modelo puedes borrar).
     - **Cargar / Liberar de RAM y VRAM** sin borrar los archivos del disco.
     - **Borrar del disco** para recuperar GB al instante.

3. **Montaje Final con FFmpeg (funciona incluso sin GPU):**
   - Línea de tiempo visual para ordenar imágenes y clips, elegir duración y movimiento por plano, asociar locuciones por plano o música de fondo global y exportar el **MP4 H.264 BT.709** verificado.

4. **Control Total del Proceso y de tus Créditos:**
   - Consola en vivo paso a paso con hora exacta y **detector de posible cuelgue (>30 s sin actividad)**.
   - **Contador de créditos de Colab** ajustable en vivo según la GPU detectada + aviso de inactividad + descarga completa en `.zip` antes de apagar la sesión.

---

## Descubridor vivo y oráculo de modelos (2026-11-05)

- **Investigación trazable:** 73 familias/proyectos y 243 referencias a repositorios/componentes, con fuentes oficiales fijadas; no son 243 modelos garantizados. [Informe completo](docs/MODEL_RESEARCH.md) e [inventario JSON](studio/data/upstream_models.json).
- **74 recetas/complementos** en el catálogo, incluidas **8 recetas nuevas de imagen**: SDXL Base, FLUX Schnell, FLUX.2 Klein 4B, Z-Image Turbo, PixArt Sigma, SANA 600M/1600M y SANA Sprint. Lumina 2 queda como candidato: requiere integrar explícitamente su transformer/formato, no un cargador genérico. Se han implementado clases explícitas/defaults propios; **inferencia GPU de las nuevas recetas aún no validada**.
- **Buscador al escribir**, paginación, orden por descargas/actualización, inspector de clase/licencia/base/revisión y etiqueta de red/caché/offline. El mismo evaluador comprueba **motor, VRAM, RAM, disco, precisión y dependencias**, sin calcular compatibilidad solo por nombre/parámetros.
- **Actualización manual o automática** (6 h por defecto, configurable/desactivable). Solo metadatos públicos en `STUDIO_DATA_DIR/state/model_discovery.json`, con límites, caché, reintentos y reemplazo atómico. No instala librerías ni baja pesos/código remoto.
- **Receta revisada / candidato / backend no soportado**, sin promoción automática. Las importaciones manuales se conservan como candidatos no ejecutables. «Compatible» es encaje estimado; `runnable_now` es más estricto y una GPU simulada nunca es ejecutable.
- **Oráculo web y MCP** con motivos, alternativas, plantillas y entradas necesarias. Puede contestar «sin receta apta». No descarga modelos alternativos por timeout ni promete tiempos de generación.

Herramientas MCP: `studio_search_models`, `studio_inspect_model`, `studio_refresh_model_index`, `studio_model_index_status`, `studio_get_model_guide`, además de `studio_recommend_pipeline`. [Manual detallado, esquemas y ejemplos](docs/MCP_MODEL_ORACLE.md). En la web: **5. Catálogo → Descubridor / Oráculo**. La investigación/recetas se actualizan con código revisado; el refresco automático actualiza metadatos, no añade cargadores.

> En esta vista previa local no hay CUDA y la salida a Hugging Face está restringida: se muestran referencias locales/caché con su estado real. Buscar en línea y medir inferencia se debe comprobar en Colab con su GPU/RAM/disco detectados. Simular una L4 aquí no aumenta la RAM real.

---

## Colab Python 3.13: actualizar sin perder los modelos

**Si la aplicación se para al cargar Wan 14B o el túnel devuelve 502/530:** [rescate y diagnóstico sin servidor](docs/COLAB_RECOVERY.md). El lanzador conserva stderr/salida/señales/RAM, el notebook incluye una celda de rescate y el puente mantiene el `job_id`. Los logs locales no sobreviven si Colab elimina la VM.

Si una sesión anterior falló con `_is_hf_initialized`, `Cannot copy out of meta tensor`
o «ningún modelo de imagen cargado»:

1. **Detén la celda del lanzador**, actualiza el código y vuelve a ejecutar la celda de arranque.
   El lanzador reinstala las dependencias necesarias y abre el servidor en un **proceso Python nuevo**.
   No basta con refrescar la página web ni con actualizar el código mientras sigue vivo el servidor antiguo.
2. Conserva `/content/studio_data`: los pesos ya verificados se reutilizan. **No hace falta volver
   a descargar Qwen, Wan o LTX ni borrar el disco** para corregir estos errores de runtime.
   No pulses «Liberar GPU» ni desconectes/elimines el entorno si quieres conservar su disco temporal.
3. Reconecta el agente usando la nueva guía `/mcp/guia` que muestra el lanzador.

La compatibilidad conserva los metadatos privados de Transformers fuera del constructor de
`Parameter` / `Params4bit` / `Int8Params`, sin sustituir pesos ni usar `to_empty()` para ocultar
un error de tensores meta. Lightning y las LoRAs de imagen se montan **antes** del CPU offload.
Las preparaciones idénticas comparten tarea; las cargas, inferencias y liberaciones de GPU se serializan.

### Flujo MCP para un paseo coherente de un minuto

Pide primero `studio_get_workflow_preset` con
`preset_id="coherencia_personajes_entorno_avance"` y la descripción completa de la escena.
Si incluye «un minuto» o «60s» y no fijas otro número de tramos, compila **12 clips de unos 5 s
+ 13 keyframes**. El redondeo de fotogramas/FPS y la duración real de las voces pueden variar
ligeramente la duración final; mide y ajusta los diálogos antes del LipSync.

- Primero **todos los keyframes**, con las cuatro LoRAs precargadas desde el ancla
  (Next Scene y AnyLight inicialmente a escala `0.0`). Los fotogramas siguientes cambian
  adaptadores sin volver a cargar los pesos.
- Después **auditoría de identidad y avance**, reparando y reauditando antes del vídeo.
- Después **todos los clips `flf2v` con el mismo modelo**, y finalmente voces, ambiente urbano,
  LipSync y montaje sin fundidos entre extremos compartidos.
- Los prompts visuales no incluyen instrucciones de duración, creación de diálogos ni selección de LoRAs.
  El texto original se conserva por separado como guía para el guion; los diálogos de plantilla son ejemplos
  que el agente debe adaptar al tema del usuario si no recibió `dialogue_lines`.
- Un trabajo `running` / `queued` se espera con su **mismo `job_id`**. Un error de compatibilidad
  no autoriza descargar otros modelos pesados en cascada: el agente debe detenerse y comunicar el error.

### Preguntas de producción y reintentos (agentes pequeños)

El plan trae `production_preferences.pending_questions` y el agente pregunta **como máximo una vez**
(`studio_ask_user` → `studio_answer_preference`), guardando la respuesta en el servidor. LoRAs rápidos,
duración de clip, formato y voces dejan de depender de lo que el agente recuerde. Un fallo se consulta con
`studio_get_work_state` antes de reintentar, y un «generación rápida» sin `video_lora_id` rápido se rechaza con
`SPEED_LORA_MISSING` en lugar de ejecutarse lento sin avisar. Detalle en
[el manual del oráculo](docs/MCP_MODEL_ORACLE.md#83-preferencias-del-usuario-una-sola-pregunta-nunca-adivinada).

### Validación de desarrollo

```bash
node --check studio/static/app.js
python -m pytest -q
```

`177 passed` en CPU: simulaciones, dobles y un hijo real con RSS acotado, **ninguna inferencia GPU**. Las pruebas de
adaptadores y del inspector 4-bit reproducen las excepciones de Diffusers/bitsandbytes con dobles, no con Torch real.

Las regresiones de parámetros, offload y concurrencia usan dobles pequeños sin descargar modelos.
Estas pruebas no sustituyen una ejecución real de inferencia en la GPU L4 de Colab.

---

## Licencia

MIT
