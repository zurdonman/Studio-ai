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
   - Soporte para **LoRA Extra / NSFW** opcional con interruptor y cualquier LoRA adicional de Hugging Face.
   - Procesamiento **FFmpeg BT.709** automático a la entrada y a la salida (`-map_metadata -1`, `yuvj444p`, `bt709`).
   - Registro automático de **procedencia** en cada imagen (modelo, LoRAs, semilla, pasos, tiempo, resolución).

2. **Catálogo Bajo Demanda + Control Separado de Disco y RAM/VRAM:**
   - Nada se descarga solo al iniciar.
   - **Imagen:** Qwen-Image-Edit 2511 (4-bit) + Lightning 4 pasos ★, LoRA Extra Qwen Edit Plus, + repositorios propios de Hugging Face.
   - **Vídeo:** `Wan 2.2 TI2V-5B` ★, `Wan 2.2 S2V-14B` (guiado por voz/diálogo), `LTX-Video 2B` (rápido para T4/L4) + **Animación FFmpeg en CPU (0 VRAM)**.
   - **Voz (TTS):** `Edge-TTS` instantáneo sin descarga (`Álvaro`, `Elvira`, etc.), `Kokoro-82M` y `Qwen3-TTS 1.7B`.
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

## Colab Python 3.13: actualizar sin perder los modelos

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

### Validación de desarrollo

```bash
node --check studio/static/app.js
python -m pytest -q
```

Las regresiones de parámetros, offload y concurrencia usan dobles pequeños sin descargar modelos.
Estas pruebas no sustituyen una ejecución real de inferencia en la GPU L4 de Colab.

---

## Licencia

MIT
