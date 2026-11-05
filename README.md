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

## Licencia

MIT
