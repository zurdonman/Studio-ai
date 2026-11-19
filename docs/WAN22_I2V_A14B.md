# Wan 2.2 I2V A14B · experimental

La sección de Vídeo incluye una receta local **imagen a vídeo**, independiente de Wan Animate/transferencia de movimiento. Usa ambos expertos de Wan 2.2 (high-noise y low-noise), UMT5 y VAE de Wan 2.1. El gráfico usa Euler/simple, shift 5, CFG configurable y un cambio de experto a mitad de los pasos, como en el flujo Comfy de dos etapas. La receta no aplica LoRA de destilación: el valor inicial es 40 pasos y CFG 3,5. Produce un vídeo mudo a 16 fps. La entrada se encaja sin recortar, con posibles márgenes grises. Hasta 81 fotogramas (`4n+1`); para 720p el ancho/alto deben ser múltiplos de 32 (p. ej., 1280×704). Mayor tamaño y duración requieren más memoria y tiempo.

La primera ejecución, tras autorización, instala el runtime Comfy aislado si falta, descarga solo la variante elegida y verifica tamaño y SHA-256 de los dos expertos y los componentes. Se conservan los archivos en disco y se supervisa el proceso; se para ante presión de memoria, pero no se garantiza evitar OOM. El cálculo de memoria es preventivo, no medido y puede rechazar una sesión. No se cambian automáticamente de variante ni se descargan alternativas. Comfy se cierra al terminar esta receta.

| Variante | Archivos de ambos expertos + UMT5/VAE (GB decimales aprox.) | Formato |
|---|---:|---|
| Q2_K | 17,6 | GGUF QuantStack |
| Q3_K_S | 20,0 | GGUF QuantStack |
| Q3_K_M | 21,3 | GGUF QuantStack |
| Q4_K_S | 24,5 | GGUF QuantStack |
| Q4_K_M | 26,3 | GGUF QuantStack |
| Q5_K_S | 27,3 | GGUF QuantStack |
| Q5_0 | 27,6 | GGUF QuantStack |
| Q5_K_M | 28,6 | GGUF QuantStack |
| Q5_1 | 29,0 | GGUF QuantStack |
| Q6_K | 31,0 | GGUF QuantStack |
| Q8_0 | 37,8 | GGUF QuantStack |
| FP8 scaled | 35,6 | safetensors Comfy-Org |
| NVFP4 mixto | 28,1 | safetensors comunitarios GitMylo, parte de capas FP8 |

Los tamaños de archivos no equivalen a VRAM/RAM ni incluyen entorno y temporales. FP4 y GGUF Q4 son formatos distintos. NVIDIA L4 no tiene cómputo FP4 nativo: NVFP4 puede necesitar conversión/emulación, ser lento o fallar. **Ninguna variante, combinación de pesos, resolución ni rendimiento ha sido validada en GPU o L4 en esta integración**; se ha comprobado la aceptación del grafo por el Comfy fijado usando archivos ficticios, no inferencia. No se promete que A14B sea mejor para todas las entradas: compara misma imagen, prompt, semilla y ajustes, y evalúa visualmente.

Fuentes y términos: [Wan-AI](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) (Apache 2.0), [GGUF QuantStack](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF), [FP8 Comfy-Org](https://huggingface.co/Comfy-Org/Wan-2.2) y [NVFP4 mixto GitMylo](https://huggingface.co/GitMylo/Wan_2.2_nvfp4). Revisa también las licencias de todos los componentes. El gráfico sigue la estructura de la [plantilla oficial I2V](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_14B_i2v.json), sin las LoRA de aceleración de esa plantilla.
