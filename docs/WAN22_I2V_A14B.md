# Wan 2.2 I2V A14B · experimental

La sección de Vídeo incluye una receta local **imagen a vídeo**, independiente de Wan Animate/transferencia de movimiento. Usa ambos expertos de Wan 2.2 (high-noise y low-noise), UMT5 y VAE de Wan 2.1. El gráfico usa Euler/simple, shift 5, CFG configurable y un cambio de experto a mitad de los pasos, como en el flujo Comfy de dos etapas. La receta no aplica LoRA de destilación: el valor inicial es 40 pasos y CFG 3,5. Produce un vídeo mudo a 16 fps. La entrada se encaja sin recortar, con posibles márgenes grises. Hasta 81 fotogramas (`4n+1`); para 720p el ancho/alto deben ser múltiplos de 32 (p. ej., 1280×704). Mayor tamaño y duración requieren más memoria y tiempo.

La primera ejecución, tras autorización, instala el runtime Comfy aislado si falta, descarga solo la variante elegida y verifica tamaño y SHA-256 de los dos expertos y los componentes. Se conservan los archivos en disco y se supervisa el proceso; se para ante presión de memoria, pero no se garantiza evitar OOM. El cálculo de memoria es preventivo, no medido y puede rechazar una sesión. No se cambian automáticamente de variante ni se descargan alternativas. Al terminar, Studio intenta liberar los pesos y mantener activo el proceso Comfy. Si no se puede confirmar una limpieza segura, cierra el proceso como medida de protección. Los archivos descargados permanecen en disco.

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

## Wan 2.2 Fun A14B InP: texto, imagen e inicio–final

En la pestaña **Vídeo**, abre **Wan 2.2 A14B**, elige el modelo **Fun InP A14B**, la modalidad y la cuantización. La misma pareja de expertos de Fun sirve para:

- **Texto a vídeo:** solo prompt; no se usan imágenes.
- **Imagen a vídeo:** prompt y fotograma inicial.
- **Inicio → final:** prompt y dos imágenes explícitas; el modelo intenta generar la transición, pero no garantiza reproducir fielmente ambos fotogramas.

El grafo usa el nodo Comfy `WanFunInpaintToVideo`, que prepara latentes y máscaras para cero, una o dos imágenes. Las imágenes se encajan sin recortar y pueden aparecer márgenes; la salida es muda a 16 fps. La receta original **I2V A14B** sigue separada y solo admite imagen inicial. No son pesos intercambiables: se descargan high y low de Fun para cada cuantización, más UMT5 y VAE compartidos. No se descarga ni cambia de modelo automáticamente entre modos.

Fun admite **FP8 scaled** de Comfy-Org y **GGUF QuantStack** Q2_K, Q3_K_S/M, Q4_0/1, Q4_K_S/M, Q5_0/1, Q5_K_S/M, Q6_K y Q8_0. FP8 son aproximadamente 35,6 GB y Q4_K_M 26,3 GB en archivos (GB decimales), sin incluir entorno, temporales ni cachés; no son requisitos de VRAM. NVFP4 de la receta I2V **no está disponible para Fun**. Los archivos se descargan con revisión fijada, tamaño y SHA-256; no se incluyen en Studio. La selección requiere autorizar descarga.

**Experimental**: se ha validado el esquema del grafo con Comfy y archivos ficticios, además de tests simulados, pero **no instalación/inferencia con pesos reales ni GPU/L4**. Puede fallar o superar la memoria disponible; los tiempos y estimaciones no son garantía. Se recomiendan primero 512×320 y clips cortos. Consultar [Wan2.2-Fun-A14B-InP y licencia Apache 2.0](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP), [GGUF QuantStack](https://huggingface.co/QuantStack/Wan2.2-Fun-A14B-InP-GGUF), [FP8 Comfy](https://huggingface.co/Comfy-Org/Wan-2.2) y licencias de dependencias.

### LightX2V experimental (solo Fun FP8 inicio → final)

La receta de Comfy documentada usa dos adaptadores LightX2V distintos, high y low,
con sus respectivos expertos: 4 pasos, CFG 1, shift 8, Euler/simple. En la interfaz
elige Fun InP, FP8 e inicio → final para activarla. Añade aproximadamente 2,45 GB
al total de descarga; los pesos se conservan en disco. Esta receta no es para
Wan 2.1 Fun ni para las cuantizaciones GGUF. Su ejecución y consumo reales no se
han probado aquí con GPU; puede reducir el dinamismo.
Referencia: https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp
