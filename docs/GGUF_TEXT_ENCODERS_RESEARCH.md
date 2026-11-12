# Investigación: codificadores de texto GGUF para vídeo + audio

Consulta: 5 de noviembre de 2026. Alcance: LTX en Studio; **sin pruebas GPU realizadas por Studio**; ver el estado de integración al final. Un codificador cuantizado puede reducir el disco y parte de la presión de memoria, pero el tamaño del archivo no equivale al pico de RAM/VRAM ni determina por sí mismo la calidad del vídeo, diálogo o voz.

## LTX 2.3: Studio ya usa ambos en GGUF

El flujo A/V actual usa el DiT `ltx-2.3-22b-distilled-Q4_K_M.gguf` y, a la vez, el codificador `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` (7.432.229.248 bytes ≈ 6,92 GiB). Hay también un archivo de conectores de la variante destilada y dos VAEs en safetensors. El grafo usa `UnetLoaderGGUF` para el DiT y `DualCLIPLoaderGGUF` tipo `ltxv` para Gemma + conectores: véanse `studio/engines/ltx_gguf_manifest.py` y `studio/engines/ltx_gguf_workflow.py`. [Unsloth documenta el uso de Gemma 3 GGUF con LTX 2.3](https://huggingface.co/unsloth/LTX-2.3-GGUF). Este flujo, con sus pesos concretos, ya produjo I2V con pista de audio en una L4 del usuario; ello no evalúa la calidad perceptiva de la voz o el español.

## LTX 2.5: Gemma 4 GGUF con proyecciones

[elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF) distribuye versiones comunitarias que incluyen las dos proyecciones de LTX (audio y vídeo), no un Gemma genérico de chat. El repositorio exige acceder/aceptar los términos LTX 2.x; también se aplica la licencia de Gemma indicada en la ficha. No se descargaron archivos ni se verificaron independientemente los hashes; los valores siguientes proceden de esa ficha:

| Codificador | Bytes | GiB | SHA-256 declarado |
| --- | ---: | ---: | --- |
| Q2_K | 5.956.930.560 | 5,55 | `a70012eeea2fbee7c9c058fca5710842ec8cb9d9e94e630f2d1a0647a8d300f6` |
| Q4_K_M | 8.414.653.376 | 7,84 | `add54057106a531383711accfaf17e341e234cbe953fdafbc6cd6b309fb5e356` |
| Q5_K_M | 9.514.920.864 | 8,86 | `1d35d4fbfa34cca1513d8e9fdd77c0573778b21ffdcbe4ca9c906f37a8c502f9` |
| BF16 actual de Studio | 26.263.858.182 | 24,46 | `ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1` |

Q5 reduce el archivo del codificador ~15,60 GiB, Q4 ~16,62 GiB y Q2 ~18,91 GiB frente al BF16 actual. Solo como suma de tamaños de archivo, DiT de Studio LTX 2.5 UD-Q2 + encoder Q5 + VAEs conv/audio ocuparía unos **19,40 GiB**, frente a **35,00 GiB** con BF16; no incluye entorno ni temporales, ni establece un requisito de memoria. La cuantización más fuerte podría perjudicar el seguimiento del prompt; no se ha evaluado diálogo en español.

### Evidencia y obstáculo concreto

El autor informa de ejecuciones completas con audio y vídeo en RTX 4070 Ti SUPER de 16 GB, 32 GB de RAM y swap, usando **Q5**, su propio transformer GGUF y ComfyUI `2f35f4a08176d993cded35dac3332be4f7287f41`; Q4 y Q2 tienen pruebas de carga y forward del encoder, **no se les atribuye prueba completa GPU** en esa ficha. Esto no prueba la combinación con los DiT Unsloth de Studio, una L4 o la receta Studio de una pasada.

La ficha documenta ComfyUI-GGUF `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` **con un parche adicional** para reconocer la arquitectura `gemma4` y tratar tres parámetros BF16 del modelo A/V. La misma revisión sin parche es la fijada por Studio; su `loader.py` solo lista `gemma3` entre las arquitecturas Gemma. El grafo actual de LTX 2.5 usa `CLIPLoader` para safetensors BF16: **copiar un GGUF a la carpeta no lo convierte en compatible**, y renombrar/falsificar la arquitectura es incorrecto. El ComfyUI fijado de Studio (`7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe`) tampoco es el revisionado que declara el autor. La ficha publica el SHA-256 del parche; el contenido protegido y el parche no han sido inspeccionados ni probados por Studio.

Como alternativa no-GGUF existe un [Gemma 4 INT8 ConvRot para ComfyUI](https://huggingface.co/Lightricks/LTX-2.5), también con proyecciones, sujeto a acceso y licencia: puede requerir menos cambios en el cargador, pero es mayor que el GGUF y tampoco está validado en Studio/L4. Comprobar metadatos y dependencia real de su formato antes de usarlo.

### Propuesta segura de integración posterior

1. Mantener intactas las rutas LTX 2.3 ya probada y 2.5 BF16; nueva variante 2.5 optativa, sin marcar compatible con L4 por defecto.
2. Tras aceptar licencias con la cuenta del usuario, fijar revisión, tamaños y SHA-256 verificados de los archivos; no copiar credenciales al repositorio. Evaluar y fijar el parche de código y sus dependencias en un entorno aislado; no modificar el plugin compartido sin pruebas de regresión.
3. Usar `CLIPLoaderGGUF` tipo `ltxv` y garantizar carga completa del tokenizer y ambas proyecciones. Contrastar resultado del encoder frente a BF16; cargar datos reales y comprobar valores finitos, no solamente validar nombres de nodos.
4. Con pesos reales y GPU, probar T2V e I2V de clips cortos con audio, instrumentando picos RAM/VRAM, cancelación, descarga y cambios de modelo. Comparar prompt, movimiento, habla y español; comprobar que la salida contiene audio no sustituido.
5. Solo con datos medidos revisar preflight; no rebajar los 45 GiB preventivos actuales como simple resta de tamaño de archivo. Fijar por separado el resultado de L4 y el de otras GPU.

Primera candidata técnica sugerida: **encoder Q5**, por ser el único de esta ficha con ejecución A/V completa comunicada; Q4/Q2 pueden investigarse luego si la memoria lo requiere. No hay garantías de encaje en Colab L4; el estado de integración figura al final. No confundir un encoder de texto con ID-LoRA o referencia de voz: el encoder más pequeño por sí solo **no** conserva la voz entre clips.

## Estado de integración posterior

La combinación optativa **LTX 2.5 UD-Q2 + Gemma 4 GGUF Q5** se ha añadido al catálogo bajo «Vídeo» con un runtime Comfy independiente y sin sustituir las variantes anteriores. Se mantiene fuera del filtro estricto. Es una receta de una sola pasada, 8 pasos, sin upscale ni referencia de voz. El encoder Q5, su repositorio y revisión están fijados con SHA-256; el parche se descarga de ese mismo repositorio con la cuenta de Hugging Face configurada en Studio, se comprueba su SHA-256 y se aplica exclusivamente al plugin aislado tras comprobar su revisión y que no tenga modificaciones. No se fuerzan parches ni se alteran archivos de otros runtimes. Los archivos de DiT y VAEs se reutilizan con la variante BF16 cuando ya existen.

Antes de preparar, aceptar los términos de [elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF) con la cuenta de Hugging Face y configurar acceso a esa cuenta en Studio; no pegar credenciales en un log o conversación. Si no se autoriza el parche, falla antes de descargar los pesos de esta variante. Las licencias LTX y Gemma se siguen aplicando.

El presupuesto inicial de esta variante es **32 GiB de RAM libre y 14 GiB de VRAM libre**, más reservas y margen para dimensiones/duración mayores: es una estimación preventiva de prueba, **no un pico medido**, y puede detener la tarea o no evitar un OOM. En la primera prueba usar 512×320, 24 fps y 2 s. Preparar descarga y verifica los pesos, pero el encoder y el DiT se cargan realmente al generar. ComfyUI con archivos dummy aceptó los grafos T2V e I2V; pruebas automatizadas cubren rutas, selección y verificación de un parche simulado. **No hemos accedido al parche real protegido, cargado los pesos ni ejecutado inferencia GPU/L4.** La compatibilidad del parche real, el audio, el español y el movimiento necesitan prueba real y log. El valor de espacio en disco no es una estimación de memoria.

### Si falla la preparación del parche

Un fallo anterior `GGUF_GEMMA4_ACCESS` podía ocultar cualquier excepción de descarga, incluso una desconexión; **ese mensaje antiguo no demuestra que la licencia o el token estuvieran mal**. Los errores posteriores distinguen rechazo HTTP de acceso (401/403 o `GatedRepoError`), problema de red/caché, revisión/archivo no encontrado y causa desconocida, sin registrar el texto de la excepción ni el token. Si hay un rechazo, comprobar que el acceso del usuario a la página de elix3r está aprobado y que la cuenta correspondiente está configurada en el botón de Hugging Face de Studio; para problemas de red, revisar la conexión. Compartir únicamente el log sin credenciales.

El runtime Gemma Q5 fija Comfy `2f35f4a` (0.33.0), que no admite `--offline`. Studio utiliza `--disable-api-nodes`: en esa versión deshabilita los nodos API y el acceso de la interfaz a internet; en el otro Comfy fijado es alias de `--offline`. Se ha arrancado cada versión en CPU con el comando completo generado por Studio y comprobado `/system_stats`, los nodos necesarios y la ausencia de `GemmaAPITextEncode`. Esta prueba usa pesos vacíos y no prueba inferencia ni comportamiento en GPU.
