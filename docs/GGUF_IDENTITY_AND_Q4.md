# LTX GGUF: alternativas Q4 e investigación de identidad

## Qué se puede ejecutar

* LTX 2.3 destilado Q4_K_M ya existente: un usuario probó I2V en L4 con audio y vídeo. Mantiene 8 pasos y CFG 1; la imagen es el **primer fotograma**, no una referencia FaceID independiente.
* LTX 2.5 destilado UD-Q2_K_XL + Gemma 4 GGUF Q5_K_M: el usuario informa de una generación en L4. No tenemos los parámetros, vídeo ni medición de memoria de esa ejecución. Percibe peor calidad que con el Q4_K_M anterior; no podemos atribuir la diferencia a un componente.
* **Nuevos, sin prueba GPU:** LTX 2.5 destilado Q4_K_M + Gemma 4 Q5 y UD-Q4_K_XL + Gemma 4 Q5. Usan el mismo entorno/parche aislado, Gemma y VAE que el UD-Q2 + Q5; al cambiar se descarga el DiT que falte y se reutilizan los archivos comunes verificados. Requieren aceptar las condiciones del [encoder elix3r](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF). En la interfaz, desmarca el filtro «ocultar los que no caben»: no hay medición que permita declarar que caben.

Los tamaños siguientes son de archivos, **no** de memoria usada ni de descarga restante. Las dos últimas combinaciones usan una guarda estimada de 35 GiB RAM y 16 GiB VRAM libres, más reservas y margen por duración/resolución. Pueden ser rechazadas en Colab o agotar memoria incluso si se admiten.

| DiT | Peso DiT | Total DiT + encoder Q5 + VAE, aprox. |
|---|---:|---:|
| UD-Q2_K_XL | 8,85 GiB | 19,4 GiB |
| Q4_K_M | 12,65 GiB | 23,2 GiB |
| UD-Q4_K_XL | 13,04 GiB | 23,6 GiB |

[Archivos y revisión Unsloth](https://huggingface.co/unsloth/LTX-2.5-GGUF/tree/ac3c540c6025903784336e99b2dc179385c4c3b7/distilled). Q4 puede preservar más información que Q2, pero no asegura mejor resultado en una generación concreta. Q4_K_M y UD-Q4 no son idénticos; tampoco está medido cuál rinde mejor. Hay Q3, Q5, Q6 y Q8 en el repositorio: no se han integrado ni probado. Q5/Q6/Q8 elevan recursos, y no se debe inferir encaje L4 por el tamaño del archivo.

Comparación útil: misma imagen autorizada, prompt, semilla, duración, tamaño, fps y fidelidad; compara UD-Q2 y Q4 de **2.5** para aislar el DiT, y luego 2.3 frente a 2.5. La salida es estocástica y una sola muestra no prueba superioridad. Comienza por un clip corto; guarda los registros y observa identidad, movimiento, audio y memoria.

## Identidad: candidatos, no funciones activadas

Se añadieron dos fichas informativas al catálogo, con descarga/carga bloqueadas; **ninguna es un botón funcional de identidad**. Desmarca el filtro de compatibilidad para verlas:

* [Best Face ID LTX 2.3](https://huggingface.co/Alissonerdx/LTX-Best-Face-ID) utiliza una foto de cara cercana, LoRA (`Best_FaceID_v1.0_LoRA.safetensors`, 2.466.643.384 bytes) y el nodo `LTXIdentityTransfer` de [BFSNodes](https://github.com/alisson-anjos/ComfyUI-BFSNodes). El autor recomienda describir también los rasgos y la escena. Puede copiar encuadre o crear un efecto de máscara; no garantiza identidad.
* El adaptador separado CharacterSheet del mismo repositorio (1.308.756.480 bytes) usa una ficha de personaje y configuración propia `native_resolution`; no intercambiar pesos sin cambiar el flujo.

La receta de referencia del nodo recomienda CFG 3–5 y no LightX2V; nuestra receta 2.3 es destilada, CFG 1 y 8 pasos. No hay evidencia de que producirán buena identidad juntos ni de que quepan en L4. Añadir el nodo modifica el forward del modelo y la gestión de latentes de vídeo y audio; Comfy puede avisar de parches LoRA no aplicados sin abortar. Para implementarlo falta aislar y fijar el código de referencia, comprobar de forma estricta qué pesos se aplicaron, probar el grafo A/V y medir memoria e identidad en GPU. La licencia del adaptador figura como `other`; revisa los términos y usa solo imágenes autorizadas.

[Ingredients oficial de LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients) es otro enfoque para fichas de personaje/objetos/escenas, con acceso restringido, entrenado para **2.5 dev** y referencia de vídeo estático, no una simple carga de LoRA en este flujo destilado. [Union Control LTX 2.3](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) sirve para guías estructurales como pose, bordes o profundidad, no garantiza identidad facial. No asumir que IP-Adapter de SD/SDXL funciona directamente con LTX. Tampoco confundir rostro con [ID-LoRA de voz](VOICE_GGUF_RESEARCH.md): es otra ruta y está bloqueada preventivamente para L4.

## Investigación adicional: ejemplos realmente publicados

Se analizaron, además de las tarjetas de modelo, los archivos de flujo de [Bambushu/ltx-faceid-lipsync](https://github.com/Bambushu/ltx-faceid-lipsync/tree/3a133e0c06b6ab58ec86fb1516b36c518bfc66a6) (revisión `3a133e0c06b6ab58ec86fb1516b36c518bfc66a6`):

* Su flujo Mac carga **LTX 2.3 dev Q4_K_M**, un LoRA de destilación a 0,6 y Best Face ID a 1, con `LTXIdentityOverlapConditioning` de BFSNodes, CFG 1 y 8 pasos. El autor publica ejemplos generados en un Mac, no en Colab L4. Su flujo CUDA apunta a **dev Q6_K**. No es la misma receta que el GGUF ya destilado de Studio: es evidencia de que GGUF + identidad + CFG 1 se puede hacer, no una validación de nuestro modelo ni de L4.
* El flujo también propone fijar un audio proporcionado mediante VAE y máscara de ruido cero. Según su autor eso guía la animación con ese audio; es distinto de clonar una voz y no está implementado en nuestra ruta.
* [ChrisColeTech/ComfyUI-GGUF-Loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader) publica un nodo `LTX Face Identity Reinforcer` que combina referencia, recorte y las fases de Best Face ID; su autor describe una comparación con GPU y compatibilidad de latentes A/V. La evidencia publicada no establece que funcione en nuestra versión fijada de Comfy, nuestro DiT destilado o una L4. El proyecto integra muchos otros nodos: no conviene reemplazar sin más el cargador GGUF actual ni cargar código adicional en el runtime sin fijar y revisar.
* [10S-Comfy-nodes](https://github.com/TenStrip/10S-Comfy-nodes) describe otra implementación del reinforcer que combina I2V y referencia facial. Es una alternativa concreta a investigar para mantener composición e identidad por separado, no una garantía de parecido.

La siguiente integración razonable sería un modo optativo y aislado de 2.3 con referencia facial + Best Face ID, validación explícita de parches LoRA y un A/B frente a I2V, primero con clips cortos. Hay que determinar si usar dev Q4 + LoRA de destilación como el ejemplo, o si nuestro ya destilado Q4 responde bien; no tratar estas opciones como equivalentes. No combinarlo inicialmente con guías de pose: el flujo Bambushu documenta un fallo concreto al combinar Face-ID con IC-LoRA. Una prueba real debe recoger memoria, configuración y vídeo; no basta que Comfy acepte el grafo.

## Caché y protección de memoria

Para las variantes 2.3 y 2.5, una segunda petición del mismo modelo reutiliza el **mismo proceso Comfy residente**, sus nodos y, si Comfy los conserva, los pesos en caché. El registro `Comfy reutiliza de caché` indica los componentes que Comfy informó como reutilizados: la existencia del proceso, por sí sola, no garantiza que todos los pesos permanezcan en VRAM. Comfy puede mover o liberar pesos; una sesión de Colab reiniciada también pierde la caché RAM/VRAM.

Si se cambia de modelo, Studio descarga el proceso anterior de la memoria antes de admitir el nuevo. Bajo presión el guardián libera modelos inactivos; una generación activa no se descarga desde otro hilo a mitad de uso. Ante presión crítica durante la ejecución o cancelación, la tarea se detiene y se libera el proceso. Si la capacidad estimada no alcanza, no comienza la nueva instalación/descarga. Ninguna de estas acciones borra pesos ya guardados en disco: tras liberar memoria habrá que recargarlos, pero no descargarlos otra vez si siguen intactos. Una unidad temporal de Colab puede desaparecer al cerrar el entorno; Studio no puede conservarla por sí solo.

Se añadieron pruebas simuladas para las tres variantes con Gemma Q5: reutilización de sesión sin reinstalación, cambio de modelo y rechazo por memoria antes de descargar, liberación al cancelar o en presión crítica, y expulsión segura de caché inactiva sin borrar archivos. No miden memoria real ni prueban la inferencia en GPU.
