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
* El flujo también propone fijar un audio proporcionado mediante VAE y máscara de ruido cero. Según su autor eso guía la animación con ese audio; es distinto de clonar una voz. Studio ahora ofrece una ruta experimental para audio completo tanto en 2.3 como en 2.5, sin validación GPU de la ruta 2.5.
* [ChrisColeTech/ComfyUI-GGUF-Loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader) publica un nodo `LTX Face Identity Reinforcer` que combina referencia, recorte y las fases de Best Face ID; su autor describe una comparación con GPU y compatibilidad de latentes A/V. La evidencia publicada no establece que funcione en nuestra versión fijada de Comfy, nuestro DiT destilado o una L4. El proyecto integra muchos otros nodos: no conviene reemplazar sin más el cargador GGUF actual ni cargar código adicional en el runtime sin fijar y revisar.
* [10S-Comfy-nodes](https://github.com/TenStrip/10S-Comfy-nodes) describe otra implementación del reinforcer que combina I2V y referencia facial. Es una alternativa concreta a investigar para mantener composición e identidad por separado, no una garantía de parecido.

La siguiente integración razonable sería un modo optativo y aislado de 2.3 con referencia facial + Best Face ID, validación explícita de parches LoRA y un A/B frente a I2V, primero con clips cortos. Hay que determinar si usar dev Q4 + LoRA de destilación como el ejemplo, o si nuestro ya destilado Q4 responde bien; no tratar estas opciones como equivalentes. No combinarlo inicialmente con guías de pose: el flujo Bambushu documenta un fallo concreto al combinar Face-ID con IC-LoRA. Una prueba real debe recoger memoria, configuración y vídeo; no basta que Comfy acepte el grafo.

## Candidato adicional: LTX 2.5 con imagen y voz emparejadas

**Integrado como experimento en Studio, todavía sin validación de inferencia con GGUF, GPU ni L4.**
[Licon MSR V2](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference)
se presenta como un LoRA para LTX 2.5 con hasta cinco imágenes de referencia y
hasta dos referencias de voz emparejadas con las imágenes 1 y 2. No es un
IP-Adapter genérico de SD/SDXL ni una garantía de parecido o diálogo exacto.
La tarjeta declara Apache-2.0 para el repositorio; el modelo base, Gemma y sus
componentes mantienen sus propios términos. La tarjeta también advierte de un
problema conocido con sonidos ambientales sin música en el audio de referencia.

Revisión inspeccionada del modelo `aa19b48f845995525d8ace60cdc89b842e951ce6`:
`LTX-2.5-Licon-MSR-V2.safetensors`, 2.215.094.328 bytes, LFS SHA-256
`7cf982168d4f77f44b02943fb458f937f90c89335dcf7cf71b2f425e78bab713`.
[ComfyUI-LTX2.5-MSR](https://github.com/liconstudio/ComfyUI-LTX2.5-MSR),
commit `03061c18f8a79d26f3c63931dd8e28f5e5d872a6`, contiene cargador y
nodo guía Apache-2.0; el ejemplo V2 carga LTX 2.5 **distilled BF16**, usa
`euler_ancestral` y dos etapas. La receta Studio es GGUF destilada y una
pasada: no se ha demostrado que produzca buenos resultados o que quepa en L4.
El mismo repositorio indica que una sola etapa es posible, pero eso no valida
nuestra combinación concreta ni sus resultados.

Studio prepara la extensión en un runtime aislado, fijada por commit, y descarga y verifica el SHA-256 del adaptador. Un cargador estricto comprueba que los tensores y parches LoRA se hayan reconocido y aplicado; el cargador especializado conserva y valida las embeddings de referencia. El guía usa el latente de vídeo antes de concatenar A/V, `pic1` y el audio codificado en `audio_ref1`; después de separar A/V se aplica `LTXVCropGuides`. La misma imagen se utiliza además como primer fotograma. La muestra es condicionamiento, **no** una pista que se garantice oír literalmente; el nodo puede recortarla a su ventana interna.

El modo se selecciona en Imagen a Vídeo > Audio y voz GGUF > Voz de referencia. El prompt debe describir la escena y el diálogo nuevo. Solo imagen y audio propios o autorizados. Se reserva un margen preventivo adicional de 5 GiB RAM y 3 GiB VRAM, no medido. Bajo presión puede rechazarse antes de descargar o detenerse durante la inferencia; no se garantiza cabida en L4. Si se necesita activar el nodo en un proceso residente anterior, se reinicia el proceso una vez sin borrar pesos y se vuelve a comprobar el presupuesto. Las generaciones posteriores reutilizan el proceso cuando sea seguro.

Los grafos se validaron contra los esquemas reales del Comfy fijado para Gemma Q5 con archivos ficticios, y las rutas y salidas se probaron con mocks. Esto **no ejecuta los pesos reales**, no comprueba la aplicación real del adaptador, ni demuestra compatibilidad de sus tensores con el modelo cuantizado; un error explícito de nodo o `MSR_LORA_INCOMPATIBLE` es un resultado posible. El registro L4 del usuario ya mostró 0,8 GiB libres con la receta base.

## Caché y protección de memoria

Para las variantes 2.3 y 2.5, una segunda petición del mismo modelo reutiliza el **mismo proceso Comfy residente**, sus nodos y, si Comfy los conserva, los pesos en caché. El registro `Comfy reutiliza de caché` indica los componentes que Comfy informó como reutilizados: la existencia del proceso, por sí sola, no garantiza que todos los pesos permanezcan en VRAM. Comfy puede mover o liberar pesos; una sesión de Colab reiniciada también pierde la caché RAM/VRAM.

Si se cambia de modelo, Studio descarga el proceso anterior de la memoria antes de admitir el nuevo. Bajo presión el guardián libera modelos inactivos; una generación activa no se descarga desde otro hilo a mitad de uso. Ante presión crítica durante la ejecución o cancelación, la tarea se detiene y se libera el proceso. Si la capacidad estimada no alcanza, no comienza la nueva instalación/descarga. Ninguna de estas acciones borra pesos ya guardados en disco: tras liberar memoria habrá que recargarlos, pero no descargarlos otra vez si siguen intactos. Una unidad temporal de Colab puede desaparecer al cerrar el entorno; Studio no puede conservarla por sí solo.

Se añadieron pruebas simuladas para las tres variantes con Gemma Q5: reutilización de sesión sin reinstalación, cambio de modelo y rechazo por memoria antes de descargar, liberación al cancelar o en presión crítica, y expulsión segura de caché inactiva sin borrar archivos. No miden memoria real ni prueban la inferencia en GPU.
