# LTX GGUF: alternativas Q4 e investigación de identidad

## Qué se puede ejecutar

* LTX 2.3 destilado Q4_K_M ya existente: un usuario probó I2V en L4 con audio y vídeo. Mantiene 8 pasos y CFG 1; la imagen es el **primer fotograma**, no una referencia FaceID independiente.
* LTX 2.5 destilado UD-Q2_K_XL + Gemma 4 GGUF Q5_K_M: el usuario informa de una generación en L4. No tenemos los parámetros, vídeo ni medición de memoria de esa ejecución. Percibe peor calidad que con el Q4_K_M anterior; no podemos atribuir la diferencia a un componente.
* **Nuevos, sin prueba GPU:** LTX 2.5 destilado Q4_K_M + Gemma 4 Q5 y UD-Q4_K_XL + Gemma 4 Q5. Usan el mismo entorno/parche aislado, Gemma y VAE que el UD-Q2 + Q5; al cambiar se descarga el DiT que falte y se reutilizan los archivos comunes verificados. Requieren aceptar las condiciones del [encoder elix3r](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF). En la interfaz, desmarca el filtro «ocultar los que no caben»: no hay medición que permita declarar que caben.

Los tamaños siguientes son de archivos, **no** de memoria usada ni de descarga restante. Las dos últimas combinaciones tienen una estimación orientativa de 35 GiB RAM y 16 GiB VRAM, con incrementos por duración/resolución. No es un pico medido; pueden detenerse por presión real o agotar memoria.

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

El modo se selecciona en Imagen a Vídeo > Audio y voz GGUF > Voz de referencia. El prompt debe describir la escena y el diálogo nuevo. Solo imagen y audio propios o autorizados. Se muestra un incremento orientativo de 5 GiB RAM y 3 GiB VRAM, no medido. La admisión comprueba el mínimo real de arranque y el vigilante puede detener Comfy bajo presión durante la inferencia; no se garantiza cabida en L4. Si se necesita activar el nodo en un proceso residente anterior, se reinicia el proceso una vez sin borrar pesos y se vuelve a comprobar el presupuesto. Después de generar, el proceso se cierra y los archivos permanecen en disco.

Los grafos se validaron contra los esquemas reales del Comfy fijado para Gemma Q5 con archivos ficticios, y las rutas y salidas se probaron con mocks. Esto **no ejecuta los pesos reales**, no comprueba la aplicación real del adaptador, ni demuestra compatibilidad de sus tensores con el modelo cuantizado; un error explícito de nodo o `MSR_LORA_INCOMPATIBLE` es un resultado posible. El registro L4 del usuario ya mostró 0,8 GiB libres con la receta base.

## Caché y protección de memoria

Para las variantes 2.3 y 2.5, Studio cierra Comfy al terminar cada vídeo o al fallar/cancelar. La generación siguiente arranca un proceso nuevo y carga los pesos desde disco, sin volver a descargarlos si los archivos siguen presentes y verificados. Comfy gestiona su memoria y offload durante cada ejecución; otros procesos también pueden ocupar VRAM.

Si se cambia de modelo, Studio descarga el proceso anterior de la memoria antes de admitir el nuevo. Bajo presión el guardián libera modelos inactivos; una generación activa no se descarga desde otro hilo a mitad de uso. Ante presión crítica durante la ejecución o cancelación, la tarea se detiene y se libera el proceso. Si la capacidad estimada no alcanza, no comienza la nueva instalación/descarga. Ninguna de estas acciones borra pesos ya guardados en disco: tras liberar memoria habrá que recargarlos, pero no descargarlos otra vez si siguen intactos. Una unidad temporal de Colab puede desaparecer al cerrar el entorno; Studio no puede conservarla por sí solo.

Las pruebas simuladas comprueban generaciones consecutivas con procesos separados, conservación de archivos, rechazo por memoria y liberación al cancelar o en presión crítica. No miden memoria real ni prueban la inferencia en GPU.

## Diagnóstico de carga en L4 (registro del 15 de noviembre)

En ese registro terminaron dos trabajos **LTX 2.3 destilado GGUF Q4**, I2V con audio nativo, a 320×512, 24 fps, 241 fotogramas y ocho pasos. El segundo informó reutilización del transformer, codificador y ambos VAE. El registro confirma la producción y verificación técnica del MP4, no que pueda reproducirse en el navegador ni que su sonido, sincronía o calidad sean satisfactorios. En una sesión anterior se informaron clips completados con **LTX 2.5 UD-Q4 + Gemma 4 GGUF Q5**: es otra receta.

Los dos intentos FP8 del nuevo registro fueron rechazados por el presupuesto de **72 GiB RAM más reserva**, con 52,96 GiB RAM totales. FP8 de Diffusers no es el modelo GGUF; no debe bajarse su presupuesto para ocultar el rechazo. La comprobación de capacidad total ahora avisa antes de liberar modelos activos cuando el presupuesto nunca puede caber; la comprobación de memoria libre permanece después. Este cálculo sigue siendo preventivo, no una medición real del pico.

Los dos intentos 2.5 GGUF + Gemma Q5 fallaron por `GatedRepoError`; el proceso no recibió token de Studio. Antes de instalar el runtime o expulsar otro modelo, se comprueba el acceso a los pesos de Gemma Q5, salvo que el encoder y el parche gestionado ya estén en disco. En caso de rechazo, iniciar sesión en [el repositorio elix3r](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF), aceptar sus condiciones y configurar en Studio el acceso de esa misma cuenta, con permiso de lectura para repositorios restringidos. No escribir el token en el chat. La aprobación es responsabilidad de Hugging Face. Una precomprobación satisfactoria no garantiza que la descarga, la memoria o la inferencia tengan éxito.

### LTX 2.5 FP8 Comfy + Gemma Q5 (experimental)

Esta receta separada descarga el transformer destilado FP8 preconvertido de
[guillaume127/LTX-2.5-FP8](https://huggingface.co/guillaume127/LTX-2.5-FP8)
(revisión `d0a8ff418087caac5288901c72726ff2ac70e199`, SHA-256 fijado). Utiliza
`UNETLoader` de Comfy con `weight_dtype=default` para conservar los tipos FP8/BF16
incluidos, y comparte Gemma Q5 y VAE con las recetas GGUF 2.5. No es la receta
Diffusers que carga BF16 y convierte a FP8. Acepta las condiciones LTX y las de
[elix3r para Gemma Q5](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF),
y configura Hugging Face en Studio con una cuenta autorizada; no pegues tokens en chat.

El transformer ocupa 21,87 GiB en disco, más ~8,86 GiB de encoder y VAE. El autor
informó una RTX 4090 de 24 GB y 64 GB de RAM, no una L4. La estimación (44 GiB RAM + 18 GiB VRAM, más aumentos según configuración) es
orientativa **sin medición de picos** y no bloquea por sí sola; no es una garantía; Comfy puede hacer
offload a CPU y fallar o ir lento. Clips largos y MSR aumentan el presupuesto.
No se ha probado la carga de los pesos ni la inferencia GPU en Studio. En L4 prueba
primero 512×320, ~2 s y sin referencia, y verifica la galería. El filtro de modelos
compatibles oculta las recetas de encaje desconocido: desactívalo para seleccionarla.
Los archivos compartidos se reutilizan; tras cada vídeo se libera el proceso sin borrar pesos. El grafo de audio completo y MSR también sigue experimental.


### LTX 2.5 22B distilled NVFP4 (DDB Comfy, experimental)

Opción comunitaria de [DmitryDB](https://huggingface.co/DmitryDB/LTX-2.5-ComfyUI-Quants)
con transformer NVFP4 (12,64 GiB), Gemma Q5 y VAE compartidos. Es distinta del
checkpoint NVFP4 oficial. Se fija revisión y SHA-256; se aplican las licencias
LTX y el acceso a elix3r. El autor reporta carga y prueba de proyección en una
RTX 4090, **no un vídeo completo**. La L4 es Ada: no tiene aceleración nativa
NVFP4; el autor describe un fallback para Ada. En el registro del 16 de noviembre
se completaron cuatro clips I2V con audio en Studio/L4: dos de ~10 s a 512×320
y dos a 768×512. No se midió la calidad ni se garantiza encaje, velocidad,
resultado o compatibilidad con MSR y otras configuraciones. El presupuesto inicial estimado es 40 GiB RAM y 18 GiB VRAM, más
reservas, activaciones y aumentos por duración y audio; no es un pico medido.
Prueba primero 512×320, ~2 s, sin referencia, y revisa la galería.

### Registro del 15 de noviembre, 18:40

UD-Q4 + Gemma Q5 completó tres clips de ~10 s en L4; los dos trabajos
intermedios se detuvieron tras reutilizar caché con ~0,77 GiB de VRAM libre.
No prueba que todos los fallos futuros tengan esa causa. Desde este cambio el
motor se cierra después de cada vídeo; la próxima carga tarda más, pero usa los
mismos archivos en disco.


### Vigilancia de memoria Comfy

En LTX 2.3 y 2.5 con Comfy, las cifras del catálogo y los incrementos por
duración/audio son estimaciones orientativas, no picos medidos ni criterios de
rechazo automáticos. Antes de arrancar se requieren 4 GiB de RAM y 2 GiB de
VRAM libres adicionales a las reservas y medición fiable. Durante el proceso
Comfy se vigila la RAM disponible efectiva (incluidos límites de cgroup y una
estimación de caché recuperable) y VRAM de la GPU; por debajo de la reserva RAM
(10 %, mínimo 1 GiB), del margen crítico VRAM (mitad de 8 %, mínimo 0,25 GiB),
o al fallar repetidamente la medición se termina el grupo del subproceso. También
se interrumpe al cancelar. Los pesos permanecen en disco y el motor se libera
tras cada vídeo. El sondeo puede llegar tarde ante picos súbitos, cargas externas
o bloqueos del sistema: **no garantiza evitar un OOM o reinicio de Colab**.
Otros motores que cargan en el proceso principal mantienen sus guardias propias.
