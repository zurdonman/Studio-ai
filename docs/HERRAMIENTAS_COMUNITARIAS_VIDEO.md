# Herramientas comunitarias de vídeo: evaluación, no instalación

Revisión de código fuente realizada sin instalar dependencias ni descargar modelos:

| Proyecto | Revisión revisada | Licencia del código | Estado en Studio |
|---|---|---|---|
| [Freefield](https://github.com/neuregex/freefield) | `f0c2358a46c7008ff8503d4324b6cc8bc7657e75` | MIT | Referencia de flujos y diseño; no instalado ni integrado como motor. |
| [Creativefield](https://github.com/rhymeas/Creativefield) | `e65c3464eab0a8ee19466db51e2b92eb7d8fc1ff` | GPL-3.0-only | Aplicación y flujos revisados como referencia; no instalada ni integrada. |
| [AInVFX Paint a plate](https://github.com/AInVFX/ainvfx-paintaplate) | `d95292c1a96073a34bfec82d7462dbd2887d0000` | Apache 2.0 | Candidato para un futuro adaptador de edición de plano; no instalado ni validado en L4. |

## Freefield

Es una aplicación independiente con frontend, servidor Node, gateway y 120 manifiestos de apps/workflows, no un único nodo Comfy. Incluye tareas de imagen, vídeo y utilidades que utilizan familias y componentes diferentes. Sus indicadores de pruebas GPU no equivalen a que todas funcionen en nuestra L4 ni autorizan los modelos. Hacer funcionar la aplicación completa añadiría otro servidor, gestión de archivos y ejecución de flujos no adaptada al gestor de trabajos y memoria de Studio.

Vía gradual propuesta: elegir una herramienta concreta por necesidad; revisar su licencia de código, manifiesto, workflow, componentes, licencias, requisitos de memoria y entradas/salidas; adaptarla a la cola y al catálogo agnósticos, sin importar 120 flujos ciegamente. Probar primero en CPU el esquema y aislamiento, y después inferencia real en hardware compatible. No se ha hecho todavía.

## Creativefield

Aplicación independiente alrededor de ComfyUI: frontend React, servidor Node/Fastify, almacenamiento y manifiestos que exponen workflows como herramientas. Su README describe editor de imágenes por capas, creación y modificación de vídeo, fotogramas clave, cortometrajes y 3D. También hay rutas remotas opcionales con proveedores, créditos y envío de datos: no equivalen a ejecución local ni se activan en Studio.

- **Voz y labios:** su flujo local `wan21-infinitetalk` anima **un retrato estático** con un audio WAV, para un solo hablante, no edita los labios de un vídeo existente. El manifiesto limita la ventana a 1–3,2 s y declara una prueba de retrato en RTX 4090 con movimiento de boca visible, sin medición fonética ni garantía de calidad general. Necesita Wan 2.1 I2V 14B FP8, parche InfiniteTalk, codificador de audio y componentes Wan adicionales. La prueba es del autor, no de Studio; no demuestra encaje L4 ni uso de memoria. Chatterbox aparece como TTS local en voz predeterminada, sin clonación en ese flujo. Kling Lip Sync figura como servicio remoto, no modelo local.
- **H3 y cortometrajes:** hay rutas H3 y HyperFlow H3 Flash con componentes y nodos propios; no son FastH3 ni nuestra ruta H3 GGUF `sd.cpp`. Algunos manifiestos declaran pruebas concretas en RTX 4090, que no validan Studio ni cada variante. La licencia de H3 es independiente y debe verificarse; no presumir que el uso personal elimina restricciones territoriales.
- **Organización:** su enfoque de proyectos, fotogramas clave, manifiestos y herramientas por tarea aporta ideas para el organizador agnóstico de Studio, que sigue pendiente. No implica que se hayan importado sus funciones.

El proyecto declara GPL-3.0-only, mientras Studio usa MIT. Estudiar ideas no equivale a tener autorización para copiar e incorporar código o workflows: antes de redistribuir derivados o integrar componentes concretos hay que revisar licencias, atribuciones, dependencias y obligaciones aplicables, además de las licencias de cada modelo. El README identifica Windows/RTX 4090 y 64 GB RAM como su máquina de pruebas; no acredita funcionamiento limpio en Linux/Colab. No se han ejecutado ni instalado sus servidores, modelos o nodos. Exponer el servidor o Comfy en Colab sin una revisión de autenticación, orígenes y aislamiento sería inseguro.

**Hardware:** la L4 es una referencia de prueba, no un techo de producto. Hay que valorar A100/H100 y GPUs locales según capacidad y memoria real, arquitectura, dependencias y compatibilidad de cada receta. La ruta H3 GGUF actual de Studio está específicamente restringida a L4/sm89; ampliar esa restricción exige implementación y validación, no basta con seleccionar una GPU mayor. El vigilante y la liberación explícita de memoria deben mantenerse en todos los perfiles.

## Paint a plate

El proyecto proporciona una página/nodos Comfy para pintar, borrar, iluminar o generar fluidos en un primer fotograma y propagar el resultado al plano; también contempla composición, gestión de color y exportación. Su propio flujo combina varios modelos (incluidos Klein, LTX 2.5 y SAM), dependencias OpenEXR/OpenColorIO, y requiere ComfyUI >= 0.43. Su README estima unos 60 GB de modelos y recomienda al menos 32 GB de VRAM; no es evidencia de compatibilidad con L4. Algunas descargas requieren aceptación o acceso individual: licencia Apache del código no significa licencia libre de los pesos.

La página expone rutas adicionales y trabaja con archivos; publicarla directamente o incluirla en la lista de nodos permitidos del Comfy privado actual ampliaría la superficie de ataque. Tampoco debe quedar un puerto Comfy sin autenticación expuesto en Colab. Integración viable solo tras auditar rutas/archivos, autenticación, dependencias, procedencia y licencias de cada modelo, versiones y pico real de memoria; después, incorporar de forma explícita a un editor de plano con aislamiento y el mismo vigilante de memoria. La necesidad de 32 GB recomendados y el conjunto de modelos hacen que la L4 sea una incógnita, no una plataforma soportada.

## Memoria: modo residente experimental de Studio

La casilla está en **Vídeo → Memoria y offload de Comfy**, solo para las recetas LTX de Comfy. Por defecto está desactivada. Conserva, si Comfy no los ha desalojado, las salidas de cuatro cargadores base y elimina las salidas intermedias, caché de objetos, historial del prompt y directorios privados de entrada/salida/temporales. No reutiliza el vídeo generado como caché de inferencia. Los LoRA y codificaciones no se conservan como caché; Comfy puede descargar o mover pesos entre RAM/VRAM. El número de cargadores conservados se comunica en el log y no mide dónde están los pesos.

Se mantiene el vigilante incluso en reposo. Liberar memoria, cambiar modelo/política/modo, error, cancelación o falta de confirmación de limpieza cierra el proceso. La vigilancia no evita un OOM instantáneo y otros mecanismos de Comfy pueden desalojar los cargadores. El destino final del usuario se conserva; los pesos descargados no se borran.

La limpieza depende de internals de las dos revisiones fijadas de Comfy, no es una extensión genérica. Se probó en CPU con cargadores y operaciones sintéticos sobre ambos árboles y caché de presión: ante dos grafos iguales, los cuatro cargadores se ejecutaron una vez y los nodos de cálculo y salida dos veces. También se comprobó que la presión de RAM puede desalojar cargadores. No es una inferencia de vídeo, no demuestra el comportamiento de modelos reales ni está probado con GPU L4. Se recomienda comenzar con un clip corto y comprobar el log, y desactivar la casilla si hay problemas.
