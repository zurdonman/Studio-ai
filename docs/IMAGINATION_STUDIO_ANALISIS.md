# Imagination Studio: funciones y persistencia para Studio IA

Análisis de código y documentación de [Imagination Studio](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman/tree/e52be317d37100a9ea9d030267e81c93ecab55c7), revisión `e52be317d37100a9ea9d030267e81c93ecab55c7`. No se ha instalado ni ejecutado. La disponibilidad y los resultados de sus modelos dependen del runtime y del hardware; no son evidencia de que encajen en una L4 de Colab. El proyecto es Apache-2.0; cada modelo tiene su propia licencia.

## Lo que hace la herramienta de referencia

- **Create** reúne motor, resolución, texto, imágenes de inicio o referencias, voz, sonido y cola. Sus opciones FastH3/H3 no equivalen a los motores LTX de Studio IA. [Guía de usuario](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman/blob/e52be317d37100a9ea9d030267e81c93ecab55c7/docs/user-guide.md).
- **Production** planifica segmentos, acción, duración y diálogo, permite continuar un segmento desde el fotograma anterior y solicitar los pendientes. **Slate** añade escenas, planos, biblia, personajes, localizaciones, tomas y compilación del plano. La implementación concreta de Slate compila para H3; no hay que trasladar esa dependencia al organizador común.
- **Editor** organiza una línea de tiempo de una pista con recortes y música global; **Library** permite buscar, reutilizar y enviar clips. La búsqueda semántica usa un modelo de visión local adicional, con su propio coste de memoria.
- Otras funciones documentadas incluyen generación/edición de imágenes, scout, referencias, encadenamiento de clips, voces, música, restauración y herramientas de acabado. Algunas dependen de instalaciones y plataformas concretas. Descripción upstream no equivale a implementación aquí.

## Cómo guarda y recupera la herramienta de referencia

- En [`app/slate.py`](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman/blob/e52be317d37100a9ea9d030267e81c93ecab55c7/app/slate.py), cada proyecto Slate se guarda como `slate/<id>/project.json`. Usa archivo temporal y sustitución atómica, y comprueba la versión `updatedAt` cuando el cliente la proporciona para evitar sobrescribir cambios más recientes. Borrar un proyecto lo mueve a una papelera con restauración.
- [`app/slate_package.py`](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman/blob/e52be317d37100a9ea9d030267e81c93ecab55c7/app/slate_package.py) prepara un paquete portátil con proyecto, archivos referenciados, guion cuando existe y listas de planos; verifica cambios de archivos durante exportación. La disponibilidad exacta de cada medio depende de que exista. No es la misma estructura que el ZIP actual de Studio IA.
- Las opciones de interfaz también usan `localStorage` del navegador; no sustituyen la copia de proyectos y medios. Las descripciones de biblioteca están en `library/captions.json` en el disco de la herramienta.
- Su gestor de memoria libera módulos inactivos y vigila presión. Su propio código llama a `/free` de Comfy con `unload_models` y `free_memory`; esto **no demuestra** que se limpien solo temporales mientras permanecen los pesos residentes. No copiar sin evaluar las diferencias de runtimes y la seguridad en Colab.

## Estado de Studio IA (no confundir con la otra aplicación)

- Resultados y subidas residen bajo `STUDIO_DATA_DIR`; en Colab, si no se cambia, se usa `/content/studio_data`, almacenamiento de sesión que puede perderse al desconectar. Las salidas tienen metadatos JSON junto al archivo. El ZIP de resultados actual incluye salidas y registro, pero **no** empaqueta las subidas, los personajes, el catálogo de modelos ni un proyecto de escenas portable completo.
- Personajes de Higgsfield se guardan en `state/characters_vault.json` bajo ese directorio. La mezcla musical y opciones de Comfy se guardan en el navegador. La línea de montaje en memoria de la página no es un proyecto persistente completo.
- El organizador de escenas/planos/tomas, exportación/importación portable completa y la reutilización de pesos en RAM/VRAM **no están implementados**. El runtime H3 GGUF experimental termina tras cada clip y conserva los pesos en disco; no promete caché residente.

## Diseño recomendado para implementar después de los modelos

1. Una sección propia **Proyecto / Producción** con proyecto → escenas → planos → tomas. Un plano guarda prompt, duración, referencias, selección del modelo y capacidades usadas; un adaptador traduce cada plano a la cola de cada motor sin forzar todos los planos a LTX o H3.
2. Guardado atómico del proyecto en servidor, versión/esquema y detección explícita de conflictos, con recuperación ante guardado interrumpido y papelera; no guardar tokens en los proyectos ni exponerlos en exportaciones.
3. Exportación e importación portables con manifiesto y medios elegidos, verificación de integridad, rutas relativas confinadas, límites de tamaño, prevención de zip-slip, espacio suficiente y restauración que no sobrescriba proyectos o archivos existentes. Guardar en Drive o descargar el paquete si se quiere sobrevivir al fin de Colab; solo guardar en `/content` no basta.
4. Vincular tomas y cola por identificadores, recuperar resultados terminados tras recargar y marcar medios ausentes sin destruir referencias. No sustituir un fotograma final sin acción explícita.
5. Evaluar la caché residente aparte: medir antes/después, validar limpieza y cancelación, limitar a un proceso propio aislado y mantener vigilancia real; un vigilante no evita todos los OOM súbitos. No activar persistencia de pesos en memoria hasta demostrarla.
