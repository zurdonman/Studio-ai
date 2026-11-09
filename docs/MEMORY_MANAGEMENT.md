# Gestión automática de RAM y VRAM

Studio vigila la memoria y libera **modelos inactivos**, no sus archivos. No hace falta pulsar «Liberar RAM/VRAM» antes de cada tarea. Con memoria suficiente conserva los modelos calientes para reutilizarlos; con presión elimina primero los usados hace más tiempo (LRU).

## Qué mide y cuándo actúa

- El vigilante empieza y termina con el servidor FastAPI. Muestrea **cada segundo**, también sin navegador ni agente conectados. `STUDIO_MEMORY_POLL_SECONDS` permite ajustar el intervalo entre 0,25 y 60 segundos; no desactiva las comprobaciones de carga/inferencia.
- **RAM efectiva:** `MemAvailable` de psutil, limitada por las cuotas cgroup v1/v2 visibles. Se comprueban también los padres visibles; no se usa la RAM del host como si toda perteneciera al contenedor. Solo se estima recuperable la caché de archivo inactiva y limpia, no toda `memory.current` ni las páginas anónimas del modelo.
- **VRAM real del dispositivo 0:** `cudaMemGetInfo`, o `nvidia-smi`. No se suma la memoria de distintas GPU. Si solo se conoce el allocator de PyTorch, la memoria libre del dispositivo queda **desconocida**: ese allocator no contabiliza todos los procesos ni los motores ONNX. No se admite una carga basándose en esa estimación incompleta.
- RSS del proceso y máximos observados se registran aparte. Son muestras de proceso/dispositivo, **no benchmarks certificados del modelo**.

Las reservas adicionales son:

| Recurso | Margen preventivo | Umbral crítico durante inferencia |
|---|---|---|
| RAM | mayor de 1 GiB o 10 % del total efectivo | mitad del margen preventivo |
| VRAM | mayor de 0,5 GiB o 8 % del total de GPU | mitad del margen preventivo |

Al bajar del margen preventivo, Studio intenta limpiar memoria recuperable y después expulsa modelos **inactivos**, de uno en uno, volviendo a medir tras cada liberación. Se recogen referencias/ciclos Python, caché CUDA y arenas libres de glibc cuando están disponibles. `empty_cache()` **no libera tensores vivos**, y el offload de GPU a CPU **no libera la RAM de esos pesos**.

## Protección de una tarea activa

Imagen, vídeo, inspector visual, TTS local, música/efectos y LipSync comparten el mismo turno de GPU. Cada operación protege sus slots mientras los utiliza. Las operaciones anidadas mantienen esa protección hasta que termina el último propietario.

El vigilante **no borra un handle desde otro hilo mientras se está cargando o usando**. Si el motor tiene el turno, la vigilancia continúa midiendo y la reclamación se pospone a un punto seguro del propio hilo del motor:

- antes/después de los cargadores locales;
- entre transferencias de parámetros/cambios de dispositivo de los módulos opcionales soportados;
- pasos de difusión y criterios de parada de generación compatibles;
- carga de cada sesión ONNX y puntos de cancelación por fotograma/chunk.

Los hooks son idempotentes, no alteran dtypes, tensores ni pesos y solo actúan dentro de una operación gestionada por Studio. Se instalan después de resolver las clases de carga perezosa. Se han revisado los puntos de carga de Diffusers 0.42.1 y Transformers 4.57.1; cambios posteriores en APIs internas requieren volver a comprobarlos.

Si la reclamación no recupera el margen, **una carga de nuevos tensores se detiene ya en el margen preventivo**. Una inferencia en marcha no se expulsa por una fluctuación leve: se detiene cooperativamente si llega al umbral crítico. Tras desenrollar la llamada fallida se retiran los handles afectados y las referencias de traceback recuperables, sin interrumpir arbitrariamente otro hilo. No hay inyección de excepciones en hilos ni señales para matar la inferencia.

## Admisión antes de descargar y cargar

Se exige memoria disponible para **base completa + componentes de la receta + todos los adaptadores residentes + reserva**. Se comprueba antes de la descarga implícita y otra vez antes de cargar. Un LoRA a escala cero sigue teniendo pesos en memoria: también cuenta. Si la base ya está residente y solo se cambian adaptadores, se comprueba la capacidad del conjunto pero se exige memoria incremental, sin cobrar la base dos veces.

El catálogo, los presets y MCP comparten estas estimaciones y reservas. El preset no aprueba una pila 14B con tres LoRA solo por detectar una L4: publica `selected_video_stack.memory_assessment` y marca `do_not_execute` en fases bloqueadas. `ok=true` en un preset indica que se ha compilado el plan, **no que su ejecución esté validada**. No se sustituye la receta bloqueada ni se inicia una descarga alternativa.

Ejemplo de planificación para una L4 con **52,96 GiB RAM y 22,03 GiB VRAM**:

- Margen adicional: aproximadamente **5,30 GiB RAM / 1,76 GiB VRAM**.
- VACE 14B FP8 + FusionX Phantom + UniAnimate + MoviiGen: estimación conjunta **44,47 GiB RAM / 17 GiB VRAM**, antes de esas reservas.
- La admisión exige aproximadamente **49,77 GiB de RAM disponible** después de liberar los modelos inactivos. Tener 52,96 GiB de RAM total no garantiza ese disponible.

Son presupuestos de ingeniería para los defaults, **no picos GPU medidos**. Resolución, fotogramas, decodificación VAE, versiones, archivos, fragmentación y procesos externos pueden cambiar el consumo.

## Diagnóstico y uso desde MCP

- La cabecera web muestra **🛡 Memoria automática** y el margen; el título informa de los slots protegidos y la última acción.
- `GET /api/status` → `memory.guardian`; `GET /api/memory/status` → observación, presión, reservas, slots protegidos, operaciones activas y eventos recientes.
- `studio_get_state_and_manual` → `memory_guardian`, con la misma política. `loaded_in_vram` se mantiene por compatibilidad, pero sus entradas pueden residir también en RAM/offload.
- Cada slot publica `last_used_at` e `in_use`. Consultar el estado no lo rejuvenece artificialmente en LRU.
- Cada job publica `memory_protection`, con motivo/código de parada y máximos observados. Mantiene su ID y queda `failed` si lo detiene la protección; **no se debe reenviar a ciegas**.
- Los últimos 40 eventos se conservan en `state/memory_guardian.json`, con secretos comunes ocultos; los últimos 12 se exponen en el estado. El ZIP de diagnóstico incluye el archivo, incluso si el servidor ya no funciona. Sobrevive solo mientras exista el mismo disco/VM.

Códigos relevantes:

- `MEMORY_BUDGET_EXCEEDED`: no se cumple presupuesto + reserva, incluso tras intentar liberar modelos inactivos. No seguir reservando ni cambiar automáticamente de modelo.
- `MEMORY_BUDGET_UNKNOWN`: falta una estimación de la base/componente/adaptador; no inventar que cabe.
- `MEMORY_SAFETY_STOP`: un punto seguro detectó memoria crítica, margen de carga agotado o medición incompleta.
- `MEMORY_MEASUREMENT_UNAVAILABLE`: falló la lectura de memoria; no se prosigue una reserva a ciegas. `measurements_healthy` y `sample_age_seconds` permiten distinguir un vigilante vivo de una medición saludable.
- `FP8_RUNTIME_UNSUPPORTED`: una receta FP8 no se expande silenciosamente a BF16/FP32 como remedio.

Revisar `last_observation`, `recent_events`, `protected_slots`, el job y el ZIP. Si no quedan modelos inactivos que liberar, no hay una «RAM vacía» que Studio pueda fabricar: toca revisar procesos externos o solicitar autorización para reducir parámetros/cambiar de receta. **Nunca se borran los pesos para resolver una falta de RAM/VRAM.** Una descarga explícita solo a disco sigue siendo una operación distinta y autorizada por el usuario.

## Límites y validación

Esto es una protección cooperativa y preventiva, **no un límite de memoria impuesto por el sistema operativo ni una garantía de cero OOM**. Una sola asignación nativa grande, un kernel, un motor externo sin callbacks, hilos internos ajenos al contexto o un proceso externo pueden adelantarse al siguiente punto de control. Un vigilante Python no puede cancelar con seguridad cualquier llamada C++ bloqueada. No se certifica aquí la inferencia Wan 14B en la L4 real.

Las pruebas automatizadas verifican cuotas/ancestros, reservas, pila completa, LRU, protección y concurrencia de propietarios, paradas en hooks/callbacks, cargas parciales, limpieza de audio/ONNX, errores y continuidad del servidor. No descargan pesos ni simulan una validación CUDA.

```bash
python -m pytest -q
python -m compileall -q app.py studio tests
node --check studio/static/app.js
```

Para aplicar el código en una Colab abierta, detener **solo el servidor**, actualizar la rama mediante `git pull --ff-only` y arrancar un proceso nuevo con el notebook actualizado. No recargar solo `memory_manager` dentro de un servidor activo: otros motores conservarían referencias antiguas. Ver [actualización y rescate sin perder pesos](COLAB_RECOVERY.md).
