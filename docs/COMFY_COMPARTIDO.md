# Comfy compartido (experimental)

Studio intenta mantener un proceso privado de ComfyUI entre trabajos de vídeo. Wan/SCAIL, LTX 2.3, LTX 2.5 y FILM comparten el proceso cuando usan el mismo runtime y la misma política de memoria. En un cambio de modelo se ejecuta un nodo privado de limpieza en el hilo de Comfy, se vacían las cachés del grafo y se liberan los pesos anteriores antes de reasignar el proceso. El vigilante de memoria sigue activo. La caché de pesos solo se intenta conservar al repetir la misma receta.

Los nodos nuevos pueden requerir un reinicio; también se reinicia si cambia la política de memoria, no se confirma la limpieza, falla el proceso, se cancela o se requiere otro runtime. El registro indica el PID, si el proceso se ha reutilizado y cuándo se recurre a otro. Los modelos descargados no se borran. La opción de residencia de LTX se puede desactivar; en transferencia de movimiento y FILM se intenta automáticamente.

Gemma 4 GGUF intenta primero aplicar su parche verificado al runtime gestionado compartido y comprobar los nodos. Si no encaja, se usa el runtime aislado fijado; si una inferencia compartida devuelve un error explícito de arquitectura no soportada o nodos ausentes, se reintenta una vez en el aislado. No se reintenta automáticamente por cualquier error, por cancelación o por falta de memoria. Los conflictos con archivos modificados por el usuario no se fuerzan ni se borran.

Las pruebas automáticas usan simulaciones y un Comfy real en CPU con nodos de prueba; no sustituyen pruebas de cada combinación de modelos y GPU. La compatibilidad visual y las mejoras de tiempo en L4 no están garantizadas. Los procesos aislados escuchan únicamente en loopback dentro del entorno de Studio.
