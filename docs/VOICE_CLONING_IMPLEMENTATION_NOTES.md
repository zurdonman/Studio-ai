# Preparación de clonación de voz: Qwen3-TTS y Fish Speech

**Estado: Qwen3-TTS integrado de forma experimental; Fish Speech todavía no está integrado.** En la pestaña Voz, la muestra autorizada de 3–15 s y el texto se envían a una tarea aislada de Qwen 0.6B. La transcripción es opcional (sin ella se usa x-vector). La salida WAV se verifica, guarda en la biblioteca y se puede enviar al modo LTX GGUF «Audio completo». La muestra se prepara en un directorio temporal y no se incluye en los metadatos de salida.

La primera ejecución prepara un entorno virtual independiente y descarga la revisión fija del código y los pesos. Se comprueban los hashes de todos los archivos del modelo. El proceso Qwen se guarda en el slot TTS para reutilizarlo; al cambiar de modelo, cancelar, fallar o necesitar memoria, el proceso se descarga sin borrar archivos. Se comprueba el presupuesto antes de la instalación y antes de cargar pesos, y se monitoriza presión durante carga e inferencia; las estimaciones de 8 GiB RAM y 5 GiB VRAM no son mediciones ni garantizan evitar OOM. Requiere CUDA BF16 y una instalación PyTorch/torchaudio compatible. Nunca se asegura voz, idioma ni contenido idénticos.

Las pruebas realizadas usan mocks y audio sintético. Se ha comprobado que el código Qwen fijado se importa con transformers 4.57.3 y las dependencias de prueba, pero **no se han descargado ni ejecutado los pesos reales, ni se ha validado la inferencia en GPU/L4**. Qwen y Fish mantienen sus licencias y políticas independientes; Fish no se ofrece en la interfaz hasta que su integración y salvaguardas estén hechas. La ruta de voz de referencia directa LTX es una opción distinta y se mantiene disponible.

## APIs inspeccionadas

- Qwen3-TTS: upstream QwenLM/Qwen3-TTS commit
  `022e286b98fbec7e1e916cb940cdf532cd9f488e`. Checkpoint
  `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, revisión
  `5d83992436eae1d760afd27aff78a71d676296fc` (Apache-2.0).
  `Qwen3TTSModel.from_pretrained(...)` y `generate_voice_clone(text=...,
  language="Spanish", ref_audio=..., ref_text=...)`. Si no hay transcripción,
  usar `x_vector_only_mode=True`; puede reducir la calidad. Los dos
  safetensors principales suman aproximadamente 2,51 GB decimales, más
  configuración y dependencias. Su paquete 0.1.1 fija transformers 4.57.3 y
  accelerate 1.12.0, potencialmente incompatibles con otros motores de Studio.
- Fish Speech v1.5.1, commit `58046eaa1a4cefb0c8cc3a3a667b34186ea02dde`,
  no la API de S2. El modelo `fishaudio/fish-speech-1.5` tiene licencia
  CC-BY-NC-SA-4.0 (sin uso comercial; revisar todas las condiciones en su
  ficha), revisión inspeccionada `275a984d33c33659e39eed41ff5bcd6e67517f4c`.
  La inferencia utiliza `TTSInferenceEngine`, la cola text2semantic, el
  decodificador Firefly y `ServeTTSRequest` con `ServeReferenceAudio(audio=bytes,
  text=transcripción)`. Necesita transcripción fiel. Los archivos `.pth` son
  pickle: upstream usa `torch.load(..., weights_only=True)` para ambos pesos;
  no rebajar esa protección. El LFS de `model.pth` declara SHA-256
  `918dc960372cc1b77bbafb14c48ef7a1634ecf75d4eb85b78607223b780d6001` y el
  decodificador `01b81dbf753224a156c3fe139b88bf0b9a0f54b11bee864f95e66511c3ccd754`.
  No se ha descargado ni comprobado aquí que esos ficheros carguen con la
  versión de PyTorch disponible. `launch_thread_safe_queue` espera sin timeout
  a la inicialización del hilo; si esta falla, puede bloquearse indefinidamente.

## Trabajo pendiente para Fish y validación real

1. Aislar dependencias de Fish, sin alterar el entorno compartido;
   fijar código/pesos y comprobar integridad y licencia/acceso HF. Evitar la
   espera indefinida del cargador Fish y conservar `weights_only=True`.
2. Conectar Fish a UI/API y trabajos: resolver la referencia con la política de medios
   de Studio, solicitar consentimiento, transcripción obligatoria para Fish,
   idioma y texto, y usar la utilidad de preparación temporal. No guardar la
   muestra en metadatos públicos; limpiar temporales incluso tras errores.
3. Integrar el proceso Fish con el slot TTS y el gestor de memoria: conservar
   el proceso cargado entre llamadas, liberar su memoria al descargar el slot,
   respetar cancelación y vigilancia también durante cargas e inferencia. La
   protección cooperativa no impide todos los OOM de operaciones nativas.
4. Guardar el WAV de Fish en la biblioteca de audio y probar descarga,
   inferencia real, errores, cancelación y reutilización con ambos motores.
   Medir memoria y rendimiento en GPU L4 antes de afirmar que caben o que la
   calidad es aceptable.

La política local provisional acepta muestras de 3–15 s, preferentemente unos
5 s, de una voz cuyo titular haya autorizado su uso. No son límites oficiales
de ambos modelos. En este entorno no se ha probado inferencia ni GPU L4.
