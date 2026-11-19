# Sonido para vídeo local (experimental)

El apartado **Sonido para vídeo · V2A** incorpora una primera receta experimental de MMAudio V2 para Foley/ambiente, no para doblaje hablado controlado. Ejecuta el modelo en la GPU local, sin servicio de inferencia remoto; la preparación descarga el código, los pesos y sus dependencias tras autorización. No se han probado ni la descarga integral ni la inferencia en L4/G4. Errores y cancelación se muestran en el monitor. La preparación verifica PyTorch, torchvision y torchaudio CUDA y, si faltan o no son compatibles, intenta instalarlos en el entorno aislado; depende de que haya ruedas compatibles, un controlador NVIDIA apropiado y acceso a sus repositorios. No se declara preparado si falta un componente.

La implementación procesa un clip de 2 a 8 segundos (máximo 200 MiB); vuelve a codificar el vídeo para el resultado y lo guarda como un nuevo MP4. El archivo de entrada no se modifica. Si ya hay audio, se requiere una autorización independiente para sustituirlo en la copia. La calidad y la sincronización no están garantizadas; el modelo puede producir sonidos parecidos a voces ininteligibles, no diálogo fiable. El vídeo se reduce y ajusta con bandas para el análisis visual, mientras la copia final mantiene las dimensiones originales. Los autores recomiendan unos 8 segundos; el tamaño y duración afectan a memoria, calidad y tiempo.

## Licencias

- Pesos [MMAudio](https://huggingface.co/hkchengrex/MMAudio/blob/main/README.md): CC BY-NC 4.0; código [MMAudio](https://github.com/hkchengrex/MMAudio) MIT con componentes de terceros, entre ellos [EDM2](https://github.com/hkchengrex/MMAudio/blob/974010a026c731054592d8f777218bd9d85a6c24/mmaudio/ext/autoencoder/edm2_utils.py), CC BY-NC-SA 4.0.
- [Apple DFN5B CLIP](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/blob/main/LICENSE): exclusivamente investigación científica no comercial o desarrollo académico en los términos de esa licencia. El uso personal no es automáticamente investigación científica. No se debe ejecutar esta receta para un uso personal ordinario si no se cumple la finalidad indicada.
- [NVIDIA BigVGAN v2](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/blob/main/LICENSE): MIT; los demás componentes mantienen sus respectivos términos.

La casilla de licencia no sustituye la lectura ni cambia las condiciones. La primera preparación descarga aproximadamente 10,7 GB decimales de pesos, además del entorno y espacio temporal; se verifica cada archivo frente a la revisión fijada. Si un archivo existente no coincide, no se sobrescribe automáticamente.

## Aún pendientes

ControlFoley, PrismAudio/ThinkSound y V2S/doblaje no están integrados. ControlFoley tiene pesos CC BY-NC 4.0 y su implementación utiliza dependencias con licencias propias. PrismAudio declara investigación/educación y prohíbe el uso comercial sin autorización. No se anuncia compatibilidad GPU ni instalación automática de esos motores.

Para explorar un flujo local apto para uso personal ordinario, conviene investigar [LTX-2.3 Foley V2A LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Foley-V2A), sujeto a su licencia comunitaria, acceso condicionado en Hugging Face y términos de componentes/datos; el workflow oficial y sus requisitos aún no están integrados ni validados en esta aplicación. [AudioX-Turbo](https://github.com/NoizAI/AudioX-Turbo) es otra posibilidad no comercial por estudiar, con licencias propias para sus dependencias. Ninguna de las dos opciones está lista en Studio.
