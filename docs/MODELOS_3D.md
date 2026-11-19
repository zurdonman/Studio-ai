# Modelos 3D locales (experimental)

La pestaña **Modelos 3D** contiene rutas experimentales para SF3D, TripoSR, Hunyuan, TRELLIS.2 e Idol. Se han revisado sus contratos de CLI; **no se ha probado inferencia con pesos reales ni GPU**. Todos los motores indicados, incluidas las variantes ONNX, disponen de recetas de preparación automática experimentales. Configurar rutas manuales solo comprueba que existan archivos, no que funcionen. Se usan CLI oficiales y adaptadores Python o trellis.cpp, con entornos independientes por sus dependencias nativas; estas rutas no usan ComfyUI.

## Salidas y uso

- Stable Fast 3D: GLB con malla, UV, textura y parámetros de material. Pesos oficiales F32 (~4,02 GB); el ejemplo oficial usa autocast BF16 en CUDA. El CLI detecta transparencia y, en caso contrario, usa rembg para retirar fondo. Es posible que rembg descargue su propio modelo la primera vez. Su modelo está gated: debes solicitar acceso y aceptar su licencia en Hugging Face. Revisa la Stability AI Community License, incluidas las condiciones de uso comercial. [Modelo y licencia](https://huggingface.co/stabilityai/stable-fast-3d), [código e instalación](https://github.com/Stability-AI/stable-fast-3d).
- TripoSR: GLB con **colores de vértice**, no atlas UV en esta receta. Su CLI usa rembg y puede descargar pesos adicionales. Revisa [modelo](https://huggingface.co/stabilityai/TripoSR), [código MIT e instalación](https://github.com/VAST-AI-Research/TripoSR) y licencias de dependencias.
- Usa imagen de un único objeto, clara, centrada y sin oclusiones. Las partes ocultas son inferidas, no reconstruidas fielmente. Las cifras del catálogo son referencias aproximadas, no requisitos garantizados. El resultado se encuentra en la pestaña y en el ZIP de exportación; puedes abrirlo en Blender u otro visor GLB. No hay visor 3D integrado.

## Preparación manual

Elige un sistema con PyTorch adecuado a tu CUDA/GPU y sigue las instrucciones oficiales para cada proyecto. **No instales las dependencias en el entorno principal de Studio**: las versiones y extensiones nativas pueden entrar en conflicto. SF3D necesita compilar texture_baker y uv_unwrapper; TripoSR necesita torchmcubes y otras dependencias. Esta integración exige una GPU NVIDIA con medición de VRAM disponible: el protector detiene el proceso si no puede medirla. Aunque los proyectos originales admitan otros dispositivos, CPU/MPS no están soportados por este adaptador de Studio; Windows no se ha validado. Comprueba espacio en disco, compilador, CUDA y compatibilidad de PyTorch. Las versiones de referencia del código auditado son SF3D `ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2` y TripoSR `107cefdc244c39106fa830359024f6a2f1c78871`; el software externo instalado es responsabilidad del administrador.

Para cada uno, configura tres **variables de entorno del servidor**, con rutas absolutas, antes de arrancar Studio (ejemplos orientativos, sustituir por tus rutas reales):

```bash
export STUDIO_SF3D_REPO=/ruta/stable-fast-3d
export STUDIO_SF3D_PYTHON=/ruta/entorno-sf3d/bin/python
export STUDIO_SF3D_WEIGHTS=/ruta/pesos-sf3d
export STUDIO_TRIPOSR_REPO=/ruta/TripoSR
export STUDIO_TRIPOSR_PYTHON=/ruta/entorno-triposr/bin/python
export STUDIO_TRIPOSR_WEIGHTS=/ruta/pesos-triposr
```

El repositorio debe contener `run.py`; el directorio de pesos SF3D debe contener `config.yaml` y `model.safetensors` del repositorio oficial, y el de TripoSR `config.yaml` y `model.ckpt`. Descárgalos manualmente desde las páginas oficiales después de cumplir sus requisitos de acceso. Si se precisa autenticación, utiliza el mecanismo oficial local de Hugging Face o el apartado seguro de Studio; no incluyas tokens en estas variables de rutas. Los repositorios se ejecutan con los permisos del servidor: configura únicamente código en el que confíes. La API no permite elegir un ejecutable o ruta de pesos.

La ejecución tiene cola y cancelación, serializa el acceso GPU con otros motores gestionados y usa el protector de memoria para el proceso hijo. Genera en un directorio temporal privado, comprueba la estructura básica GLB y publica el resultado únicamente al terminar. No se garantiza que una comprobación de cabecera demuestre calidad, textura correcta o geometría útil. Un fallo o cancelación no publica salida parcial. Si falla, revisa el monitor, permisos, compatibilidad CUDA, dependencias, acceso a pesos y espacio. El proceso puede requerir conexión para rembg u otras dependencias; no se promete funcionamiento sin conexión en la primera ejecución.

## Otros modelos del catálogo

Estos adaptadores son experimentales y requieren depuración; Idol y Hunyuan 2.0/2mini admiten preparación automática, así como TRELLIS.2 GGUF q4/q8; Hunyuan 2.1 también incluye preparación experimental. La licencia actual de [Hunyuan 2.0/2mini](https://huggingface.co/tencent/Hunyuan3D-2mini/blob/main/LICENSE) y [2.1](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE) excluye expresamente UE, Reino Unido y Corea del Sur; consulta los términos para tu situación. ComfyUI nativo genera geometría Hunyuan, no textura: Paint requiere otra etapa. TRELLIS.2 requiere su propio runtime; sus [GGUF q4/q8 comunitarios](https://huggingface.co/ilintar/trellis2-gguf) son archivos por componente y no son compatibles automáticamente con Comfy. DINOv3 tiene licencia y acceso independientes. Idol produce Gaussian splats PLY, no una malla, y requiere DINOv3 y FLUX.2 VAE con términos propios. Las variantes ONNX tienen un runtime separado experimental. No confundas variante de arquitectura con cuantización.

## Ampliación: Hunyuan, TRELLIS.2 e Idol

La pestaña incluye ahora **adaptadores Python experimentales** para todos los modelos anteriores. No se han probado con sus dependencias nativas, pesos reales ni GPU. Preparar cada repositorio y su propio entorno siguiendo los enlaces oficiales, incluyendo extensiones compiladas, autenticación y licencias de dependencias. No basta con poner las variables: la indicación «rutas configuradas» no certifica que el entorno funcione. La primera ejecución puede descargar modelos de Hugging Face y otros repositorios. La pestaña permite cancelar y el proceso aislado se detiene ante presión de memoria medida; esto no garantiza evitar todos los OOM.

| Motor | Repositorio (variable `_REPO`) | Python (`_PYTHON`) | `_WEIGHTS` opcional (predeterminado) |
|---|---|---|---|
| Hunyuan 2 | `STUDIO_HUNYUAN2_REPO` | `STUDIO_HUNYUAN2_PYTHON` | `STUDIO_HUNYUAN2_WEIGHTS` (`tencent/Hunyuan3D-2`) |
| Hunyuan mini | `STUDIO_HUNYUANMINI_REPO` (usa el repositorio Hunyuan3D-2) | `STUDIO_HUNYUANMINI_PYTHON` | `STUDIO_HUNYUANMINI_WEIGHTS` (`tencent/Hunyuan3D-2mini`) |
| Hunyuan 2.1 | `STUDIO_HUNYUAN21_REPO` | `STUDIO_HUNYUAN21_PYTHON` | `STUDIO_HUNYUAN21_WEIGHTS` (`tencent/Hunyuan3D-2.1`) |
| TRELLIS.2 oficial | `STUDIO_TRELLIS2_REPO` | `STUDIO_TRELLIS2_PYTHON` | `STUDIO_TRELLIS2_WEIGHTS` (`microsoft/TRELLIS.2-4B`) |
| Idol | `STUDIO_IDOL_REPO` | `STUDIO_IDOL_PYTHON` | `STUDIO_IDOL_WEIGHTS` (`speridlabs/idol`) |

Las variables de repositorio y Python son rutas absolutas locales; los pesos pueden ser ruta local o el ID predeterminado del modelo. El código externo corre con los permisos del servidor. Revisa las licencias, y solo configura código y checkpoints de confianza. El checkbox confirma derechos: **no anula las exclusiones territoriales de Tencent**. Hunyuan usa eliminación de fondo cuando la imagen no tiene alfa; «Paint» es una segunda etapa y consume modelos, memoria y dependencias adicionales. Para mini, Paint usa el modelo completo 2.0, no un supuesto modelo mini con textura. Se puede configurar `STUDIO_HUNYUANMINI_PAINT_WEIGHTS`, `STUDIO_HUNYUAN2_PAINT_WEIGHTS` o `STUDIO_HUNYUAN21_PAINT_WEIGHTS`; 2.1 requiere además RealESRGAN y los componentes nativos en las rutas esperadas por el proyecto oficial. Sin Paint, la salida es solo geometría. Las opciones base/fast/turbo son variantes FP16 oficiales, no GGUF.

Para TRELLIS.2 comunitario q4/q8, compila/configura por separado [trellis.cpp](https://github.com/pwilkin/trellis.cpp) y usa los componentes correspondientes de [ilintar/trellis2-gguf](https://huggingface.co/ilintar/trellis2-gguf): `STUDIO_TRELLIS_GGUF_EXE` apunta a `trellis-cli` y `STUDIO_TRELLIS_GGUF_Q4_WEIGHTS` / `STUDIO_TRELLIS_GGUF_Q8_WEIGHTS` a sus carpetas completas. No son compatibles con el runtime Python oficial. Además de los flow cuantizados se necesitan los demás componentes de cada carpeta y licencias de DINOv3. Las variantes ONNX usan otro runtime y no los adaptadores oficiales.

Consulta también [Mundos 3D](MUNDOS_3D.md). Los modelos se producen como GLB salvo Idol, que produce Gaussian splats PLY, no malla. Para convertir splats en mallas se necesita un proceso externo y puede perderse información.

### Preparación automática experimental

En la pestaña de modelos 3D, selecciona un motor compatible, acepta las licencias y pulsa **Preparar motor seleccionado**; también se prepara al pulsar Generar si aún falta el entorno. El backend descarga código y dependencias en entornos separados en `DATA_ROOT/runtimes/scene3d`, y los pesos principales en `DATA_ROOT/models/scene3d`. La instalación y la compilación pueden tardar y consumir mucho disco; el monitor muestra el estado, los errores y permite cancelar. Se necesita un Python con `venv`, compilador y, para extensiones GPU, CUDA toolkit compatible y driver NVIDIA; no se instalan paquetes del sistema con sudo. Algunos modelos auxiliares se descargan al generar. SF3D necesita aceptar primero las condiciones en Hugging Face y configurar el token en Studio; Idol además exige acceso a DINOv3 y FLUX.2 VAE. No se ha validado inferencia real GPU. El marcador solo certifica que la instalación y los imports indicados terminaron y que los pesos principales están presentes, no que el motor vaya a funcionar en tu GPU. Las variables manuales tienen prioridad y una configuración manual parcial no se mezcla con la instalación gestionada.

La preparación automática experimental incluye **Hunyuan 2.1**, TRELLIS.2 oficial BF16/FP16 y las dos variantes ONNX. Su instalación real y su inferencia aún no se han validado con GPU. Hunyuan 2.1 descarga también RealESRGAN y compila extensiones nativas; pueden necesitarse dependencias adicionales.

Hunyuan 2.0 y 2mini permiten preparar base, fast o turbo; se instalan también las extensiones y pesos Paint (descargas grandes). TRELLIS.2 oficial compila FlashAttention, CuMesh, FlexGEMM y otras extensiones CUDA; necesita Python 3.10 y al menos 32 GiB libres para comenzar (el uso real puede ser mayor). Descarga los componentes principales y puede descargar pesos adicionales de DINOv3, BiRefNet y TRELLIS al generar. TRELLIS.2 q4/q8 compila trellis.cpp con CUDA y descarga todos sus componentes GGUF. Hay que preparar cada variante elegida; el botón de preparación y el monitor indican el estado. Se requieren compilador, CMake para GGUF, CUDA toolkit compatible y espacio adicional. No se ha validado con GPU. Los pesos se cargan en VRAM al generar, no permanecen cargados después de preparar.

### Variantes ONNX experimentales

Selecciona **ONNX CUDA** en SF3D o Hunyuan 2mini, acepta licencias y pulsa Preparar o Generar. Son grafos comunitarios ejecutados en el **servidor** mediante ONNX Runtime GPU y CUDA/cuDNN 12, no WebGPU del navegador. Los modelos se descargan con revisiones fijadas y SHA-256. Usa una imagen con fondo ya transparente o limpio; el preprocesado ONNX no ejecuta rembg. El resultado es una **malla aproximada sin textura** mediante muestreo de campo y marching cubes; no reproduce la extracción de tetraedros, deformación ni materiales del pipeline oficial de SF3D. Hunyuan usa cinco pasos Euler fijos y su forma mini turbo: ni el control de pasos ni Paint cambian esta variante. Requiere bastante RAM/VRAM y ORT contrib ops; el guardián exige GPU NVIDIA con VRAM medible. Los operadores pueden caer parcialmente en CPU. La disponibilidad de proveedores no garantiza que ejecuten los grafos. La implementación está probada solo con sesiones sintéticas, no con los pesos reales o GPU. En caso de incompatibilidad u OOM consulta el monitor. Revisa también la licencia original: la variante de Hunyuan **excluye UE, Reino Unido y Corea del Sur**; la de SF3D está sujeta a las condiciones de Stability AI.
