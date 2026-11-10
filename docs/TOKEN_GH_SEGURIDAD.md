# Tu token de GitHub: qué riesgo tiene y cómo quitárselo

Documento corto, sin adornos, para que puedas decidir.

## Lo único que necesita el canal

| Permiso | Para qué | Alternativa |
|---|---|---|
| `Contents: Read and write` **solo en `zurdonman/Studio-ai`** | publicar el resultado de cada orden en la rama `studio-channel` | ninguna: sin escritura no hay resultados |

Todo lo demás debe quedar **desmarcado**: `Actions`, `Workflows`, `Administration`, `Secrets`,
`Webhooks`, `Packages`, `Issues`, `Pull requests`, `Gists`… y sobre todo **no** marcar `All repositories`.

## Dónde ponerlo

**Opción A, Colab Secrets (🔑).** El secreto vive en tu cuenta de Google, no en la máquina de Colab. Se
configura una vez y sobrevive a reinicios, a `rm -rf /content` y a sesiones nuevas. Puedes pulsar
«Revocar» cuando quieras y deja de existir al instante.

**Opción B, cada sesión.** Guardarlo en `studio_data/state/github_channel.json` o pegarlo en el panel MCP.
Desaparece al cerrar la máquina, pero lo repites cada vez.

Si no te fías del almacenamiento, usa la B: no es más segura, es igual de segura pero más incómoda. El
token es el mismo.

## Qué pasa si se filtra

Con el canal público, el riesgo real es este:

1. Alguien con el token puede **escribir en las ramas del repositorio**. El repo es público: el daño
   visible es contenido falso, no robo de datos.
2. Puede **empujar a `main`** si el token tiene escritura en ese repo. No hay protección de rama
   configurada (lo he comprobado: la API devuelve 403, no hay regla). *Mitigación:* activa
   «Require a pull request before merging» en `Settings → Branches`.
3. **Ninguna orden se ejecuta sin que la veas**: cada una aparece en el panel y en el log de la celda
   antes de ejecutarse.
4. El token **no da acceso a tu Colab**, a tu Google Drive ni a tu cuenta de Google. Solo a GitHub.

## Cómo revocarlo (10 segundos)

`github.com → Settings → Developer settings → Personal access tokens →` tu token → **Delete**.
A partir de ese momento el canal se queda en «solo lectura» y no ejecuta nada más.

## Límite que la app impone por su cuenta

- Lista cerrada de verbos: **sin shell**, sin descargas de modelos, sin borrar pesos.
- Argumentos con credenciales rechazados.
- Una orden por `id`, con registro local: nunca se repite, aunque falle la publicación.
- Sin token **no se ejecuta nada** (evita trabajos cuya salida nadie podría leer).
