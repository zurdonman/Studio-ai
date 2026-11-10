# Canal de control · Studio IA

Buzón de mensajes entre la instancia de Studio IA (tu celda de Colab) y un agente externo
que **no puede abrir conexiones hacia tu túnel** (arena.ai: su egreso solo permite
`api.github.com`, `pypi.org` y `registry.npmjs.org`).

## La idea en una línea

El agente deja órdenes en un fichero JSON de este repositorio; tu Studio IA las lee, las
ejecuta y publica los resultados en el mismo fichero. Tú no copias ni pegas nada.

## Dónde está el buzón

| | |
|---|---|
| Repositorio | `zurdonman/Studio-ai` (público) |
| Rama | `studio-channel` (solo contiene este estado, no código) |
| Fichero | `channel/agent_state.json` |
| Lectura | **sin credenciales** (repo público) |
| Escritura | token de GitHub con `Contents: read/write` |

```json
{
  "schema": 1,
  "agent": { "orders": [
      {"id": "a1", "verb": "get_state", "args": {"include_recent_assets": false}}
  ]},
  "app": { "executed_ids": ["a1"], "results": [], "updated": "2026-11-05T10:00:00Z" }
}
```

El agente escribe `agent.orders`; la app escribe `app.results` y `app.executed_ids`.
Cada orden se ejecuta **una sola vez** por `id`.

## Verbos permitidos (lista cerrada)

    get_state · get_workflow_preset · get_work_state · answer_preference
    generate_image · generate_video · generate_speech · generate_ambience
    analyze_coherence · lipsync · render_montage · wait_for_job · export_session_log

Nada de shell, descargas de modelos ni borrado de pesos: esas herramientas **no existen**
en el canal. Los argumentos con credenciales (`token`, `password`, `api_key`…) se rechazan.

## Cómo se usa

1. Abre el panel **MCP** de la interfaz → «🔗 Canal para agente arena.ai».
2. Pega un token de GitHub con `Contents: read/write` (se guarda en `studio_data/state`,
   nunca en el repositorio) y pulsa **Encender**.
3. El panel muestra el punto de encuentro y un botón para copiarlo. **Solo hace falta
   pasárselo al agente la primera vez**: es un sitio fijo, no cambia entre arranques.

La app sondea cada 20 s. Si prefieres arrancarlo a mano fuera de la interfaz:

```bash
python3 tools/mcp_github_channel.py --poll 20 --backend state
```

## Por qué una rama aparte y no un gist

El token de una GitHub App no puede crear gists, pero sí escribir contenido en el
repositorio. Una rama dedicada mantiene el buzón fuera de `main` (no dispara despliegues ni
ensucia el historial de código) y su lectura es anónima, así que la app puede leer órdenes
**sin token** y solo necesita credenciales para publicar resultados.

## Límite honesto

Este buzón no es un canal en tiempo real: como mucho hay 20 s de latencia, y si la sesión
de Colab muere el agente deja de recibir resultados. Para saber que está viva, la app
publica un latido en `app.updated`.
