# Canal de control por GitHub

El agente no puede abrir conexiones hacia tu Colab (su egreso solo permite unos pocos dominios,
y `trycloudflare.com` no está entre ellos). `api.github.com` sí lo está, así que este canal invierte
la dirección: el agente **escribe órdenes en el repo** y tu Colab las **lee y ejecuta en proceso**.

## Cómo se usa

En la celda de Colab, con un token de GitHub que pueda escribir en este repo:

```bash
!GH_TOKEN=ghp_xxx python3 /content/studio/tools/mcp_github_channel.py --poll 20
```

El agente publica órdenes en `channel/ordenes.json`; el canal las ejecuta en orden y publica el
resultado en `state/channel/resultados.json`, que el agente lee con un GET normal.

## Límites deliberados

- Lista cerrada de verbos (`ALLOWED_VERBS`): nada de shell, de descargas de modelos ni de borrado de pesos.
- Una orden se ejecuta una sola vez por `id`, aunque el fichero cambie.
- Los argumentos con `token`, `password`, `api_key` se rechazan.
- Si una herramienta falla, el error se reporta y el canal sigue vivo.
