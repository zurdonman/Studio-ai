"""Download the pinned Apache-licensed FILM checkpoint and verify SHA-256."""
import sys
from pathlib import Path
from studio.engines.freefield_film import MODEL
from studio.engines.ltx_gguf_download import main

if __name__ == '__main__':
    main(Path(sys.argv[1]), (MODEL,))
