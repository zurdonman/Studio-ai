"""Download and verify the pinned voice adapter using the common GGUF downloader."""
from pathlib import Path
import sys
from studio.engines.ltx23_voice import LORA
from studio.engines.ltx_gguf_download import main

if __name__ == '__main__':
    main(Path(sys.argv[1]), (LORA,))
