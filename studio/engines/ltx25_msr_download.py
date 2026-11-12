"""Download and hash the pinned LTX 2.5 MSR V2 adapter."""
from pathlib import Path
import sys
from studio.engines.ltx25_msr import LORA
from studio.engines.ltx_gguf_download import main

if __name__ == '__main__':
    main(Path(sys.argv[1]), (LORA,))
