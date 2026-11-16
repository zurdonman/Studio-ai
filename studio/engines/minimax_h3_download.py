"""Download and verify the pinned H3 manifest in an isolated process."""
import sys
from pathlib import Path
from studio.engines.ltx_gguf_download import main
from studio.engines.minimax_h3 import files_for

if __name__ == '__main__':
    main(Path(sys.argv[1]), files_for(sys.argv[2]))
