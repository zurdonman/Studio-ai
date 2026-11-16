"""Download only the selected pinned motion recipe."""
import sys
from pathlib import Path
from studio.engines.motion_manifest import files_for
from studio.engines.ltx_gguf_download import main

if __name__ == '__main__':
    main(Path(sys.argv[1]), files_for(sys.argv[2], sys.argv[3]))
