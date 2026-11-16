import sys
from pathlib import Path
from studio.engines.ltx_motion import UNION
from studio.engines.ltx_gguf_download import main
if __name__ == '__main__':
    main(Path(sys.argv[1]), (UNION,))
