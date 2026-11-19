"""Download the selected fixed, checksum-verified Wan I2V recipe."""
import sys
from pathlib import Path
from studio.engines.ltx_gguf_download import main
from studio.engines.wan22_i2v_manifest import files_for

if __name__ == '__main__':
    main(Path(sys.argv[1]), files_for(sys.argv[2]), motion_cache=True)
