"""Download the selected fixed, checksum-verified Wan I2V recipe."""
import sys
from pathlib import Path
from studio.engines.ltx_gguf_download import main
from studio.engines.wan22_i2v_manifest import recipe_files

if __name__ == '__main__':
    main(Path(sys.argv[1]), recipe_files(sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "i2v", sys.argv[4] if len(sys.argv)>4 else "none"), motion_cache=True)
