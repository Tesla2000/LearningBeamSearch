import os
from pathlib import Path

if __name__ == "__main__":
    for path in Path("output_rl_models/20").glob("*.pth"):
        os.rename(path, Path('output_rl_models/20/10').joinpath(path.name.rpartition("_")[0]).with_suffix(".pth"))
