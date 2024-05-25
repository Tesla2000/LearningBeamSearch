import os
from pathlib import Path

from pixels2svg import pixels2svg

from Config import Config

if __name__ == "__main__":
    for path in Path("output_rl_results").iterdir():
        if path.is_dir():
            continue
        name, beta, tasks = path.name.split("_")
        if int(beta) <= 5:
            continue
        folder = Path("output_rl_results") / tasks / "10" / "time" / beta
        folder.mkdir(exist_ok=True, parents=True)
        os.rename(path, folder.joinpath(name))
    for path in Config.PLOTS.glob("*.png"):
        print(path)
        svg_path = Config.SVG_PLOTS.joinpath(path.with_suffix('.svg').name)
        if svg_path.exists():
            continue
        pixels2svg(path, svg_path)
