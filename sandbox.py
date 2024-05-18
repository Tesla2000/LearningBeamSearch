import os
from pathlib import Path

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
