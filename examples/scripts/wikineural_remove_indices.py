from pathlib import Path
from string import digits

import typer


def remove_indices(input_path: Path, output_path: Path):
    """Helper function to remove the indices for the WikiNeural dataset"""
    with input_path.open() as f:
        lines = f.readlines()

    with output_path.open("w") as f:
        for line in lines:
            new_line = line if line == "\n" else line.lstrip(digits)[1:]
            f.write(new_line)


if __name__ == "__main__":
    typer.run(remove_indices)
