from pathlib import Path
from ud_boundary_tokenization.experiment import run

if __name__ == "__main__":
    run(output_dir=Path("results"), data_dir=Path("data/ud"), ref="r2.18")
