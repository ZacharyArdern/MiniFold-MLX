"""Top-level minifoldx dispatcher.

Peeks at argv for --colab / --mlx before any argument parser runs,
then delegates to the appropriate backend with the flag stripped.

    minifoldx seqs.fasta [mlx options...]       # MLX (default)
    minifoldx --mlx seqs.fasta [options...]     # MLX (explicit)
    minifoldx --colab run seqs.fasta [options...] # Colab
    minifoldx --colab fetch JOB_ID
    minifoldx --colab list
"""

import sys


def main() -> None:
    argv = sys.argv[1:]

    if "--colab" in argv:
        argv.remove("--colab")
        argv = [a for a in argv if a != "--mlx"]
        sys.argv = [sys.argv[0]] + argv
        from minifold_mlx.colab.cli import main as colab_main
        colab_main()
    else:
        argv = [a for a in argv if a != "--mlx"]
        sys.argv = [sys.argv[0]] + argv
        from minifold_mlx.cli import main as mlx_main
        mlx_main()
