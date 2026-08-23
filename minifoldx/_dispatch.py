"""Top-level minifoldx dispatcher.

Peeks at argv for --colab / --remote / --cpu / --mlx before any argument parser runs,
then delegates to the appropriate backend with the flag stripped.

    minifoldx seqs.fasta [mlx options...]              # MLX / Apple Silicon (default)
    minifoldx --mlx seqs.fasta [options...]            # MLX (explicit)
    minifoldx --colab run seqs.fasta [options...]      # Google Colab GPU
    minifoldx --colab fetch JOB_ID
    minifoldx --colab list
    minifoldx --remote user@host run seqs.fasta [...]  # own CUDA server via SSH
    minifoldx --remote user@host list
    minifoldx --cpu seqs.fasta [options...]            # CPU (all cores, PyTorch)
"""

import sys


def main() -> None:
    argv = sys.argv[1:]

    if "--cpu" in argv:
        argv = [a for a in argv if a not in ("--mlx", "--colab")]
        sys.argv = [sys.argv[0]] + argv
        from minifoldx.pytorch.predict import predict as cpu_predict
        cpu_predict()
    elif "--remote" in argv:
        idx = argv.index("--remote")
        if idx + 1 >= len(argv):
            print("Error: --remote requires user@host argument", file=sys.stderr)
            sys.exit(1)
        host = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
        argv = [a for a in argv if a not in ("--mlx", "--colab")]
        sys.argv = [sys.argv[0], "--host", host] + argv
        from minifoldx.remote.cli import main as remote_main
        remote_main()
    elif "--colab" in argv:
        argv.remove("--colab")
        argv = [a for a in argv if a != "--mlx"]
        sys.argv = [sys.argv[0]] + argv
        from minifoldx.colab.cli import main as colab_main
        colab_main()
    else:
        argv = [a for a in argv if a != "--mlx"]
        sys.argv = [sys.argv[0]] + argv
        from minifoldx.cli import main as mlx_main
        mlx_main()
