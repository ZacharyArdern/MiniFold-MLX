"""MiniFold-Colab CLI — run MiniFold predictions on Google Colab GPUs.

Usage
-----
    minifold-colab run seqs.fasta --gpu A100
    minifold-colab fetch JOB_ID
    minifold-colab list

Requirements
------------
    brew install rclone   (with gdrive remote already configured)
    colab CLI on PATH     (pip install colab-cli)

Drive layout (under My Drive):
    Colab_Data/minifold_colab/weights/   — cached model weights (shared across runs)
    Colab_Data/minifold_colab/jobs/<id>/ — per-job results
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import click

from ._jobs import load_jobs, save_job
from ._runner import check_deps, colab, rclone, run_prediction

GPU_CHOICES = ["T4", "L4", "G4", "H100", "A100"]


@click.group()
def main():
    """MiniFold on Google Colab — predict protein structures using Colab GPUs."""


@main.command()
@click.argument("fasta", type=click.Path(exists=True))
@click.option("--gpu",  default="A100", type=click.Choice(GPU_CHOICES), show_default=True)
@click.option("--model-size", default="48L", type=click.Choice(["12L", "48L"]), show_default=True)
@click.option("--token-per-batch", default=2048, show_default=True)
@click.option("--num-recycling",   default=3,    show_default=True)
@click.option("--timeout", default=3600, show_default=True,
              help="Max seconds to wait for prediction to complete.")
@click.option("--keep", is_flag=True,
              help="Leave the Colab session running after prediction (for debugging).")
@click.option("--int8-esm2", "int8_esm2", is_flag=True,
              help="Quantize ESM2 to int8 via bitsandbytes (~3× smaller, saves GPU memory).")
@click.option("--triton-kernels", "triton_kernels", is_flag=True,
              help="Use Triton-fused MLP/gating kernels for faster inference on A100/H100.")
def run(
    fasta: str,
    gpu: str,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    timeout: int,
    keep: bool,
    int8_esm2: bool,
    triton_kernels: bool,
) -> None:
    """Run a MiniFold prediction on a fresh Colab GPU session."""
    check_deps()

    fasta_path = Path(fasta).expanduser().resolve()
    job_id     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session    = f"minifold_{job_id}"

    click.echo(f"Job ID  : {job_id}")
    click.echo(f"FASTA   : {fasta_path.name}")
    click.echo(f"GPU     : {gpu}  |  model: {model_size}  |  recycling: {num_recycling}")
    click.echo(f"ESM2    : {'int8 / bitsandbytes' if int8_esm2 else 'full precision'}")
    click.echo(f"Kernels : {'Triton (fused MLP/gating)' if triton_kernels else 'standard PyTorch'}")
    click.echo("")

    click.echo(f"[1/5] Creating {gpu} session '{session}' ...")
    colab("new", "--gpu", gpu, "-s", session)

    out_dir = None
    try:
        out_dir = run_prediction(
            fasta_path, job_id, session, model_size,
            token_per_batch, num_recycling, timeout, int8_esm2, triton_kernels,
        )

        pdbs = list(out_dir.rglob("*.pdb"))
        click.echo("")
        click.echo(f"Done. {len(pdbs)} structure(s) in {out_dir}/")
        for p in sorted(pdbs):
            click.echo(f"  {p.relative_to(out_dir)}")

    finally:
        if not keep:
            click.echo(f"\nStopping session '{session}' ...")
            subprocess.run(["colab", "stop", "-s", session])

    if out_dir is not None:
        save_job({
            "job_id":        job_id,
            "fasta":         str(fasta_path),
            "gpu":           gpu,
            "model_size":    model_size,
            "num_recycling": num_recycling,
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "results":       str(out_dir),
        })


@main.command()
@click.argument("job_id")
@click.option("--out-dir", default="./minifold_results", show_default=True)
def fetch(job_id: str, out_dir: str) -> None:
    """Re-download results for JOB_ID from Google Drive."""
    check_deps()

    from ._runner import JOBS_REMOTE
    dest = Path(out_dir) / job_id
    dest.mkdir(parents=True, exist_ok=True)
    click.echo(f"Downloading {job_id} → {dest}/")
    rclone("copy", f"{JOBS_REMOTE}/{job_id}/outputs", str(dest), "--progress")

    pdbs = list(dest.rglob("*.pdb"))
    click.echo(f"{len(pdbs)} file(s) downloaded.")


@main.command(name="list")
def list_jobs() -> None:
    """List all local prediction jobs."""
    jobs = load_jobs()
    if not jobs:
        click.echo("No jobs yet. Run:  minifoldx --colab run seqs.fasta")
        return

    click.echo(f"{'JOB ID':<22} {'FASTA':<28} {'GPU':<6} {'MODEL':<5} CREATED")
    click.echo("-" * 82)
    for jid, info in sorted(jobs.items()):
        fasta_name = Path(info["fasta"]).name
        created    = info.get("created_at", "")[:19].replace("T", " ")
        click.echo(f"{jid:<22} {fasta_name:<28} {info.get('gpu','?'):<6} "
                   f"{info['model_size']:<5} {created}")
