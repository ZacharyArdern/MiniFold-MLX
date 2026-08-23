"""MiniFold remote GPU CLI — run MiniFold on your own CUDA server via SSH.

Usage
-----
    minifoldx --remote user@host run seqs.fasta
    minifoldx --remote user@host run seqs.fasta --model 12L --rec 0
    minifoldx --remote user@host list

Requirements
------------
    ssh / scp / rsync on PATH
    Python with CUDA + pip on the remote host (conda/mamba/venv all fine)

Remote layout (on server, under ~/.cache/minifoldx/):
    code/MiniFoldX/    — cloned repo (updated each run)
    weights/           — cached model weights + ESM2
    jobs/<id>/         — per-job FASTA input + output PDBs
"""

from datetime import datetime, timezone
from pathlib import Path

import click

from ._runner import check_deps, run_remote


@click.group()
@click.option("--host", required=True, help="SSH target: user@hostname")
@click.pass_context
def main(ctx, host: str) -> None:
    """MiniFold on a remote CUDA server — predict protein structures via SSH."""
    ctx.ensure_object(dict)
    ctx.obj["host"] = host


@main.command()
@click.argument("fasta", type=click.Path(exists=True))
@click.option("--model-size", "--model", default="48L",
              type=click.Choice(["12L", "48L"]), show_default=True)
@click.option("--num-recycling", "--rec", default=3, show_default=True,
              help="Number of recycling iterations.")
@click.option("--token-per-batch", default=2048, show_default=True,
              help="Tokens per batch.")
@click.option("--out-dir", default="./minifold_results", show_default=True,
              help="Local directory to save downloaded results.")
@click.pass_context
def run(
    ctx,
    fasta: str,
    model_size: str,
    num_recycling: int,
    token_per_batch: int,
    out_dir: str,
) -> None:
    """Run a MiniFold prediction on a remote CUDA server."""
    check_deps()

    host      = ctx.obj["host"]
    fasta_path = Path(fasta).expanduser().resolve()
    job_id    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    import gzip as _gz
    _opener = _gz.open if str(fasta_path).endswith(".gz") else open
    with _opener(fasta_path, "rt") as _fh:
        n_seqs = sum(1 for line in _fh if line.startswith(">"))

    click.echo(f"Job ID  : {job_id}")
    click.echo(f"Host    : {host}")
    click.echo(f"FASTA   : {fasta_path.name}  ({n_seqs} sequences)")
    click.echo(f"Model   : {model_size}  |  recycling: {num_recycling}")
    click.echo("")

    local_out = run_remote(
        fasta_path, job_id, host,
        model_size, token_per_batch, num_recycling,
        Path(out_dir).expanduser(),
    )

    pdbs = list(local_out.rglob("*.pdb"))
    click.echo("")
    click.echo(f"Done. {len(pdbs)} structure(s) in {local_out}/")
    for p in sorted(pdbs):
        click.echo(f"  {p.relative_to(local_out)}")


@main.command(name="list")
@click.option("--host", "host_flag", default=None, hidden=True)
@click.pass_context
def list_jobs(ctx, host_flag) -> None:
    """List remote jobs stored on the server."""
    host = ctx.obj["host"]
    from ._runner import ssh_check, REMOTE_JOBS
    result = ssh_check(host, f"ls -1t {REMOTE_JOBS} 2>/dev/null || echo ''")
    jobs = [j for j in result.stdout.strip().splitlines() if j]
    if not jobs:
        click.echo(f"No jobs found on {host} ({REMOTE_JOBS})")
        return
    click.echo(f"{'JOB ID':<22} HOST")
    click.echo("-" * 40)
    for jid in jobs:
        click.echo(f"{jid:<22} {host}")
