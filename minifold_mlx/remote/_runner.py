"""SSH-based remote GPU runner for MiniFold predictions."""

import subprocess
import sys
from pathlib import Path

import click

REMOTE_BASE    = "~/.cache/minifoldx"
REMOTE_CODE    = f"{REMOTE_BASE}/code/MiniFoldX"
REMOTE_WEIGHTS = f"{REMOTE_BASE}/weights"
REMOTE_JOBS    = f"{REMOTE_BASE}/jobs"

HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"

REPO_URL = "https://github.com/ZacharyArdern/MiniFoldX.git"

_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]


def _ssh_cmd(host: str, remote_cmd: str) -> list[str]:
    return ["ssh", *_SSH_OPTS, host, remote_cmd]


def ssh_stream(host: str, remote_cmd: str) -> None:
    """Run a shell command on the remote host, streaming output."""
    proc = subprocess.Popen(
        _ssh_cmd(host, remote_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in proc.stdout:
        sys.stdout.write(line.decode(errors="replace"))
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, remote_cmd)


def ssh_check(host: str, remote_cmd: str) -> subprocess.CompletedProcess:
    """Run a command on the remote host, return CompletedProcess (no streaming)."""
    return subprocess.run(
        _ssh_cmd(host, remote_cmd),
        capture_output=True,
        text=True,
    )


def scp_upload(local: Path, host: str, remote_path: str) -> None:
    subprocess.run(
        ["scp", "-q", *_SSH_OPTS, str(local), f"{host}:{remote_path}"],
        check=True,
    )


def rsync_download(host: str, remote_path: str, local: Path) -> None:
    """Download remote_path/ into local/ using rsync."""
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", "--progress", f"{host}:{remote_path}/", str(local)],
        check=True,
    )


def check_deps() -> None:
    for tool in ("ssh", "scp", "rsync"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            raise click.ClickException(f"'{tool}' not found on PATH.")


def setup_remote(host: str) -> None:
    """Ensure MiniFoldX is cloned/updated and deps installed on the remote."""
    ssh_stream(host, f"mkdir -p {REMOTE_CODE} {REMOTE_WEIGHTS} {REMOTE_JOBS}")

    result = ssh_check(host, f"test -f {REMOTE_CODE}/minifold_mlx/pytorch/predict.py && echo yes || echo no")
    if "yes" in result.stdout:
        click.echo("  MiniFoldX already installed — checking for updates ...")
        ssh_stream(host, f"git -C {REMOTE_CODE} fetch --depth=1 origin HEAD && git -C {REMOTE_CODE} reset --hard FETCH_HEAD 2>/dev/null || true")
    else:
        click.echo("  Cloning MiniFoldX ...")
        ssh_stream(host, f"git clone --depth=1 {REPO_URL} {REMOTE_CODE}")

    pytorch_dir = f"{REMOTE_CODE}/minifold_mlx/pytorch"
    ssh_stream(host, (
        f"pip install -q -e {pytorch_dir} "
        "fair-esm safetensors ml_collections dm-tree modelcif einops edit_distance "
        "huggingface_hub[hf_xet] 2>&1"
    ))


def run_remote(
    fasta_path: Path,
    job_id: str,
    host: str,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    out_dir: Path,
) -> Path:
    """Full orchestration: setup → upload → predict → download. Returns local output dir."""
    # Expand ~ on remote via ssh before use
    fasta_suffix = "".join(fasta_path.suffixes[-2:]) or fasta_path.suffix or ".fasta"
    fasta_name   = f"input{fasta_suffix}"

    # Expand remote paths via ssh
    exp = ssh_check(host, f"echo {REMOTE_JOBS}/{job_id}").stdout.strip()
    remote_job     = exp
    remote_fasta   = f"{remote_job}/{fasta_name}"
    remote_outputs = f"{remote_job}/outputs"

    click.echo("[1/4] Setting up MiniFoldX on remote server ...")
    ssh_stream(host, f"mkdir -p {remote_job} {remote_outputs}")
    setup_remote(host)

    # Upload HF token if available (faster weight downloads on server)
    if HF_TOKEN_PATH.exists():
        click.echo("  Uploading HuggingFace token ...")
        ssh_stream(host, "mkdir -p ~/.cache/huggingface")
        scp_upload(HF_TOKEN_PATH, host, "~/.cache/huggingface/token")

    click.echo(f"[2/4] Uploading {fasta_path.name} ({fasta_path.stat().st_size // 1024} KB) ...")
    scp_upload(fasta_path, host, remote_fasta)

    click.echo(f"[3/4] Running prediction (model={model_size}, recycling={num_recycling}) ...")
    predict_py = f"{REMOTE_CODE}/minifold_mlx/pytorch/predict.py"
    ssh_stream(host, (
        f"HF_HUB_ENABLE_HF_XET=1 "
        f"python {predict_py} {remote_fasta} "
        f"--out_dir {remote_outputs} "
        f"--cache {REMOTE_WEIGHTS} "
        f"--model_size {model_size} "
        f"--token_per_batch {token_per_batch} "
        f"--num_recycling {num_recycling}"
    ))

    click.echo("[4/4] Downloading results ...")
    local_out = out_dir / job_id
    rsync_download(host, remote_outputs, local_out)

    return local_out
