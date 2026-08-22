"""Colab session orchestration and job-script generation."""

import subprocess
import tempfile
from pathlib import Path

import click

WEIGHTS_REMOTE  = "gdrive:Colab_Data/minifold_colab/weights"
JOBS_REMOTE     = "gdrive:Colab_Data/minifold_colab/jobs"
RCLONE_CONF     = Path.home() / ".config" / "rclone" / "rclone.conf"
RCLONE_VERSION  = "v1.75.0"


def _run(*cmd: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=True, **kwargs)


def colab(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return _run("colab", *args, **kwargs)


def rclone(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return _run("rclone", *args, **kwargs)


def check_deps() -> None:
    for tool in ("colab", "rclone"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            raise click.ClickException(
                f"'{tool}' not found on PATH.\n"
                + ("  brew install rclone" if tool == "rclone" else "  pip install colab-cli")
            )
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    if "gdrive:" not in result.stdout:
        raise click.ClickException(
            "rclone 'gdrive' remote not configured.\n"
            "Run:  rclone config"
        )
    scope_result = subprocess.run(
        ["rclone", "config", "show", "gdrive"],
        capture_output=True, text=True,
    )
    for line in scope_result.stdout.splitlines():
        if line.strip().startswith("scope"):
            scope = line.split("=", 1)[-1].strip()
            if scope == "drive":
                click.echo(
                    "Warning: gdrive remote has scope = drive (full Drive access).\n"
                    "For safer operation, re-configure with drive.file scope:\n"
                    "  rclone config reconnect gdrive: --drive-scope drive.file\n"
                    "This restricts access to only files this app creates.",
                    err=True,
                )
            break


def make_job_script(
    job_id: str,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    int8_esm2: bool,
    triton_kernels: bool = False,
    out_dest: str = "both",
) -> str:
    """Return a Python script that runs on the Colab VM."""
    import subprocess as _sp
    _repo = Path(__file__).resolve().parent.parent.parent
    _commit = _sp.run(["git", "-C", str(_repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    kernels_flag = ', "--kernels"' if triton_kernels else ''
    predict_script = '"/content/MiniFoldX/minifold_mlx/pytorch/predict.py"'
    predict_invocation = (
        f'run(sys.executable, "/content/minifold_predict_int8.py",\n'
        f'    FASTA, "--out_dir", OUTPUTS, "--cache", WEIGHTS,\n'
        f'    "--checkpoint", CKPT_PATH,\n'
        f'    "--model_size", MODEL_SIZE, "--token_per_batch", str(TOKEN_PER_BATCH){kernels_flag})'
        if int8_esm2 else
        f'run(sys.executable, {predict_script},\n'
        f'    FASTA, "--out_dir", OUTPUTS, "--cache", WEIGHTS,\n'
        f'    "--checkpoint", CKPT_PATH,\n'
        f'    "--model_size", MODEL_SIZE, "--token_per_batch", str(TOKEN_PER_BATCH){kernels_flag})'
    )

    int8_install = (
        'run("uv", "pip", "install", "--system", "bitsandbytes")'
        if int8_esm2 else
        '# (standard precision ESM2)'
    )

    triton_install = (
        'run("uv", "pip", "install", "--system", "triton")'
        if triton_kernels else
        '# (standard PyTorch ops)'
    )

    int8_wrapper = r"""
# Write the int8 ESM2 wrapper (monkey-patches create_model before running predict)
INT8_WRAPPER = '''
import sys, os
sys.path.insert(0, "/content/MiniFoldX/minifold_mlx/pytorch")
import torch, torch.nn as nn

def _quantize_esm2_int8(module):
    import bitsandbytes as bnb
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            q = bnb.nn.Linear8bitLt(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            q.weight = bnb.nn.Int8Params(
                child.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            )
            if child.bias is not None:
                q.bias = nn.Parameter(child.bias.data.clone())
            setattr(module, name, q)
        else:
            _quantize_esm2_int8(child)

import predict as _predict
_orig = _predict.create_model
def _int8_create_model(checkpoint, device, compile=False, kernels=False):
    alphabet, model = _orig(checkpoint, device, compile, kernels)
    print("Quantizing ESM2 to int8 (bitsandbytes) ...", flush=True)
    _quantize_esm2_int8(model.esm)
    model = model.to(device)
    model.eval()
    return alphabet, model
_predict.create_model = _int8_create_model
_predict.predict()
'''
with open("/content/minifold_predict_int8.py", "w") as fh:
    fh.write(INT8_WRAPPER)
""" if int8_esm2 else "# (no int8 wrapper needed)"

    return f"""\
#!/usr/bin/env python3
# MiniFold job script — job {job_id}
import subprocess, sys, os

JOB_ID          = "{job_id}"
MODEL_SIZE      = "{model_size}"
TOKEN_PER_BATCH = {token_per_batch}
NUM_RECYCLING   = {num_recycling}
WEIGHTS_REMOTE  = "{WEIGHTS_REMOTE}"
JOBS_REMOTE     = "{JOBS_REMOTE}"

FASTA   = "/content/input.fasta"
WEIGHTS = "/content/weights"
OUTPUTS = "/content/outputs"
VM_CACHE      = os.path.join(WEIGHTS, "vm_cache")
UV_BIN_CACHE  = os.path.join(VM_CACHE, "uv")
MINIFOLD_TAR  = os.path.join(VM_CACHE, "minifold.tar.gz")
UV_PKG_CACHE  = os.path.join(VM_CACHE, "uv_packages")
ESM2_HF_REPO  = "z-ardern/MiniFoldX_weights"
ESM2_HUB_DIR  = os.path.join(WEIGHTS, "hub", "checkpoints")
ESM2_PT_PATH  = os.path.join(ESM2_HUB_DIR, "esm2_t36_3B_UR50D.pt")
CKPT_PATH     = os.path.join(WEIGHTS, f"minifold_{{MODEL_SIZE}}.safetensors")

# Point torch hub (and fair-esm) at our Drive-cached weights dir
os.environ["TORCH_HOME"] = WEIGHTS

def run(*cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    import sys as _sys
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stderr_lines = []
    for line in proc.stdout:
        decoded = line.decode(errors="replace")
        print(decoded, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))

def rclone_copy(src, dst, *extra):
    # Exit code 3 = source directory not found (normal on first run); treat as no-op.
    print(f"+ rclone copy {{src}} {{dst}}", flush=True)
    r = subprocess.run([
        "rclone", "copy", src, dst, "--progress",
        "--drive-chunk-size", "256M",
        "--transfers", "8",
        "--buffer-size", "256M",
        *extra,
    ])
    if r.returncode not in (0, 3):
        raise RuntimeError(f"rclone copy failed with exit code {{r.returncode}}")

os.makedirs(WEIGHTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(ESM2_HUB_DIR, exist_ok=True)
os.makedirs(VM_CACHE, exist_ok=True)
os.makedirs(UV_PKG_CACHE, exist_ok=True)

def step(msg):
    print(f"\\n=== {{msg}} ===", flush=True)

# rclone must be downloaded before Drive is accessible — cannot be cached on Drive
step("Installing rclone")
RCLONE_VERSION = "{RCLONE_VERSION}"
rclone_zip = f"/tmp/rclone-{{RCLONE_VERSION}}-linux-amd64.zip"
rclone_url = f"https://github.com/rclone/rclone/releases/download/{{RCLONE_VERSION}}/rclone-{{RCLONE_VERSION}}-linux-amd64.zip"
run("curl", "-fsSL", rclone_url, "-o", rclone_zip)
run("unzip", "-q", "-o", rclone_zip, "-d", "/tmp/rclone_bin")
run("cp", f"/tmp/rclone_bin/rclone-{{RCLONE_VERSION}}-linux-amd64/rclone", "/usr/local/bin/rclone")

step("GPU info")
run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")

step("Pulling weights from Drive")
rclone_copy(WEIGHTS_REMOTE, WEIGHTS)

step("Installing uv")
new_model_weights = False  # ESM2 + MiniFold checkpoint (large)
new_vm_cache = False       # uv binary, repo tar, pip wheels (small)
import shutil
if os.path.exists(UV_BIN_CACHE):
    print("Using cached uv binary", flush=True)
    shutil.copy(UV_BIN_CACHE, "/usr/local/bin/uv")
    os.chmod("/usr/local/bin/uv", 0o755)
else:
    run(sys.executable, "-m", "pip", "install", "-q", "uv")
    uv_path = shutil.which("uv")
    if uv_path:
        shutil.copy(uv_path, UV_BIN_CACHE)
    new_vm_cache = True

step("Setting up MiniFoldX code")
os.environ["UV_CACHE_DIR"] = UV_PKG_CACHE
MINIFOLDX_DIR = "/content/MiniFoldX"
PYTORCH_DIR   = f"{{MINIFOLDX_DIR}}/minifold_mlx/pytorch"
MINIFOLD_TAR_COMMIT = "{_commit}"
MINIFOLD_TAR_STAMP  = os.path.join(VM_CACHE, "minifold_commit.txt")
cached_commit = open(MINIFOLD_TAR_STAMP).read().strip() if os.path.exists(MINIFOLD_TAR_STAMP) else ""
if os.path.exists(MINIFOLD_TAR) and cached_commit == MINIFOLD_TAR_COMMIT:
    print(f"Using cached MiniFoldX repo ({{MINIFOLD_TAR_COMMIT[:8]}})", flush=True)
    run("tar", "-xzf", MINIFOLD_TAR, "-C", "/content")
else:
    if cached_commit and cached_commit != MINIFOLD_TAR_COMMIT:
        print(f"Code updated ({{cached_commit[:8]}} → {{MINIFOLD_TAR_COMMIT[:8]}}), re-cloning ...", flush=True)
    run("git", "clone", "--depth=1", "https://github.com/ZacharyArdern/MiniFoldX.git", MINIFOLDX_DIR)
    run("tar", "-czf", MINIFOLD_TAR, "-C", "/content", "MiniFoldX")
    open(MINIFOLD_TAR_STAMP, "w").write(MINIFOLD_TAR_COMMIT)
    new_vm_cache = True

step("Installing Python dependencies")
run("uv", "pip", "install", "--system", "-e", PYTORCH_DIR,
    "huggingface_hub", "fair-esm", "safetensors", "ml_collections",
    "dm-tree", "modelcif", "einops", "edit_distance")
{int8_install}
{triton_install}

step("Preparing MiniFold checkpoint")
os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")
from huggingface_hub import hf_hub_download
ckpt_name = f"minifold_{{MODEL_SIZE}}.safetensors"
ckpt_dest = os.path.join(WEIGHTS, ckpt_name)
if not os.path.exists(ckpt_dest):
    print(f"Downloading {{ckpt_name}} from HuggingFace ...", flush=True)
    hf_hub_download(
        repo_id=ESM2_HF_REPO,
        filename=ckpt_name,
        local_dir=WEIGHTS,
        local_dir_use_symlinks=False,
    )
    new_model_weights = True
    step("Saving MiniFold checkpoint to Drive")
    rclone_copy(WEIGHTS, WEIGHTS_REMOTE, "--exclude", "vm_cache/**", "--exclude", "hub/**")
else:
    print(f"Using cached {{ckpt_name}}", flush=True)

if new_vm_cache:
    step("Saving VM cache to Drive")
    rclone_copy(VM_CACHE, f"{{WEIGHTS_REMOTE}}/vm_cache")
else:
    print("VM cache unchanged — skipping Drive push.", flush=True)

# ESM2 is downloaded by torch hub (TORCH_HOME=WEIGHTS) on first run,
# then pushed to Drive after predict so subsequent runs skip the download.
esm2_cached_before = os.path.exists(ESM2_PT_PATH)
if esm2_cached_before:
    print(f"ESM2 weights cached at {{ESM2_PT_PATH}}", flush=True)
else:
    print("ESM2 not cached — will download from fbaipublicfiles during model load.", flush=True)

{int8_wrapper}
step("Running structure prediction")
{predict_invocation}

OUT_DEST = "{out_dest}"

if not esm2_cached_before and os.path.exists(ESM2_PT_PATH):
    step("Caching ESM2 to Drive")
    rclone_copy(ESM2_HUB_DIR, f"{{WEIGHTS_REMOTE}}/hub/checkpoints")

step("Saving results")
if OUT_DEST in ("both", "drive"):
    job_remote = f"{{JOBS_REMOTE}}/{{JOB_ID}}/outputs"
    rclone_copy(OUTPUTS, job_remote)
    print(f"Results saved to Drive: {{job_remote}}", flush=True)
else:
    print("Skipping Drive result push (--out pwd).", flush=True)
"""


def _download_from_vm(session: str, remote_dir: str, local_dir: Path) -> None:
    """Tar outputs on VM, download the archive, extract locally."""
    import tarfile
    remote_tar = "/tmp/minifold_outputs.tar.gz"
    colab("exec", "-s", session, "--timeout", "60", "-f", "/dev/stdin",
          input=f"import subprocess; subprocess.run(['tar','-czf','{remote_tar}','-C','{remote_dir}','.'], check=True)",
          text=True)
    local_tar = local_dir / "outputs.tar.gz"
    colab("download", "-s", session, remote_tar, str(local_tar))
    with tarfile.open(local_tar) as tf:
        tf.extractall(local_dir)
    local_tar.unlink()


def run_prediction(
    fasta_path: Path,
    job_id: str,
    session: str,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    timeout: int,
    int8_esm2: bool,
    triton_kernels: bool = False,
    out_dest: str = "both",
) -> Path:
    """Upload FASTA, execute job script, download results. Returns local output dir."""

    need_drive = out_dest in ("both", "drive")
    need_local = out_dest in ("both", "pwd")

    # Upload rclone credentials (only needed if pushing to Drive)
    if need_drive:
        click.echo("[2/5] Uploading rclone credentials ...")
        colab("exec", "-s", session, "--timeout", "10",
              "-f", "/dev/stdin",
              input="import os; os.makedirs('/root/.config/rclone', exist_ok=True)",
              text=True)
        colab("upload", "-s", session,
              str(RCLONE_CONF), "/root/.config/rclone/rclone.conf")
    else:
        click.echo("[2/5] Skipping rclone credentials (--out pwd) ...")

    # Upload FASTA
    click.echo(f"[3/5] Uploading {fasta_path.name} ...")
    colab("upload", "-s", session, str(fasta_path), "/content/input.fasta")

    # Generate and run job script
    click.echo(f"[4/5] Running prediction (timeout {timeout}s) ...")
    job_script = make_job_script(
        job_id, model_size, token_per_batch, num_recycling,
        int8_esm2, triton_kernels, out_dest,
    )
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(job_script)
        job_script_path = f.name

    colab("exec", "-s", session, "-f", job_script_path, "--timeout", str(timeout))

    # Download results
    out_dir = Path("./minifold_results") / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if need_local:
        click.echo("[5/5] Downloading results ...")
        if need_drive:
            # Pull from Drive (already pushed by job script)
            result = subprocess.run(
                ["rclone", "copy", f"{JOBS_REMOTE}/{job_id}/outputs",
                 str(out_dir), "--progress"]
            )
            if result.returncode not in (0, 3):
                raise click.ClickException(f"rclone download failed (exit {result.returncode})")
        else:
            # Pull directly from VM before session is terminated
            _download_from_vm(session, "/content/outputs", out_dir)
    else:
        click.echo("[5/5] Skipping local download (--out drive) ...")

    return out_dir
