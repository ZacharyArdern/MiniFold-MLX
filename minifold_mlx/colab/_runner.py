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
) -> str:
    """Return a Python script that runs on the Colab VM."""

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
ESM2_HF_FILE  = "ESM2_fp16/esm2_t36_3B_UR50D_fp16.safetensors"
ESM2_HUB_DIR  = os.path.join(WEIGHTS, "hub", "checkpoints")
ESM2_PT_PATH  = os.path.join(ESM2_HUB_DIR, "esm2_t36_3B_UR50D.pt")
CKPT_PATH     = os.path.join(WEIGHTS, f"minifold_{{MODEL_SIZE}}_final.ckpt")

# Point torch hub (and fair-esm) at our Drive-cached weights dir
os.environ["TORCH_HOME"] = WEIGHTS

def run(*cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    import sys as _sys
    r = subprocess.run(list(cmd), stderr=subprocess.PIPE)
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr.decode(errors="replace"), file=_sys.stderr, flush=True)
        raise subprocess.CalledProcessError(r.returncode, list(cmd))

def rclone_copy(src, dst, *extra):
    # Exit code 3 = source directory not found (normal on first run); treat as no-op.
    print(f"+ rclone copy {{src}} {{dst}}", flush=True)
    r = subprocess.run(["rclone", "copy", src, dst, "--progress", *extra])
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
new_downloads = False
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
    new_downloads = True

step("Setting up MiniFoldX code")
os.environ["UV_CACHE_DIR"] = UV_PKG_CACHE
MINIFOLDX_DIR = "/content/MiniFoldX"
PYTORCH_DIR   = f"{{MINIFOLDX_DIR}}/minifold_mlx/pytorch"
if os.path.exists(MINIFOLD_TAR):
    print("Using cached MiniFoldX repo", flush=True)
    run("tar", "-xzf", MINIFOLD_TAR, "-C", "/content")
else:
    run("git", "clone", "--depth=1", "https://github.com/ZacharyArdern/MiniFoldX.git", MINIFOLDX_DIR)
    run("tar", "-czf", MINIFOLD_TAR, "-C", "/content", "MiniFoldX")
    new_downloads = True

step("Installing Python dependencies")
run("uv", "pip", "install", "--system", "-e", PYTORCH_DIR,
    "huggingface_hub", "fair-esm", "safetensors", "ml_collections",
    "dm-tree", "modelcif", "einops", "edit_distance")
{int8_install}
{triton_install}

step("Preparing ESM2 weights")
os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")
from huggingface_hub import hf_hub_download
if not os.path.exists(ESM2_PT_PATH):
    print("Downloading ESM2 fp16 safetensors from HF ...", flush=True)
    import tempfile
    st_tmp = hf_hub_download(
        repo_id=ESM2_HF_REPO,
        filename=ESM2_HF_FILE,
        local_dir=tempfile.mkdtemp(),
        local_dir_use_symlinks=False,
    )
    print("Converting safetensors → torch hub cache ...", flush=True)
    import torch
    from safetensors.torch import load_file
    state = load_file(st_tmp)
    torch.save({{'model': state}}, ESM2_PT_PATH)
    print(f"ESM2 cached to {{ESM2_PT_PATH}}", flush=True)
    new_downloads = True
else:
    print("Using cached ESM2 weights", flush=True)

step("Preparing MiniFold checkpoint")
ckpt_name = f"minifold_{{MODEL_SIZE}}_final.ckpt"
ckpt_dest = os.path.join(WEIGHTS, ckpt_name)
if not os.path.exists(ckpt_dest):
    print(f"Downloading {{ckpt_name}} from HuggingFace ...", flush=True)
    hf_hub_download(
        repo_id="jwohlwend/minifold",
        filename=ckpt_name,
        local_dir=WEIGHTS,
        local_dir_use_symlinks=False,
    )
    new_downloads = True
else:
    print(f"Using cached {{ckpt_name}}", flush=True)

{int8_wrapper}
step("Running structure prediction")
{predict_invocation}

step("Saving weights to Drive")
if new_downloads:
    print("Pushing new weights to Drive cache ...", flush=True)
    rclone_copy(WEIGHTS, WEIGHTS_REMOTE)
else:
    print("Weights unchanged — skipping Drive push.", flush=True)

# Push results to Drive
job_remote = f"{{JOBS_REMOTE}}/{{JOB_ID}}/outputs"
rclone_copy(OUTPUTS, job_remote)
print(f"Results saved to Drive: {{job_remote}}", flush=True)
"""


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
) -> Path:
    """Upload FASTA, execute job script, download results. Returns local output dir."""

    # Upload rclone credentials
    click.echo("[2/5] Uploading rclone credentials ...")
    colab("exec", "-s", session, "--timeout", "10",
          "-f", "/dev/stdin",
          input="import os; os.makedirs('/root/.config/rclone', exist_ok=True)",
          text=True)
    colab("upload", "-s", session,
          str(RCLONE_CONF), "/root/.config/rclone/rclone.conf")

    # Upload FASTA
    click.echo(f"[3/5] Uploading {fasta_path.name} ...")
    colab("upload", "-s", session, str(fasta_path), "/content/input.fasta")

    # Generate and run job script
    click.echo(f"[4/5] Running prediction (timeout {timeout}s) ...")
    job_script = make_job_script(job_id, model_size, token_per_batch, num_recycling, int8_esm2, triton_kernels)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(job_script)
        job_script_path = f.name

    colab("exec", "-s", session, "-f", job_script_path, "--timeout", str(timeout))

    # Download results
    click.echo("[5/5] Downloading results ...")
    out_dir = Path("./minifold_results") / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["rclone", "copy", f"{JOBS_REMOTE}/{job_id}/outputs", str(out_dir), "--progress"]
    )
    if result.returncode not in (0, 3):
        raise click.ClickException(f"rclone download failed (exit {result.returncode})")

    return out_dir
