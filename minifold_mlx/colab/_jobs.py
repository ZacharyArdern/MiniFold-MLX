"""Job registry — persists job metadata to ~/.config/minifold_colab/jobs.json."""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "minifold_colab"
JOBS_FILE  = CONFIG_DIR / "jobs.json"


def load_jobs() -> dict:
    if not JOBS_FILE.exists():
        return {}
    return json.loads(JOBS_FILE.read_text())


def save_job(job: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs()
    jobs[job["job_id"]] = job
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))
