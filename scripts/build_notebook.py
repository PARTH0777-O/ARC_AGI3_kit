"""Splice `agent/my_agent.py` into `notebooks/submission.ipynb` following the official Kaggle ARC-AGI-3 runner."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# Pick accelerator: cpu, t4, p100, rtx6000
ACCELERATOR = "cpu"

_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text(encoding="utf-8")

    install_cell = code_cell(
        dedent(
            """\
            import os
            import sys
            import glob
            import subprocess

            wheel_dirs = [
                "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
                "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
            ]
            for p in glob.glob("/kaggle/input/**/arc_agi_3_wheels", recursive=True):
                if p not in wheel_dirs:
                    wheel_dirs.append(p)

            installed = False
            for wdir in wheel_dirs:
                if os.path.isdir(wdir):
                    print(f"Installing wheels from: {wdir}")
                    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wdir, "arc-agi", "python-dotenv"]
                    subprocess.check_call(cmd)
                    installed = True
                    break

            if not installed:
                whls = glob.glob("/kaggle/input/**/*.whl", recursive=True)
                if whls:
                    print(f"Installing individual wheels: {whls}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index"] + whls)
                else:
                    print("Warning: No wheels found in /kaggle/input, attempting standard import...")
            """
        )
    )

    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = """\
import os
import sys
import glob
import shutil
import subprocess
import time

print("="*60)
print("ARC-AGI-3 Grandmaster Agent Submission Initializing...")
print("="*60)

# Locate official ARC-AGI-3-Agents framework
framework_cands = [
    "/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
]
for p in glob.glob("/kaggle/input/**/ARC-AGI-3-Agents", recursive=True):
    if p not in framework_cands:
        framework_cands.append(p)

src_fw = None
for cand in framework_cands:
    if os.path.exists(cand) and os.path.isdir(cand):
        src_fw = cand
        break

dst_fw = "/kaggle/working/ARC-AGI-3-Agents"
if src_fw:
    if os.path.exists(dst_fw):
        shutil.rmtree(dst_fw)
    shutil.copytree(src_fw, dst_fw)
    print(f"Copied framework from {src_fw} to {dst_fw}")

    # Install my_agent.py
    shutil.copyfile("/tmp/my_agent.py", os.path.join(dst_fw, "agents", "templates", "my_agent.py"))
    print("Installed my_agent.py to agents/templates/my_agent.py")

    # Register MyAgent in agents/__init__.py
    with open(os.path.join(dst_fw, "agents", "__init__.py"), "w") as f:
        f.write('''from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    'random': Random,
    'myagent': MyAgent,
}
''')

# Probe for live competition gateway sidecar across common endpoints
probe_urls = [
    ("gateway", "http://gateway:8001/api/games", "http://gateway:8001/"),
    ("127.0.0.1", "http://127.0.0.1:8001/api/games", "http://127.0.0.1:8001/"),
    ("localhost", "http://localhost:8001/api/games", "http://localhost:8001/"),
]

active_host = None
active_base_url = None
print("Probing competition gateway sidecar...")

# Retry loop (up to ~30 seconds) to allow sidecar container startup in rerun environment
for attempt in range(15):
    for host_name, probe_url, base_url in probe_urls:
        try:
            res = subprocess.run(
                ["curl", "-s", "--connect-timeout", "1", "--max-time", "2", probe_url],
                capture_output=True, text=True
            )
            if res.returncode == 0 and len(res.stdout) > 0 and ("[" in res.stdout or "{" in res.stdout):
                active_host = host_name
                active_base_url = base_url
                print(f"Gateway is READY on {probe_url}! (Attempt {attempt+1})")
                break
        except Exception:
            pass
    if active_host:
        break
    time.sleep(2)

env_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
print(f"KAGGLE_IS_COMPETITION_RERUN: {env_rerun}")
print(f"Active Gateway Host: {active_host}")

# Execute official framework if gateway is alive or rerun flag is set
should_run = bool(active_host) or bool(env_rerun and env_rerun.lower() not in ('0', 'false'))

if should_run and src_fw:
    host_to_use = active_host or "gateway"
    base_to_use = active_base_url or "http://gateway:8001/"
    
    with open(os.path.join(dst_fw, ".env"), "w") as f:
        f.write(f'''SCHEME=http
HOST={host_to_use}
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL={base_to_use}
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
''')

    env = os.environ.copy()
    env["MPLBACKEND"] = "agg"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["ARC_BASE_URL"] = base_to_use
    env["HOST"] = host_to_use

    print(f"Executing: python main.py --agent myagent against {base_to_use}...")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--agent", "myagent"],
        cwd=dst_fw, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print(f"Framework main.py finished with return code: {proc.returncode}")

    # Search and copy generated submission.parquet
    for cand in [
        os.path.join(dst_fw, "submission.parquet"),
        "/kaggle/working/server_recording/submission.parquet",
    ] + glob.glob("/kaggle/working/**/*.parquet", recursive=True):
        if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath("/kaggle/working/submission.parquet"):
            shutil.copyfile(cand, "/kaggle/working/submission.parquet")
            print(f"Found and copied {cand} -> /kaggle/working/submission.parquet")
            break

# Ensure valid submission.parquet always exists
if not os.path.exists('/kaggle/working/submission.parquet'):
    import pandas as pd
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score']
    )
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
    print("Saved fallback/commit placeholder submission.parquet")

if os.path.exists('/kaggle/working/submission.parquet'):
    import pandas as pd
    sub = pd.read_parquet('/kaggle/working/submission.parquet')
    print("Final /kaggle/working/submission.parquet preview:")
    print(sub.head(10))
    print(f"Total rows: {len(sub)}")
"""
    run_cell = code_cell(run_cell_source)

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Grandmaster Agent with Causal Memory, Kinematics Probing, A* Grid Pathfinding, and Sokoban Solvers.\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`."
            ),
            install_cell,
            write_agent_cell,
            run_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)} (accelerator: {ACCELERATOR})")

    # Mirror to arc_agi3_kit subdirectory if it exists
    sub_notebook = ROOT / "arc_agi3_kit" / "notebooks" / "submission.ipynb"
    if sub_notebook.parent.exists():
        sub_notebook.write_text(json.dumps(build(), indent=1), encoding="utf-8")

    # Sync kernel metadata
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"[build_notebook] Synced enable_gpu={wanted} in {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
