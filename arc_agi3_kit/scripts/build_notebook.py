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

# Quick probe for live competition gateway sidecar
gateway_url = "http://gateway:8001/api/games"
gateway_alive = False
try:
    res = subprocess.run(["curl", "-s", "--connect-timeout", "1", "--max-time", "2", gateway_url], capture_output=True, text=True)
    if res.returncode == 0 and len(res.stdout) > 0 and ("[" in res.stdout or "{" in res.stdout):
        gateway_alive = True
        print(f"Gateway sidecar is responsive: {res.stdout[:100]}...")
except Exception as e:
    pass

env_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
is_rerun = bool(env_rerun and env_rerun.lower() not in ('0', 'false')) or gateway_alive
print(f"KAGGLE_IS_COMPETITION_RERUN: {env_rerun}")
print(f"Gateway alive: {gateway_alive}")
print(f"Is Rerun / Competition mode: {is_rerun}")

if is_rerun:
    if not gateway_alive:
        print("Waiting for gateway sidecar at http://gateway:8001/api/games...")
        for attempt in range(60):
            try:
                res = subprocess.run(["curl", "-s", "--connect-timeout", "1", "--max-time", "2", gateway_url], capture_output=True, text=True)
                if res.returncode == 0 and len(res.stdout) > 0 and ("[" in res.stdout or "{" in res.stdout):
                    print(f"Gateway sidecar is READY! Response: {res.stdout[:100]}...")
                    gateway_alive = True
                    break
            except Exception as e:
                pass
            time.sleep(2)

    if not gateway_alive:
        print("Warning: Gateway sidecar not responding to probe, proceeding with agent execution...")

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

        shutil.copyfile("/tmp/my_agent.py", os.path.join(dst_fw, "agents", "templates", "my_agent.py"))
        print("Installed my_agent.py to agents/templates/my_agent.py")

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

        with open(os.path.join(dst_fw, ".env"), "w") as f:
            f.write('''SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
''')

        env = os.environ.copy()
        env["MPLBACKEND"] = "agg"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        print("Executing: python main.py --agent myagent...")
        proc = subprocess.Popen([sys.executable, "main.py", "--agent", "myagent"], cwd=dst_fw, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        print(f"Framework main.py finished with return code: {proc.returncode}")

        # Find and copy generated submission.parquet
        found_parquet = False
        for cand in [
            os.path.join(dst_fw, "submission.parquet"),
            "/kaggle/working/server_recording/submission.parquet",
        ] + glob.glob("/kaggle/working/**/*.parquet", recursive=True):
            if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath("/kaggle/working/submission.parquet"):
                shutil.copyfile(cand, "/kaggle/working/submission.parquet")
                print(f"Found and copied {cand} -> /kaggle/working/submission.parquet")
                found_parquet = True
                break
        if not found_parquet:
            print("Note: No specific submission.parquet generated by main.py yet.")
    else:
        print("Error: ARC-AGI-3-Agents directory not found in /kaggle/input!")

if not os.path.exists('/kaggle/working/submission.parquet'):
    import pandas as pd
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
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
