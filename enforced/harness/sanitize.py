"""
sanitize.py -- FROZEN. What an agent may see of a job's output: its own model's errors and the
training curve, never infrastructure details (paths, node names, job / task ids, harness code).
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

HOST = os.uname().nodename.split(".")[0]

FRAME = re.compile(r'^\s*File "(.+)", line (\d+), in (.+)$')
RANK = re.compile(r"^\[rank\d+\]:\s?")
EXC = re.compile(r"^([A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)[\w.]*)(:.*)?$")


def scrub(text, model_path=None):
    """replace infrastructure details; the agent's model file becomes `model.py`"""
    if model_path:
        text = text.replace(str(model_path), "model.py")
    for root in sorted({str(paths.REPO), str(paths.WORK), str(paths.DATA)}, key=len, reverse=True):
        text = re.sub(re.escape(root) + r"\S*?(?=[\s\"'),:]|$)", "<path>", text)
    text = re.sub(r"/(?:home|tmp|usr)/\S+?(?=[\s\"'),:]|$)", "<path>", text)
    text = re.sub(r"\b" + re.escape(HOST) + r"\b", "<node>", text)
    text = re.sub(r"\bt\d{10}-[0-9a-f]{6}\b", "<task>", text)
    text = re.sub(r"\b(job|SLURM_JOB_ID|GQ_JOB)[=: ]+\d{6,}\b", r"\1=<job>", text)
    return text


def model_error(text, model_path):
    """the last error in `text`: the traceback frames inside the agent's model file (as `model.py`)
    and the exception line(s); every other frame (harness, PyTorch) is dropped"""
    lines = [RANK.sub("", l) for l in text.splitlines()]
    last_exc = max((i for i, l in enumerate(lines) if EXC.match(l.strip())), default=None)
    if last_exc is None:
        return ""
    start = max((i for i, l in enumerate(lines[:last_exc]) if l.startswith("Traceback (most recent call last)")),
                default=0)
    out, i = [], start
    while i < last_exc:
        m = FRAME.match(lines[i])
        if m and m.group(1) == str(model_path):
            out.append(f'  File "model.py", line {m.group(2)}, in {m.group(3)}')
            if i + 1 < last_exc and not FRAME.match(lines[i + 1]):
                out.append(lines[i + 1])
        i += 1
    exc = lines[last_exc:last_exc + 4]
    exc = [l for l in exc if l.strip() and not FRAME.match(l) and not l.startswith(("Traceback", "  "))
           and not re.match(r"^[EWI]\d{4} ", l)][:2]                       # drop torchrun log lines
    header = "Error in your model (only the lines inside your model file are shown):" if out else \
             "Error (it was not raised inside your model file):"
    return scrub("\n".join([header] + out + exc), model_path) + "\n"


def train_curve(log_text, model_path):
    """train.log without the leading model listing and run header: params, then the step lines"""
    lines = log_text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("params=")), 0)
    return scrub("\n".join(lines[start:]), model_path) + "\n"
