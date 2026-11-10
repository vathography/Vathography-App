# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""models.py — finds the model files the manifest lists (model_manifest.json) in a local folder.
A folder is accepted file by file: same name AND same size (always) AND same sha256 (with verify=True — the first-run
check; the download itself is download.py). The node's own `models\` folder qualifies: the base files are byte-identical
to Comfy-Org's, the see-through files to the pinned Hugging Face commit.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "model_manifest.json")
VAE_CONFIG_DIR = os.path.join(HERE, "vae_config")
# the models folder beside the app: the portable package = <APP_HOME>\app\vathography_app ; the dev repo =
# <APP_HOME>\vathography_app
_PARENT = os.path.dirname(HERE)
APP_HOME = os.path.dirname(_PARENT) if os.path.basename(_PARENT).lower() == "app" else _PARENT
DEFAULT_DIR = os.path.join(APP_HOME, "models")


def _sha(p, progress=None):
    """progress(n): called with each chunk's byte count (hashing up to 11.6 GB shows its progress)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
            if progress:
                progress(len(b))
    return h.hexdigest()


def locate(models_dir, model="depth", verify=False, progress=None, cache=None):
    """-> dict(backbone, lora, vae, conditioning) of paths for `model` ('depth' | 'see-through').
    progress(n): bytes hashed so far grow by n (only when a check has to read files: verify, or another file name).
    cache: a dict path -> sha256 shared between calls, so a file both models use (the backbone, the conditioning) or a
    same-size candidate is read once."""
    def sha(p):
        if cache is not None and p in cache:
            return cache[p]
        h = _sha(p, progress)
        if cache is not None:
            cache[p] = h
        return h
    man = json.load(open(MANIFEST, encoding="utf-8"))
    want = man["models"][model]
    out, problems = {}, []
    found, by_size = {}, {}
    for root, _, files in os.walk(models_dir):
        for f in files:
            p = os.path.join(root, f)
            found.setdefault(f, []).append(p)
            if f.endswith(".safetensors"):
                by_size.setdefault(os.path.getsize(p), []).append(p)
    for entry in man["files"]:
        key = entry["role"] if entry["role"] in ("backbone_nf4", "conditioning") else f"{entry['role']}:{entry['model']}"
        if key not in want:
            continue
        name = os.path.basename(entry["path"])
        hits = [p for p in found.get(name, []) if os.path.getsize(p) == entry["size"]]
        if verify:                         # the first copy that matches is the one used: stop there (a folder may
            hits = next(([p] for p in hits if sha(p) == entry["sha256"]), [])      # hold several copies)
        if not hits:                       # another file name (e.g. Comfy-Org's): same size AND same sha256 only
            hits = next(([p] for p in by_size.get(entry["size"], []) if sha(p) == entry["sha256"]), [])
        if not hits:
            problems.append(f"{name} ({entry['size'] / 1e9:.2f} GB) not found{' with the right checksum' if verify else ''} in {models_dir}")
            continue
        out[{"backbone_nf4": "backbone", "conditioning": "conditioning"}.get(entry["role"], entry["role"])] = hits[0]
    if problems:
        raise FileNotFoundError("; ".join(problems))
    return out
