# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""download.py — first-run downloads: the models from the Vathography Hugging Face, PyTorch + OpenCV from their
publishers (the manifest's `third_party` entries, each an exact URL).
Plain HTTPS, no huggingface_hub: a model = GET https://huggingface.co/<repo>/resolve/<revision>/<path>, a third-party
wheel = GET its `url`; both with a Range resume into <name>.part, sha256 against model_manifest.json, then os.replace to
the final name (a crash never leaves a file under the final name that is not the right one). Size + free disk are
checked BEFORE the first byte. The only module of the app that uses the network.
"""
import hashlib
import json
import os
import shutil
import urllib.request

from . import models

URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
CHUNK = 1 << 20
SPARE = 1 << 30                                   # keep 1 GiB free on the drive after the download


class Cancelled(Exception):
    pass


def manifest():
    return json.load(open(models.MANIFEST, encoding="utf-8"))


def third_party():
    """The manifest's `third_party` entries (PyTorch, OpenCV). ⛔ Never part of `files`: those are the models."""
    return manifest().get("third_party", [])


def is_third_party(entry):
    return "url" in entry


def source_url(entry):
    return entry["url"] if is_third_party(entry) else URL.format(**entry)


def hosted(entry):
    """False for a model whose repo is not online (a '(' in its repo or revision); a third-party URL is always hosted."""
    if is_third_party(entry):
        return True
    return "(" not in entry["repo"] and "(" not in entry["revision"]


def entries_for(model):
    """The manifest entries `model` ('depth' | 'see-through') needs (the same selection as models.locate)."""
    man = manifest()
    want = man["models"][model]
    out = []
    for e in man["files"]:
        key = e["role"] if e["role"] in ("backbone_nf4", "conditioning") else f"{e['role']}:{e['model']}"
        if key in want:
            out.append(e)
    return out


def target(dest_dir, entry):
    return os.path.join(dest_dir, entry["file"] if is_third_party(entry) else os.path.basename(entry["path"]))


def missing(models_dir, model_list):
    """Entries not yet in models_dir under their own name with their size (dedup over the models)."""
    seen, out = set(), []
    for m in model_list:
        for e in entries_for(m):
            t = target(models_dir, e)
            if t in seen:
                continue
            seen.add(t)
            if not (os.path.isfile(t) and os.path.getsize(t) == e["size"]):
                out.append(e)
    return out


def bytes_needed(dest_dir, entries):
    """Bytes still to fetch (a .part already on disk counts as fetched; a finished third-party wheel counts whole)."""
    n = 0
    for e in entries:
        dst = target(dest_dir, e)
        if is_third_party(e) and os.path.isfile(dst) and os.path.getsize(dst) == e["size"]:
            continue
        part = dst + ".part"
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        n += max(e["size"] - have, 0)
    return n


def check_space(models_dir, entries, extra=0, what="the models"):
    """None when the drive has room, else a sentence. extra: more bytes needed on the same drive (e.g. an unpack)."""
    os.makedirs(models_dir, exist_ok=True)
    need = bytes_needed(models_dir, entries) + extra
    free = shutil.disk_usage(models_dir).free
    if free < need + SPARE:
        return (f"Not enough free space on the drive of {models_dir}: {what} need {need / 1e9:.1f} GB "
                f"and {free / 1e9:.1f} GB are free.")
    return None


def _hash_file(path, h, cancel=None):
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            h.update(b)


def finished(entry, dest_dir, cancel=None):
    """A third-party wheel already under its final name with the pinned size AND sha256 (not fetched again)."""
    dst = target(dest_dir, entry)
    if not (os.path.isfile(dst) and os.path.getsize(dst) == entry["size"]):
        return False
    h = hashlib.sha256()
    _hash_file(dst, h, cancel)
    return h.hexdigest() == entry["sha256"]


def fetch(entry, dest_dir, progress=None, cancel=None, timeout=60):
    """Download one manifest entry (resuming a .part), verify its sha256, move it into place. -> final path.
    progress(done_bytes, total_bytes); cancel = a threading.Event (the .part is kept for the next resume).
    A third-party wheel already finished with the right hash is returned as it is (a crash after the fetch re-downloads
    nothing); the models' path is unchanged."""
    if not hosted(entry):
        raise RuntimeError(f"{os.path.basename(entry['path'])} is not online yet.")
    dst = target(dest_dir, entry)
    part = dst + ".part"
    os.makedirs(dest_dir, exist_ok=True)
    size = entry["size"]
    if is_third_party(entry) and finished(entry, dest_dir, cancel):
        if progress:
            progress(size, size)
        return dst
    for attempt in (1, 2):                        # a wrong hash: the .part is deleted and fetched once more
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        h = hashlib.sha256()
        if have:
            _hash_file(part, h, cancel)           # resume: the bytes already on disk enter the hash first
        if have < size:
            req = urllib.request.Request(source_url(entry), headers={"User-Agent": "vathography-app"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if have and r.status != 206:      # the server ignored the Range: start over
                    have, h = 0, hashlib.sha256()
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
                        h.update(b)
                        have += len(b)
                        if progress:
                            progress(have, size)
        if have < size:                           # the connection ended early: keep the .part for the next resume
            raise RuntimeError(f"{os.path.basename(dst)}: the download stopped at {have / 1e9:.2f} of "
                               f"{size / 1e9:.2f} GB; start it again to resume.")
        if h.hexdigest() == entry["sha256"]:
            os.replace(part, dst)
            return dst
        os.remove(part)
        if attempt == 2:
            raise RuntimeError(f"{os.path.basename(dst)}: the downloaded file does not match its checksum.")
    return dst
