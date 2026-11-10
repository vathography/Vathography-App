# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""outnames.py — versioned output names. Every output gets the next free 3-digit version: <base>_v001.<ext>, _v002, ... An existing file
is never replaced. Standard library only: the window imports it too (no PyAV / torch in the window process).
"""
import os
import re

# Every output gets the next free 3-digit version: <base>_v001.<ext>, _v002, ... An existing file is never replaced.
VERSION_RE = r"_v(\d{3,})"


def highest_version(folder, base, aliases=()):
    """Highest version among <base>[_vNNN].<any ext> (and each alias base) in `folder`; an unversioned name = 0;
    none = -1. Temp files (.tmp) do not count."""
    pats = [re.compile(re.escape(b) + "(?:" + VERSION_RE + ")?$", re.IGNORECASE) for b in (base,) + tuple(aliases)]
    best = -1
    try:
        names = os.listdir(folder)
    except OSError:
        return best
    for fn in names:
        if fn.lower().endswith(".tmp"):
            continue
        stem = os.path.splitext(fn)[0]
        for p in pats:
            m = p.match(stem)
            if m:
                best = max(best, int(m.group(1)) if m.group(1) else 0)
    return best


def latest_outputs(folder, base, aliases=()):
    """The files of the highest version of <base> in `folder` (an unversioned or alias name = version 0)."""
    v = highest_version(folder, base, aliases)
    if v < 0:
        return []
    want = [f"{base}_v{v:03d}"] if v > 0 else [base] + list(aliases)
    return sorted(os.path.join(folder, fn) for fn in os.listdir(folder)
                  if not fn.lower().endswith(".tmp") and os.path.splitext(fn)[0].lower() in [w.lower() for w in want])


def write_versioned(folder, base, files, aliases=()):
    """files: [(ext, bytes), ...] written as ONE version: <base>_vNNN.<ext> for every ext, N = 1 + the highest version
    present (aliases count too). Each file: <name>.tmp, then os.rename (no replace); a name taken meanwhile => the
    renamed ones go back to .tmp and the whole set retries at N+1. Returns the written paths."""
    os.makedirs(folder, exist_ok=True)
    n = highest_version(folder, base, aliases) + 1
    n = max(n, 1)
    tmps = []
    try:
        for ext, data in files:
            tmp = os.path.join(folder, f"{base}_v{n:03d}.{ext}.tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            tmps.append((ext, tmp))
        for _attempt in range(1000):
            done = []
            try:
                for ext, tmp in tmps:
                    dst = os.path.join(folder, f"{base}_v{n:03d}.{ext}")
                    os.rename(tmp, dst)
                    done.append((tmp, dst))
                return [dst for _, dst in done]
            except FileExistsError:
                for tmp, dst in done:              # put the set back, try the next number
                    os.rename(dst, tmp)
                n = max(n + 1, highest_version(folder, base, aliases) + 1)
        raise RuntimeError(f"no free version number for {base} in {folder}")
    finally:
        for _, tmp in tmps:
            if os.path.exists(tmp):
                os.remove(tmp)
