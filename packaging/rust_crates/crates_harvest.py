"""Licence harvest for the Rust crates of kornia_rs 0.2.0 + safetensors 0.8.0 (Vathography App 0.1.7, step 12 row 105).
Input: closures.json (crate name + version from each Cargo.lock's dependency closure). For each crate: the crates.io API
licence expression (1 request / s), then the published .crate (static.crates.io), read IN MEMORY: only files whose
name says licence / notice / copying / copyright / authors are kept as text. Nothing is unpacked to disk or run.
Output: harvest.json (resumable: crates already done are skipped)."""
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "crates")
UA = {"User-Agent": "vathography-app-licence-check (https://github.com/vathography)"}
KEEP = re.compile(r"(?i)(^|/)(licen[cs]e|notice|copying|copyright|authors|unlicense|patents)[^/]*$")
out_p = os.path.join(D, "harvest.json")
done = json.load(open(out_p, encoding="utf-8")) if os.path.exists(out_p) else {}
clos = json.load(open(os.path.join(D, "closures.json"), encoding="utf-8"))
crates = sorted({tuple(c) for v in clos.values() for c in v})
print(f"{len(crates)} crates, {len(done)} already done", flush=True)


def get(url, timeout=60):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as ex:                       # noqa: BLE001
            if i == 3:
                raise
            time.sleep(2 + 3 * i)


for n, (name, ver) in enumerate(crates, 1):
    key = f"{name} {ver}"
    if key in done:
        continue
    rec = {"name": name, "version": ver}
    try:
        meta = json.loads(get(f"https://crates.io/api/v1/crates/{name}/{ver}"))["version"]
        rec["license"] = meta.get("license")
        rec["license_file"] = meta.get("license_file")
        rec["repository"] = None
        time.sleep(1.0)
        blob = get(f"https://static.crates.io/crates/{name}/{name}-{ver}.crate")
        texts = {}
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
            for m in t.getmembers():
                rel = m.name.split("/", 1)[-1]
                if m.isfile() and KEEP.search(rel) and rel.count("/") <= 2 and m.size < 400_000:
                    raw = t.extractfile(m).read()
                    try:                      # some crates ship Windows-1252 texts (rav1e PATENTS)
                        texts[rel] = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        texts[rel] = raw.decode("cp1252")
        rec["files"] = texts
    except Exception as ex:                           # noqa: BLE001
        rec["error"] = f"{type(ex).__name__}: {ex}"
    done[key] = rec
    if n % 20 == 0 or n == len(crates):
        json.dump(done, open(out_p + ".tmp", "w", encoding="utf-8"), indent=0)
        os.replace(out_p + ".tmp", out_p)
        print(f"{n}/{len(crates)}", flush=True)
json.dump(done, open(out_p + ".tmp", "w", encoding="utf-8"), indent=0)
os.replace(out_p + ".tmp", out_p)
err = [k for k, r in done.items() if "error" in r]
print(f"done: {len(done)} crates, {len(err)} errors: {err[:10]}", flush=True)
