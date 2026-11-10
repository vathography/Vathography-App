"""Builds packaging\\licences\\Rust-crates.txt from harvest.json (Vathography App 0.1.7, step 12 row 105, ruling A).
Per crate, the licence choice with the fewest obligations: a no-notice licence (Unlicense, CC0-1.0, 0BSD, MIT-0) >
Apache-2.0 (its text = licences\\Apache-2.0.txt; the crate's own NOTICE files are added, Apache 2.0 section 4d) >
the crate's own licence texts. An AND expression needs every part: its extra parts' texts come from the crate's files.
A crate that ships no licence file gets the standard text of its licence, naming the crate (no notice to copy)."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
h = json.load(open(os.path.join(HERE, "crates", "harvest.json"), encoding="utf-8"))
clos = json.load(open(os.path.join(HERE, "crates", "closures.json"), encoding="utf-8"))
OUT = sys.argv[1]
NO_NOTICE = ("Unlicense", "CC0-1.0", "0BSD", "MIT-0")
STD_MIT = ("Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated "
           "documentation files (the \"Software\"), to deal in the Software without restriction, including without "
           "limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the "
           "Software, and to permit persons to whom the Software is furnished to do so, subject to the following "
           "conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or "
           "substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, "
           "EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
           "PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, "
           "DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN "
           "CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.")
APACHE_HEAD = re.compile(r"^\s*Apache License\s*Version 2\.0, January 2004", re.S)


def parse(expr):
    """-> list of AND-groups, each a list of OR alternatives (SPDX subset used by crates.io; '/' = OR)."""
    e = (expr or "").replace("/", " OR ")
    groups = []
    for part in re.split(r"\s+AND\s+", e):
        alts = [a.strip(" ()") for a in re.split(r"\s+OR\s+", part.strip(" ()")) if a.strip(" ()")]
        groups.append(alts)
    return groups


def choose(alts):
    for a in alts:
        if a in NO_NOTICE:
            return a
    for a in alts:
        if a.startswith("Apache-2.0"):
            return "Apache-2.0"
    return alts[0]


def is_plain_apache(text):
    return bool(APACHE_HEAD.match(text)) and "Copyright" not in text[:400]


deep = json.load(open(os.path.join(HERE, "crates", "deep.json"), encoding="utf-8"))
IJG_TEXT = deep["turbojpeg-sys 1.2.0"]["libjpeg-turbo/README.ijg"]
# bundled C / C++ code inside a crate keeps ITS licence whatever the crate's own choice (read at full depth, deep.json)
BUNDLED = {"turbojpeg-sys 1.2.0": deep["turbojpeg-sys 1.2.0"], "apriltag-sys 0.3.0": deep["apriltag-sys 0.3.0"]}
# a crate that ships no licence file: its repository's licence file (read 2026-11-09)
REPO_FILE = {"alloc-stdlib 0.2.2": {"LICENSE (dropbox/rust-alloc-no-stdlib)":
                                    open(os.path.join(HERE, "crates", "alloc-stdlib-LICENSE.txt"), encoding="utf-8").read()},
             # crates.io says BSD-3-Clause, the crate's own LICENSE file is Apache-2.0 (relicensed 2020, d81a895a87):
             # both honoured; the BSD text = nalgebra's original LICENSE (commit 8ecd37ebc8)
             "nalgebra 0.32.6": {"LICENSE before 2020 (dimforge/nalgebra 8ecd37ebc8)":
                                 open(os.path.join(HERE, "crates", "nalgebra-BSD-LICENSE.txt"), encoding="utf-8").read()}}
# facts only (user's rule 2026-11-09): the declared NCSA part = LLVM's own licence file (Apache-2.0 WITH LLVM-exception
# + the legacy University of Illinois/NCSA licence), no interpretation
NOTES = {"NCSA": ("The libFuzzer sources bundled in libfuzzer-sys 0.4.13 carry the header 'SPDX-License-Identifier: "
                  "Apache-2.0 WITH LLVM-exception' (libfuzzer/FuzzerCorpus.h at tag 0.4.13). LLVM's licence file "
                  "(llvm/LICENSE.TXT), with its legacy University of Illinois/NCSA licence:\n\n"
                  + open(os.path.join(HERE, "crates", "llvm-LICENSE.TXT"), encoding="utf-8").read())}

rows, texts = [], {}          # texts: normalised text -> (title, [crates])
for key in sorted(h, key=lambda k: (h[k]["name"].lower(), k)):
    r = h[key]
    groups = parse(r.get("license"))
    chosen = [choose(g) for g in groups] if r.get("license") else ["(not stated)"]
    files = dict(r.get("files") or {})
    files.update(REPO_FILE.get(key, {}))
    rows.append((r["name"], r["version"], r.get("license") or f"file: {r.get('license_file')}", " AND ".join(chosen)))
    need = [c for c in chosen if c not in NO_NOTICE]
    keep = {f"bundled: {f}": t for f, t in BUNDLED.get(key, {}).items()}
    for c in need:
        if c == "IJG":
            keep["IJG licence (libjpeg-turbo README.ijg)"] = IJG_TEXT
            continue
        if c in NOTES:
            keep[f"note on {c}"] = NOTES[c]
            continue
        if c == "Apache-2.0":
            for f, t in files.items():
                if re.search(r"(?i)notice", f.rsplit("/", 1)[-1]):
                    keep[f] = t
            continue
        picked = {f: t for f, t in files.items() if not is_plain_apache(t)
                  and not re.search(r"(?i)apache", f.rsplit("/", 1)[-1])}
        if c == "MIT":
            mit = {f: t for f, t in picked.items() if re.search(r"(?i)mit", f) or "Permission is hereby granted" in t}
            picked = mit or picked
        if picked:
            keep.update(picked)
        elif c == "MIT":
            keep["(standard text)"] = f"MIT License\n\nCopyright (c) the {r['name']} authors\n\n{STD_MIT}"
        else:
            keep["(no licence file)"] = (f"{r['name']} {r['version']} is licensed under {c}; the published crate ships no "
                                         f"licence file. The standard {c} text applies, naming the {r['name']} authors.")
    for f, t in keep.items():
        norm = "\n".join(l.rstrip() for l in t.replace("\r\n", "\n").strip().splitlines())
        title = f"{r['name']} {r['version']}: {f}"
        texts.setdefault(norm, [title, []])[1].append(f"{r['name']} {r['version']}")

roots = {"kornia-py": "kornia_rs 0.2.0 (kornia/kornia-rs, tag v0.2.0, Cargo.lock)",
         "safetensors-python": "safetensors 0.8.0 (PyPI source distribution, bindings/python/Cargo.lock)"}
lines = ["Rust crates compiled into runtime\\Lib\\site-packages\\kornia_rs and runtime\\Lib\\site-packages\\safetensors",
         "",
         "These two Python packages are compiled from Rust. Their binaries include the Rust libraries (crates) below, each "
         "under its own licence. The list is the full dependency set of each package as recorded in its Cargo.lock:",
         *[f"- {v}: {len(clos[k])} crates" for k, v in roots.items()],
         "It may name crates that the Windows build does not compile in (optional or other-platform dependencies).",
         "",
         "Where a crate offers a choice of licences, it is used here under the one named in the last column. Apache-2.0: "
         "the licence text is licences\\Apache-2.0.txt; NOTICE files of those crates follow below. Crates used "
         "under Unlicense, CC0-1.0, 0BSD or MIT-0 have no text below. Every other licence text, with its copyright lines, follows below, once per "
         "distinct text, with the crates it belongs to.",
         "",
         "CRATE | VERSION | LICENCE (as published) | USED UNDER",
         *[f"{n} | {v} | {l} | {c}" for n, v, l, c in rows],
         ""]
for norm, (title, who) in sorted(texts.items(), key=lambda x: x[1][0].lower()):
    lines += ["=" * 100, f"{title}" + (f"  (same text: {', '.join(who[1:])})" if len(who) > 1 else ""), "=" * 100, norm, ""]
out = "\n".join(lines) + "\n"
open(OUT + ".tmp", "w", encoding="utf-8", newline="\n").write(out)
os.replace(OUT + ".tmp", OUT)
used = {}
for _, _, _, c in rows:
    for p in c.split(" AND "):
        used[p] = used.get(p, 0) + 1
print(f"{len(rows)} crates, {len(texts)} distinct texts, {len(out) / 1024:.0f} KB | used under: {sorted(used.items(), key=lambda x: -x[1])}")
print("standard / no-file entries:", sum(1 for t, (title, w) in texts.items() if "(standard text)" in title or "(no licence file)" in title))
