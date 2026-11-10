"""Vathography App 0.1.7 (step 12, second reader on rc4 #2/#13, ruled 2026-11-09 "ship upstream's full lists"):
builds licences\\SciPy-1.15.3-bundled.txt and licences\\NumPy-2.2.5-bundled.txt from each project's OWN source release
(PyPI sdist, the exact version in the zip): its LICENSES_bundled.txt verbatim, then every licence text it refers to,
verbatim, each with the path it was read from. A reference to a SOURCE file (.py .h .hpp .c .pyx ...) = that file's
leading licence header only. SciPy's list carries stale paths: MOVED maps each to where 1.15.3 has it (read by hand,
2026-11-09); EXTRA adds the licence files of code bundled inside a bundled project (HiGHS' own extern/). A reference
with no file in the release is written as such. Anything else unresolved STOPS the build of the file.
Usage: python make_bundled.py <sdist folder> <licences folder>"""
import os
import re
import sys
import tarfile

src, out_dir = sys.argv[1:3]
JOBS = (("scipy-1.15.3.tar.gz", "SciPy-1.15.3-bundled.txt", "SciPy 1.15.3"),
        ("numpy-2.2.5.tar.gz", "NumPy-2.2.5-bundled.txt", "NumPy 2.2.5"),
        ("pillow-12.1.1.tar.gz", "Pillow-12.1.1-vendored.txt", "Pillow 12.1.1"))
MOVED = {"scipy/io/_fast_matrix_market/fast_matrix_market/dependencies/ryu/LICENSE-BOOST":
         "scipy/io/_fast_matrix_market/fast_matrix_market/dependencies/ryu/LICENSE-Boost",
         "scipy/optimize/_highs/LICENCE": "subprojects/highs/LICENSE.txt",
         "scipy/_lib/boost_math/LICENSE.txt": "scipy/_lib/boost_math/LICENSE",
         "scipy/_lib/array-api-compat/LICENCE": "scipy/_lib/array_api_compat/LICENSE",
         "scipy/_build_utils/tempita/LICENCE.txt": "scipy/_build_utils/tempita/LICENSE.txt",
         "numpy/_build_utils/tempita/LICENCE.txt": "numpy/_build_utils/tempita/LICENSE.txt",
         "scipy/sparse/linalg/dsolve/SuperLU/License.txt": "scipy/sparse/linalg/_dsolve/SuperLU/License.txt",
         "scipy/sparse/linalg/eigen/arpack/ARPACK/COPYING": "scipy/sparse/linalg/_eigen/arpack/ARPACK/COPYING",
         "scipy/spatial/qhull/COPYING.txt": "scipy/spatial/qhull_src/COPYING.txt",
         "scipy/_lib/uarray/LICENSE": "scipy/_lib/_uarray/LICENSE",
         "scipy/stats/_rcont/logfactorial.h": "scipy/stats/libnpyrandom/logfactorial.h",
         "scipy/special/special/third_party/kokkos/mdspan.hpp": "scipy/special/xsf/third_party/kokkos/mdspan.hpp",
         # code renamed: the list's file is gone, its code + notice live here (searched for the authors, 2026-11-09)
         "scipy/interpolate/_aaa.py": "scipy/interpolate/_bary_rational.py",
         "scipy/optimize/lbfgsb/README": "scipy/optimize/__lbfgsb.c"}
ABSENT = {"scipy/linalg/src/id_dist/doc/doc.tex":
          "not in the SciPy 1.15.3 source release (its id_dist sources are no longer there); the licence named in the "
          "list above applies"}
EXTRA = {"subprojects/highs/LICENSE.txt": ["subprojects/highs/extern/LICENCE_1_0.txt",
                                           "subprojects/highs/extern/filereaderlp/LICENSE",
                                           "subprojects/highs/extern/pdqsort/license.txt"]}
SOURCE_EXT = (".py", ".pyx", ".pxd", ".h", ".hpp", ".c", ".cc", ".cpp", ".f", ".f90")
REF = re.compile(r"(?i)for (?:details|license text),? see\s+(?:the\s+)?(header (?:inside|of)\s+)?([\w./\-]+?)[.,]?(?=\s|$)")
# the lists are incomplete (SciPy's omits PROPACK, cobyqa, ...; NumPy's omits pocketfft, highway, ...): EVERY licence /
# copying / notice file inside the package's own source tree is added after the list's references
TREE = {"scipy-1.15.3.tar.gz": ("scipy/", "subprojects/"), "numpy-2.2.5.tar.gz": ("numpy/",),
        "pillow-12.1.1.tar.gz": ("src/thirdparty/",)}
# vendored source files whose licence is in their header (Pillow has no LICENSES_bundled.txt; its own LICENSE covers
# the libraries it links, not src/thirdparty/)
HEADERS = {"pillow-12.1.1.tar.gz": ["src/thirdparty/pythoncapi_compat.h"]}   # fribidi-shim = Pillow's own code
TREE_FILE = re.compile(r"(?i)^(licen[cs]e|copying|notice|copyright)[^/]*$")


def header_of(text, path=""):
    """The licence header at the top of a source file: leading comment lines (// and /* ... */ in C / C++; # and a
    leading docstring in Python; ! or c in Fortran), up to the first line of code. In C, # = the preprocessor = code."""
    py = path.lower().endswith((".py", ".pyx", ".pxd"))
    fortran = path.lower().endswith((".f", ".f90"))
    out, in_block, in_doc = [], False, None
    for line in text.splitlines():
        s = line.strip()
        if in_doc:
            out.append(line)
            if in_doc in s:
                break
            continue
        if in_block:
            out.append(line)
            if "*/" in s:
                in_block = False
            continue
        if not s:
            out.append(line)
            continue
        if py and s.startswith(('"""', "'''")):
            q = s[:3]
            out.append(line)
            if s.count(q) >= 2 and len(s) > 3:
                break
            in_doc = q
            continue
        if s.startswith("/*"):
            out.append(line)
            in_block = "*/" not in s
            continue
        if s.startswith("//") or (py and s.startswith("#")) or (fortran and (s.startswith("!") or re.match(r"(?i)^c\s", line))):
            out.append(line)
            continue
        if not py and not fortran and s.startswith("#") and not any(x.strip() for x in out):
            continue                              # an include guard before the licence block (Kokkos mdspan.hpp)
        break
    return "\n".join(out).strip()


rc = 0
for tgz, out_name, title in JOBS:
    with tarfile.open(os.path.join(src, tgz)) as t:
        files = {m.name.split("/", 1)[1]: m for m in t.getmembers() if m.isfile() and "/" in m.name}

        def read(p):
            b = t.extractfile(files[p]).read()
            try:
                return b.decode("utf-8")
            except UnicodeDecodeError:
                return b.decode("latin-1")
        lb = read("LICENSES_bundled.txt") if "LICENSES_bundled.txt" in files else ""
        if lb:
            parts = [f"{title}: its bundled third-party works, as listed by {title} itself (LICENSES_bundled.txt of its "
                     f"source release), followed by each licence text that list refers to, read from the same source "
                     f"release.", "", "=" * 100, "LICENSES_bundled.txt", "=" * 100, lb.strip(), ""]
        else:
            parts = [f"{title}: the third-party code it vendors in its own source tree (src/thirdparty), with each "
                     f"licence text, read from its source release.", ""]
            for hp in HEADERS.get(tgz, []):
                parts += ["=" * 100, hp + " (its licence header)", "=" * 100, header_of(read(hp), hp), ""]
        unresolved, n_texts = [], 0
        for m in REF.finditer(lb):
            header, ref = bool(m.group(1)), m.group(2)
            if ref in ABSENT:
                parts += ["=" * 100, f"{ref}: {ABSENT[ref]}", "=" * 100, ""]
                continue
            path = ref if ref in files else MOVED.get(ref)
            if path is None or path not in files:
                unresolved.append(ref)
                continue
            for p in [path] + EXTRA.get(path, []):
                as_header = header or p.lower().endswith(SOURCE_EXT)
                text = header_of(read(p), p) if as_header else read(p).strip()
                if not text:
                    unresolved.append(p + " (no header text)")
                    continue
                label = p + (" (its licence header)" if as_header else "") + (f"  [listed as {ref}]" if p == path and path != ref else "")
                parts += ["=" * 100, label, "=" * 100, text, ""]
                n_texts += 1
        done = {l.split(" (its licence header)")[0].split("  [listed as")[0] for l in parts}
        tree = sorted(n for n in files if n.startswith(TREE[tgz]) and "/.github/" not in n
                      and TREE_FILE.search(n.rsplit("/", 1)[-1]) and n not in done
                      and not n.lower().endswith((".py", ".c", ".h", ".rst", ".qbk", ".html", ".yml")))
        if tree:
            parts += ["=" * 100, f"Every other licence file inside the {title} source tree ({len(tree)} files)", "=" * 100, ""]
            for n in tree:
                parts += ["=" * 100, n, "=" * 100, read(n).strip(), ""]
                n_texts += 1
        if unresolved:
            print(f"{title}: UNRESOLVED {unresolved}")
            rc = 1
            continue
        out = "\n".join(parts) + "\n"
        p = os.path.join(out_dir, out_name)
        open(p + ".tmp", "w", encoding="utf-8", newline="\n").write(out)
        os.replace(p + ".tmp", p)
        print(f"{title}: {lb.count('Name:')} entries, {n_texts} texts, {len(out) // 1024} KB -> {out_name}")
sys.exit(rc)
