import tomllib, re, sys, json, os
def closure(lock, root):
    pk = tomllib.load(open(lock, "rb"))["package"]
    byname = {}
    for p in pk:
        byname.setdefault(p["name"], []).append(p)
    def resolve(dep):
        parts = dep.split(" ")
        cands = byname.get(parts[0], [])
        if len(parts) > 1:
            cands = [c for c in cands if c["version"] == parts[1]] or cands
        return cands[0]
    seen, todo = {}, [byname[root][0]]
    while todo:
        p = todo.pop()
        k = (p["name"], p["version"])
        if k in seen: continue
        seen[k] = p
        todo += [resolve(d) for d in p.get("dependencies", [])]
    return seen
pat = re.compile(rb"index\.crates\.io-[0-9a-f]+[/\x5c]([A-Za-z0-9_-]+?)-(\d+\.\d+\.\d+[A-Za-z0-9_.+-]*)[/\x5c]")
def strings(pyd):
    return {(n.decode(), v.decode()) for n, v in pat.findall(open(pyd, "rb").read())}
rt = sys.argv[1]
out = {}
for lock, root, pyd in (("kornia-rs-0.2.0.Cargo.lock", "kornia-py", "kornia_rs/kornia_rs.cp310-win_amd64.pyd"),
                        ("safetensors-0.8.0.Cargo.lock", "safetensors-python", "safetensors/_safetensors_rust.pyd")):
    c = closure(lock, root)
    reg = {k: p for k, p in c.items() if "source" in p}
    s = strings(os.path.join(rt, pyd))
    print(f"{root}: closure {len(c)} (registry {len(reg)}) | crates in the binary {len(s)} | in the binary, not in closure: {sorted(s - set(c))}")
    out[root] = sorted(reg)
json.dump(out, open("closures.json", "w"), indent=0)
