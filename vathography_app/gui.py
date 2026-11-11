# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""gui.py — the Vathography App window (tkinter + tkinterdnd2).
The window never imports torch: the work runs in ONE worker process per batch = `python -m vathography_app depth|relief
... --progress-json` (the CLI path), read line by line. Cancel ends the worker (the GPU is freed at once;
the .tmp of the photo in progress is deleted). GPU / RAM / disk checks without torch (nvidia-smi, Win32).
House rule: no always-on hint lines; a sentence appears only on a problem.
"""
import ctypes
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, ttk
from tkinter import font as tkfont

from . import __version__, cli, download, models, outnames, third_party

APP_NAME = "Vathography App"
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.dirname(PKG_DIR)
APP_HOME = models.APP_HOME          # the portable package = <APP_HOME>\app\vathography_app ; the dev repo = <APP_HOME>
SETTINGS = os.path.join(APP_HOME, "settings.json")
RUN_LOG = os.path.join(APP_HOME, "last_run.log")
DEFAULT_MODELS = models.DEFAULT_DIR                     # beside the app
ICON = os.path.join(PKG_DIR, "assets", "vathography_app.ico")
# the notices shipped beside the app (packaging\NOTICE.txt in the dev repo)
NOTICE = next((p for p in (os.path.join(APP_HOME, "NOTICE.txt"), os.path.join(APP_HOME, "packaging", "NOTICE.txt"))
               if os.path.isfile(p)), os.path.join(APP_HOME, "NOTICE.txt"))
BUDGET_MIN = 4.6            # GiB: one input per weight pass (backend BASE 3.9 + MARGIN 0.7); measured on a 6 GB card
BUDGET_MAX = 6.5            # GiB: = BUDGET_MIN, the budget is fixed (measured: no speed gain above 6.5)
RAM_WARN_GB = 16
MIN_CC = 8.0                # Ampere: torch 2.10 cu128 + the measured cards

# the brand = the Vathography site's DARK theme (website css/style.css :root[data-theme="dark"]) + its fonts as Windows
# renders them (--serif -> Palatino Linotype, --sans -> Segoe UI; both ship with Windows).
MASTHEAD = "Vathography App"
ISSUES_URL = "https://github.com/vathography/Vathography-App/issues"
BG, BG_ALT, SURFACE = "#121110", "#191715", "#1d1b19"
TEXT, TEXT_SOFT, TEXT_FAINT = "#eeebe6", "#b3aca2", "#837c72"
LINE, ACCENT, ACCENT_SOFT, ON_ACCENT = "#2e2b27", "#d8ac4d", "#3a3020", "#1b1a18"
TAB_BACK, TAB_DISABLED = "#0b0a09", "#4a4540"   # an unselected tab (behind the window) · a tab with nothing to show
SERIF, SANS = "Palatino Linotype", "Segoe UI"
COMBO_WIDTH = 44            # characters: the longest choice + room for a chosen folder's path
BESIDE, CHOOSE = "Next to each photo", "Choose a folder…"
BESIDE_FILE = "Next to each file"                       # the Relief tab's inputs are depth maps, not photos
# Output, highest precision first (user 2026-10-31; a 0..1 depth map: 16-bit PNG step 1.5e-5 < half EXR 2.4e-4 near
# 1.0, measured); the first one is the default. Value = the CLI's --format.
OUTPUTS = (("exr32", "32-bit EXR"), ("png16", "16-bit PNG"), ("exr16", "16-bit EXR"), ("png8", "8-bit PNG"))
REMOVE_MARK = "✕"          # the last column of a row: click = that row out of the list

PREVIEW_BOX = (440, 340)    # pixels: the preview's starting size; it grows with the window
PREVIEW_MAX = 1600          # pixels: the long side kept in memory per previewed picture

from PIL import Image, ImageTk  # noqa: E402  (Pillow: in the runtime already; no torch)

try:
    from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD
except Exception:          # the window still works without drag and drop (the Add buttons)
    TkinterDnD = None


def fit_left(text, width, fnt):
    """`text` if it fits `width` pixels, else '…' + its longest END that fits (a path keeps its file name)."""
    if fnt.measure(text) <= width:
        return text
    lo, hi = 1, len(text)                               # smallest cut k with '…' + text[k:] fitting
    while lo < hi:
        k = (lo + hi) // 2
        if fnt.measure("…" + text[k:]) <= width:
            hi = k
        else:
            lo = k + 1
    return "…" + text[lo:]


def preview_image(path):
    """A picture file -> an 8-bit RGB PIL image, long side <= PREVIEW_MAX. 8-bit files through Pillow (fast, JPEG drafts
    at reduced size); 16-bit PNG / TIFF and EXR through the app's own reader (photo_io = what the engine reads), values
    clipped to 0..1 (depth maps are written 0..1, near = bright)."""
    try:
        im = Image.open(path)
        if im.mode in ("RGB", "RGBA", "L", "LA", "P", "CMYK"):
            im.draft("RGB", (PREVIEW_MAX, PREVIEW_MAX))
            im = im.convert("RGB")
            im.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
            return im
    except Exception:
        pass
    import numpy as np
    from . import photo_io
    x = photo_io.read_image_float(path)
    step = max(1, max(x.shape[:2]) // (2 * PREVIEW_MAX))           # cheap first reduction, Pillow does the rest
    u8 = (np.clip(x[::step, ::step], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(np.ascontiguousarray(u8), "RGB")
    im.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return im


# ---------------------------------------------------------------- settings / system checks (no torch)
def load_settings():
    try:
        return json.load(open(SETTINGS, encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d):
    try:
        tmp = SETTINGS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, SETTINGS)
    except Exception:
        pass


def gpu_info():
    """-> dict(name, total_gib, free_gib, cc) of GPU 0, or None (no NVIDIA driver)."""
    exe = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        name, total, free, cc = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return dict(name=name, total_gib=float(total) / 1024, free_gib=float(free) / 1024, cc=float(cc))
    except Exception:
        return None


def ram_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 1e9


def preflight():
    """-> (budget_gib or None, [sentences]). None = cannot run."""
    g = gpu_info()
    if g is None:
        return None, ["No NVIDIA graphics card was found (or its driver is missing). The app needs one."]
    if g["cc"] < MIN_CC:
        return None, [f"{g['name']} is too old for the app (it needs an NVIDIA RTX 30-series or newer)."]
    budget = min(g["free_gib"] - 0.5, BUDGET_MAX)
    if budget < BUDGET_MIN:
        return None, [f"Only {g['free_gib']:.1f} GB of graphics memory is free on {g['name']}; the app needs about "
                      f"{BUDGET_MIN + 0.5:.1f} GB. Close other programs that use the graphics card and try again."]
    notes = []
    if ram_gb() < RAM_WARN_GB:
        notes.append(f"This computer has {ram_gb():.0f} GB of memory; large photos may be slow or fail below {RAM_WARN_GB} GB.")
    return round(budget, 1), notes


def worker_python():
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":
        exe = os.path.join(os.path.dirname(exe), "python.exe")
    return exe


# ---------------------------------------------------------------- the worker (one process per batch)
class Worker:
    """Runs one CLI batch; events land in `self.q` as dicts; stderr goes to last_run.log."""

    def __init__(self, args):
        self.q = queue.Queue()
        # -E -s: a PYTHONPATH / PYTHONHOME / user site-packages on the user's machine must not reach the runtime;
        # the package is found from cwd (-m); -u replaces PYTHONUNBUFFERED (ignored under -E)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.log = open(RUN_LOG, "w", encoding="utf-8")
        self.log.write(" ".join(args) + "\n\n")
        self.log.flush()
        self.t0 = time.time()                            # [+12.3 s] on every log line
        self._stage_key = None
        try:
            self.p = subprocess.Popen([worker_python(), "-E", "-s", "-u", "-m", "vathography_app"] + args + ["--progress-json"],
                                      cwd=PKG_PARENT, env=env, stdout=subprocess.PIPE, stderr=self.log,
                                      text=True, encoding="utf-8", errors="replace",
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError:
            self.log.close()
            raise
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                ev = {"ev": "log", "msg": line}
            stamp = f"[+{time.time() - self.t0:.1f} s] "
            if ev.get("ev") == "log":
                self.log.write(stamp + ev["msg"] + "\n")
                self.log.flush()
            elif ev.get("ev") == "stage":                # one line per new stage (digits aside) + its last event
                key = (ev.get("i"), "".join("#" if c.isdigit() else c for c in ev["name"]))
                last = ev.get("total") and ev.get("done") == ev.get("total")
                if key != self._stage_key or last:
                    self._stage_key = key
                    pct = f" ({ev['done']}/{ev['total']})" if ev.get("total") else ""
                    self.log.write(f"{stamp}stage: {ev['name']}{pct}\n")
                    self.log.flush()
            self.q.put(ev)
        self.q.put({"ev": "exit", "code": self.p.wait()})

    def cancel(self):
        if self.p.poll() is None:
            self.p.kill()


# ---------------------------------------------------------------- the window
class App:
    def __init__(self, root, initial):
        self.root = root
        self.ui = queue.Queue()                          # callables from worker threads, run by poll() on the Tk thread
        self.s = load_settings()
        self.models_dir = self.s.get("models_dir", DEFAULT_MODELS)
        self.worker = None
        self.job = None                                 # dict(kind, items, out_dir, suffixes, current)
        root.title(APP_NAME)

        self.masthead = tk.Frame(root, bg=BG)          # the brand strip (= the site's masthead: serif title, centred)
        self.masthead.pack(fill="x")
        self.title_label = tk.Label(self.masthead, text=MASTHEAD, bg=BG, fg=TEXT, font=(SERIF, 28))
        self.title_label.pack(pady=(18, 14))
        # small, on the title's baseline, placed so that the title itself stays centred
        self.version_label = tk.Label(self.masthead, text=__version__, bg=BG, fg=TEXT_FAINT, font=(SANS, 9), bd=0, padx=0, pady=0)
        baseline = tkfont.Font(font=(SERIF, 28)).metrics("ascent") + self.title_label.winfo_pixels(self.title_label.cget("pady")) + self.title_label.winfo_pixels(self.title_label.cget("bd"))
        self.version_label.place(in_=self.title_label, relx=1.0, x=4, y=baseline + tkfont.Font(font=(SANS, 9)).metrics("descent"), anchor="sw")
        tk.Frame(self.masthead, bg=LINE, height=1).pack(fill="x", padx=8)
        # the one menu: Models folder · Troubleshooting log · About (a gear at the masthead's right end)
        self.gear = tk.Label(self.masthead, text="⚙", bg=BG, fg=TEXT_FAINT, font=(SANS, 18), cursor="hand2")
        self.gear.place(relx=1.0, x=-18, rely=0.45, anchor="e")
        self.gear.bind("<Enter>", lambda e: self.gear.configure(fg=ACCENT))
        self.gear.bind("<Leave>", lambda e: self.gear.configure(fg=TEXT_FAINT))
        self.gear.bind("<Button-1>", lambda e: self.open_menu())
        self.menu = tk.Menu(root, tearoff=0, bg=SURFACE, fg=TEXT, activebackground=ACCENT_SOFT,
                            activeforeground=ACCENT, bd=0, font=(SANS, 10))
        self.menu.add_command(label="Set models folder…", command=self.change_models)
        self.menu.add_command(label="Troubleshooting log", command=self.show_log)
        self.menu.add_separator()
        self.menu.add_command(label="About Vathography App", command=self.show_about)

        self.banner = ttk.Frame(root, padding=(10, 8), style="Notice.TFrame")  # models missing / download progress
        self.banner_text = tk.StringVar()
        ttk.Label(self.banner, textvariable=self.banner_text, wraplength=420,
                  style="Notice.TLabel").pack(side="left", fill="x", expand=True)
        self.btn_dl = ttk.Button(self.banner, text="Download", command=self.start_download)
        self.btn_have = ttk.Button(self.banner, text="I already have them…", command=self.pick_models)
        self.dl_bar = ttk.Progressbar(self.banner, length=160, mode="determinate")
        self.check = None                               # the "I already have them…" check while it runs (its banner text)
        # PyTorch + OpenCV (downloaded from their publishers at first start, third_party.py): their own row, shown
        # only while one is missing; nothing is downloaded before its Download click
        self.tp_banner = ttk.Frame(root, padding=(10, 8), style="Notice.TFrame")
        self.tp_text = tk.StringVar()
        self.tp_label = ttk.Label(self.tp_banner, textvariable=self.tp_text, wraplength=520, style="Notice.TLabel")
        self.tp_label.pack(side="left", fill="x", expand=True)
        self.btn_tp = ttk.Button(self.tp_banner, text="Download", command=self.start_third_party)
        self.btn_tp.pack(side="right")
        self.tp_bar = ttk.Progressbar(self.tp_banner, length=160, mode="determinate")
        self.tp_job = None                              # the PyTorch + OpenCV download / install while it runs
        self.gpu_note = None                            # the graphics-card check shown in that row (None = not run yet)

        nb = self.nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=8, pady=(10, 0))
        self.pv_cache = {}                              # (path, mtime) -> PIL image reduced to PREVIEW_MAX
        self.tab_depth = self._tab(nb, "depth", "Depth Estimation", "Drop photos or a folder",
                                   ("Photograph", "Depth map"))
        self.tab_relief = self._tab(nb, "relief", "Relief Accentuation", "Drop Blender add-on exports or depth maps",
                                    ("Base depth map", "Accentuated Relief"))
        self._depth_options(self.tab_depth)
        self._relief_options(self.tab_relief)
        self._tab_now = self.tab_depth                  # one divider position for both pages (user 2026-11-11)
        nb.bind("<<NotebookTabChanged>>", self._tab_changed, add="+")

        # the line under the list: errors and warnings only (progress is written on the photo's own row, never here,
        # so "stopped unexpectedly" always shows)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=(10, 6), wraplength=740, style="Status.TLabel").pack(fill="x")

        self.check_models()
        if third_party.over_old_version():               # a problem only: a sentence under the list
            self.status.set("This folder was unzipped over an older version of the app. Unzip the new version into a "
                            "new, empty folder, then move the models and third_party folders into it.")
        # size from the content, not fixed pixels: the type grows with the display scaling (125 %, 150 %)
        root.update_idletasks()
        w, h = max(760, root.winfo_reqwidth()), max(600, root.winfo_reqheight())
        root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
        root.geometry(f"{w}x{h}")
        if initial:
            self.add(self.tab_depth, initial)
        root.after(100, self.poll)
        root.after(1000, self._clock)

    # -- layout
    def _tab(self, nb, kind, title, drop_text, pv_titles):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text=title)
        f.kind = kind
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        # list | preview with a draggable divider between them (user 2026-11-11: one handle on the EDGE resizes both;
        # the list's own column handle is switched off below)
        paned = ttk.PanedWindow(f, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")
        f.paned = paned
        left = ttk.Frame(paned, padding=(0, 0, 5, 0))
        paned.add(left, weight=1)
        box = ttk.Frame(left)                           # the list + its scrollbar (shown only when rows overflow)
        box.pack(fill="both", expand=True)
        tree = ttk.Treeview(box, columns=("status", "remove"), height=8, selectmode="extended")
        sb = ttk.Scrollbar(box, orient="vertical", command=tree.yview)

        def yset(lo, hi, sb=sb):
            sb.set(lo, hi)
            if float(lo) <= 0.0 and float(hi) >= 1.0:
                sb.pack_forget()
            elif not sb.winfo_manager():
                sb.pack(side="right", fill="y", before=tree)
        tree.configure(yscrollcommand=yset)
        f.scrollbar = sb
        tree.heading("#0", text=drop_text)
        tree.heading("status", text="")
        tree.heading("remove", text="")
        # widths are set by _layout (Issue ID 2026-10-31 b): status = as wide as its text, the name gets the rest
        tree.column("#0", width=360, minwidth=60, stretch=False)
        tree.column("status", width=4, minwidth=4, anchor="w", stretch=False)
        tree.column("remove", width=34, minwidth=34, anchor="center", stretch=False)
        tree.bind("<Configure>", lambda e, f=f: self._layout(f), add="+")
        tree.pack(side="left", fill="both", expand=True)
        f.tree = tree
        f.paths = {}                                    # item id -> path
        f.outputs = {}                                  # item id -> [written files] (this session's runs)
        tree.bind("<<TreeviewSelect>>", lambda e, f=f: self._selected(f))
        tree.bind("<Button-1>", lambda e, f=f: self._click_remove(f, e))    # before the class binding (selection)
        # a quick second click arrives as a double-click: same handler, so ✕ still removes and the column edge stays shut
        tree.bind("<Double-Button-1>", lambda e, f=f: self._click_remove(f, e))
        tree.bind("<Motion>", lambda e, f=f: self._tree_motion(f, e))
        tree.bind("<Leave>", lambda e: tree.configure(cursor=""))
        if TkinterDnD is not None:
            tree.drop_target_register(DND_FILES)
            tree.dnd_bind("<<Drop>>", lambda e, f=f: self.add(f, self.root.tk.splitlist(e.data)))
        right = ttk.Frame(paned, padding=(5, 0, 0, 0))
        self._preview_pane(f, right, pv_titles).pack(fill="both", expand=True)
        paned.add(right, weight=1)
        # the divider is dragged by our own handlers: it stops where the buttons under the list touch (user 2026-11-11)
        f.sash_drag = None
        paned.bind("<ButtonPress-1>", lambda e, f=f: self._sash_press(f, e))
        paned.bind("<B1-Motion>", lambda e, f=f: self._sash_motion(f, e))
        paned.bind("<ButtonRelease-1>", lambda e, f=f: self._sash_release(f, e))
        paned.bind("<Configure>", lambda e, f=f: self.root.after_idle(lambda: self._sash_set(f, paned.sashpos(0))),
                   add="+")
        row = ttk.Frame(left)
        f.btn_row = row
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Add files…", command=lambda: self.add(f, filedialog.askopenfilenames())).pack(side="left")
        ttk.Button(row, text="Add folder…", command=lambda: self.add(f, [filedialog.askdirectory()])).pack(side="left", padx=4)
        ttk.Button(row, text="Clear", command=lambda: self.clear(f)).pack(side="left")
        f.btn_cancel = ttk.Button(row, text="Cancel", command=self.cancel, state="disabled")
        f.btn_cancel.pack(side="right")
        f.btn_start = ttk.Button(row, text="Start", command=lambda: self.start(f), style="Accent.TButton")
        f.btn_start.pack(side="right", padx=4)
        f.opts = ttk.Frame(left)
        f.opts.pack(fill="x", pady=(8, 0))
        return f

    # -- preview: the selected row's input + its output once one exists (this run, or an earlier one on disk)
    def _preview_pane(self, f, parent, titles):
        pv = ttk.Notebook(parent, style="Preview.TNotebook")
        f.pv, f.pv_shown, f.pv_imgs = pv, None, [None, None]
        f.pv_labels = []
        f.pv_files = [None, None]                       # the file each preview shows = the file a drag carries out
        for k, t in enumerate(titles):
            box = tk.Frame(pv, bg=SURFACE, width=PREVIEW_BOX[0], height=PREVIEW_BOX[1])
            box.pack_propagate(False)
            lab = tk.Label(box, bg=SURFACE, bd=0)
            lab.pack(fill="both", expand=True)
            box.bind("<Configure>", lambda e, f=f: self._pv_fit_later(f))
            # the INPUT preview takes drops (photos in, like the list); the OUTPUT preview is dragged out (its file into
            # Blender, a folder, …). User 2026-10-31: the estimated map is what leaves, the photograph is what comes in.
            if TkinterDnD is not None and k == 0:
                for w in (box, lab):
                    w.drop_target_register(DND_FILES)
                    w.dnd_bind("<<Drop>>", lambda e, f=f: self._dropped_on_preview(f, e.data))
            elif TkinterDnD is not None:
                lab.drag_source_register(1, DND_FILES)
                lab.dnd_bind("<<DragInitCmd>>", lambda e, f=f: self._drag_out(f))
            pv.add(box, text=t)
            f.pv_labels.append(lab)
        pv.tab(1, state="disabled")
        return pv

    def _drag_out(self, f):
        p = f.pv_files[1]
        if not p or not os.path.isfile(p):
            return REFUSE_DROP
        return (COPY, DND_FILES, (os.path.normpath(p),))

    def _dropped_on_preview(self, f, data):
        """Dropped on the input preview = added to the list, and the first dropped picture is shown."""
        self.add(f, self.root.tk.splitlist(data), show=True)
        return COPY

    def _selected(self, f):
        sel = f.tree.selection()
        self.show_preview(f, sel[0] if sel else None)

    def _output_of(self, f, iid, for_drag=False):
        """The output of row `iid`: written by this session's run, else an earlier one found where it would be written.
        Shown: the PNG when there are both (decodes faster). Dragged: the file type chosen under Output."""
        files = [p for p in f.outputs.get(iid, []) if os.path.isfile(p)]
        if not files:
            files = self._outputs_on_disk(f, iid)
        if not files:
            return None
        want = ".exr" if (for_drag and self.fmt.get().startswith("exr")) else ".png"
        return sorted(files, key=lambda p: not p.lower().endswith(want))[0]

    def _outputs_on_disk(self, f, iid):
        src = f.paths[iid]
        stem, ext = os.path.splitext(os.path.basename(src))
        # the HIGHEST version on disk (0.1.2: every run adds _vNNN; an unversioned 0.1.x file = version 0)
        d = self._out_for(f) or os.path.dirname(src)
        if f.kind == "depth":
            suffix = "_depth_see_through" if self.model.get() == "see-through" else "_depth"
            found = outnames.latest_outputs(d, f"{stem}{suffix}")
        else:
            n = int(self.level.get())                    # + a relief from before the 2026-10-31 rename (= version 0)
            found = outnames.latest_outputs(d, f"{stem}_relief_strength{n}", aliases=(f"{stem}_relief_level{n}",))
            found = [p for p in found if p.lower().endswith(ext.lower())]
        return [p for p in found if os.path.splitext(p)[1].lower() in (".png", ".exr")]

    def show_preview(self, f, iid, to_output=False):
        f.pv_shown = iid
        out = self._output_of(f, iid) if iid else None
        f.pv_files = [f.paths[iid] if iid else None, self._output_of(f, iid, for_drag=True) if iid else None]
        f.pv_labels[1].configure(cursor="hand2" if f.pv_files[1] else "")
        f.pv.tab(1, state="normal" if out else "disabled")
        if to_output and out:
            f.pv.select(1)
        elif not out and f.pv.index("current") == 1:
            f.pv.select(0)
        for k, path in enumerate((f.paths[iid] if iid else None, out)):
            self._pv_load(f, k, path)

    def _pv_load(self, f, k, path):
        """Decode in a thread (a 60 MP photo takes a second or two); a result lands only if its row is still shown."""
        f.pv_imgs[k] = None
        f.pv_labels[k].configure(image="")
        if not path:
            return
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return
        if key in self.pv_cache:
            f.pv_imgs[k] = self.pv_cache[key]
            self._pv_fit(f)
            return

        def run():
            try:
                im = preview_image(path)
            except Exception:
                im = None

            def done():
                if im is None:
                    return
                if len(self.pv_cache) > 24:
                    self.pv_cache.pop(next(iter(self.pv_cache)))
                self.pv_cache[key] = im
                if f.pv_shown is not None and path in (f.paths.get(f.pv_shown), self._output_of(f, f.pv_shown)):
                    f.pv_imgs[k] = im
                    self._pv_fit(f)
            self.ui.put(done)
        threading.Thread(target=run, daemon=True).start()

    def _pv_fit_later(self, f):
        if getattr(f, "pv_after", None):
            self.root.after_cancel(f.pv_after)
        f.pv_after = self.root.after(80, lambda: self._pv_fit(f))

    def _pv_fit(self, f):
        """Each loaded picture scaled to fit its box, aspect kept."""
        f.pv_after = None
        f.pv_tk = []                                     # Tk drops an image nobody holds
        for im, lab in zip(f.pv_imgs, f.pv_labels):
            if im is None:
                lab.configure(image="")
                continue
            bw, bh = max(lab.master.winfo_width(), 50), max(lab.master.winfo_height(), 50)
            s = min(bw / im.width, bh / im.height)
            fit = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            tkim = ImageTk.PhotoImage(fit)
            f.pv_tk.append(tkim)
            lab.configure(image=tkim)

    def _depth_options(self, f):
        o = f.opts
        # default = see-through (it can keep background detail the base model loses); a choice already saved in
        # settings.json is kept
        self.model = tk.StringVar(value=self.s.get("model", "see-through"))
        # earlier settings held png / exr / both (16-bit PNG / 32-bit EXR / both) -> the same file in the new list
        fmt = {"png": "png16", "exr": "exr32", "both": "exr32"}.get(self.s.get("format"), self.s.get("format"))
        self.fmt = tk.StringVar(value=fmt if fmt in dict(OUTPUTS) else OUTPUTS[0][0])
        ttk.Label(o, text="Model").grid(row=0, column=0, sticky="w")
        self._combo(o, self.model, (("depth", "Base Depth Estimation"),
                                    ("see-through", "See-through Depth Estimation"))).grid(row=0, column=1, sticky="w")
        ttk.Label(o, text="Output").grid(row=1, column=0, sticky="w")
        self._combo(o, self.fmt, OUTPUTS).grid(row=1, column=1, sticky="w")
        self._save_to(f, o, 2, "out_dir")
        for w in o.grid_slaves():
            w.grid_configure(padx=(0, 16), pady=3)

    def _relief_options(self, f):
        o = f.opts
        self.level = tk.IntVar(value=self.s.get("level", 3))
        ttk.Label(o, text="Strength").grid(row=0, column=0, sticky="w")
        self._combo(o, self.level, [(n, str(n)) for n in range(1, 7)], width=6).grid(row=0, column=1, sticky="w")
        self._save_to(f, o, 1, "relief_out_dir")
        for w in o.grid_slaves():
            w.grid_configure(padx=(0, 16), pady=3)

    def _save_to(self, f, o, row, key):
        """The Save to dropdown of one tab; each tab keeps its own choice and its own remembered folder (settings `key`)."""
        f.where = tk.StringVar(value="beside")
        f.out_dir = self.s.get(key, "")
        f.out_key = key
        f.beside = BESIDE_FILE if f.kind == "relief" else BESIDE
        ttk.Label(o, text="Save to").grid(row=row, column=0, sticky="w")
        f.cb_where = ttk.Combobox(o, state="readonly", width=COMBO_WIDTH)
        f.cb_where.bind("<<ComboboxSelected>>", lambda e: self._where_picked(f))
        f.cb_where.grid(row=row, column=1, sticky="w")
        f.where.trace_add("write", lambda *_: self._show_where(f))
        self._show_where(f)

    @staticmethod
    def _out_for(f):
        """The chosen folder of tab `f`, or None = next to each file."""
        return f.out_dir if (f.where.get() == "chosen" and f.out_dir) else None

    def _combo(self, parent, var, choices, width=None):
        """A read-only dropdown over (value, label) pairs; `var` keeps holding the VALUE (what start() reads)."""
        shown = tk.StringVar()
        cb = ttk.Combobox(parent, textvariable=shown, values=[t for _, t in choices], state="readonly",
                          width=width or COMBO_WIDTH)

        def to_shown(*_):
            v = var.get()
            shown.set(next((t for val, t in choices if val == v), choices[0][1]))
        var.trace_add("write", to_shown)
        cb.bind("<<ComboboxSelected>>", lambda e: (var.set(choices[cb.current()][0]), cb.selection_clear()))
        to_shown()
        return cb

    # -- lists
    def add(self, f, paths, show=False):
        """show=True (a drop on the preview): select the first dropped picture even when it was already listed."""
        if self.job:
            return
        paths = [p for p in (paths or ()) if p]
        first = None
        for p in cli.photos(paths, kind="relief" if f.kind == "relief" else "depth"):
            iid = next((i for i, q in f.paths.items() if q == p), None)
            if iid is None:
                iid = f.tree.insert("", "end", text=p, values=("", REMOVE_MARK))
                f.paths[iid] = p
            elif not show:
                continue
            first = first or iid
        self._layout(f)
        if first and (show or not f.tree.selection()):  # something to look at straight away
            f.tree.selection_set(first)
            f.tree.see(first)
            f.pv.select(0)

    # the list's columns: the status column only as wide as its longest text (nothing before a run), the name column
    # takes the rest up to the ✕, and a path too long for it is cut on the LEFT so the file name always shows
    def _layout(self, f):
        tree = f.tree
        total = tree.winfo_width()
        if total <= 1:
            return
        if not hasattr(self, "_list_font"):
            spec = ttk.Style(self.root).lookup("Treeview", "font") or "TkDefaultFont"
            self._list_font = tkfont.Font(self.root, font=spec)
        fnt = self._list_font
        rows = tree.get_children()
        need = max([fnt.measure(tree.set(i, "status")) for i in rows] + [0])
        if self.job and self.job.get("tab") is f:       # during a run: room for the longest progress text at once,
            need = max(need, fnt.measure("depth pass 12/12 · 100 % · about 59 min left · 59:59"))   # so the names do not jump at every step
        sw = need + 16 if need else 4
        rw = tree.column("remove", "width")
        nw = max(60, total - sw - rw - 4)
        if tree.column("status", "width") != sw:
            tree.column("status", width=sw)
        if tree.column("#0", "width") != nw:
            tree.column("#0", width=nw)
        room = nw - 14                                  # the cell's own padding
        for i in rows:
            txt = fit_left(f.paths[i], room, fnt)
            if tree.item(i, "text") != txt:
                tree.item(i, text=txt)

    # one row out: the ✕ at the right end of the row (hidden while a batch runs: the worker works from the list)
    def _remove_cell(self, f, e):
        """-> the row id when (e.x, e.y) is on a row's ✕, else None."""
        if self.job or f.tree.identify_region(e.x, e.y) != "cell":
            return None
        if f.tree.identify_column(e.x) != "#2":         # #1 = status, #2 = remove
            return None
        return f.tree.identify_row(e.y) or None

    # -- the divider between the list and the preview
    def _sash_set(self, f, pos):
        """Place the divider, never left of where the buttons under the list touch (they keep their size)."""
        # and never so far right that the preview's two tab titles are cut: past that ttk hides the preview pane
        # entirely and a wider window then goes to the list only (found 2026-11-11)
        tabs = self._pv_tabs_width(f)
        if tabs:
            pos = min(int(pos), f.paned.winfo_width() - 6 - 5 - tabs)   # 6 = divider, 5 = preview side's padding
        pos = max(f.btn_row.winfo_reqwidth() + 5, int(pos))   # + 5 = the list side's padding at the divider
        if f.paned.sashpos(0) != pos:
            f.paned.sashpos(0, pos)

    def _pv_tabs_width(self, f=None):
        """Room for the preview's tab titles = the WIDER of the two pages', so one divider position fits both
        (Depth 188 px, Relief 256 px at 100 %; = where the last title ends as read off the drawn notebook)."""
        if not hasattr(self, "tab_relief"):              # still building the pages
            return None
        if getattr(self, "_pv_tabs_w", None) is None:
            st = ttk.Style(self.root)
            fnt = tkfont.Font(self.root, font=st.lookup("Preview.TNotebook.Tab", "font") or "TkDefaultFont")
            pad = st.lookup("Preview.TNotebook.Tab", "padding")
            padx = self.root.winfo_pixels(str(pad[0])) if pad else 12
            self._pv_tabs_w = max(sum(fnt.measure(t.pv.tab(i, "text")) + 2 * padx for i in range(t.pv.index("end"))) + 6
                                  for t in (self.tab_depth, self.tab_relief))
        return self._pv_tabs_w

    def _sash_press(self, f, e):
        if str(f.paned.identify(e.x, e.y)) == "":
            return None
        f.sash_drag = e.x - f.paned.sashpos(0)
        return "break"

    def _sash_motion(self, f, e):
        if f.sash_drag is None:
            return None
        self._sash_set(f, e.x - f.sash_drag)
        return "break"

    def _sash_release(self, f, e):
        if f.sash_drag is None:
            return None
        f.sash_drag = None
        return "break"

    def _tab_changed(self, e=None):
        """The page shown takes the divider position of the page left (each page keeps its own pane otherwise)."""
        new = self.nb.nametowidget(self.nb.select())
        old, self._tab_now = self._tab_now, new
        if old is new or not hasattr(new, "paned"):
            return
        pos = old.paned.sashpos(0)
        self.root.update_idletasks()
        self._sash_set(new, pos)

    def _tree_motion(self, f, e):
        # the column widths are the app's (_layout): no resize cursor on the heading's column edges (user 2026-11-11)
        if f.tree.identify_region(e.x, e.y) == "separator":
            f.tree.configure(cursor="")
            return "break"
        f.tree.configure(cursor="hand2" if self._remove_cell(f, e) else "")
        return None

    def _click_remove(self, f, e):
        if f.tree.identify_region(e.x, e.y) == "separator":
            return "break"                              # no column drag: the divider beside the preview resizes
        iid = self._remove_cell(f, e)
        if not iid:
            return None
        self.remove(f, iid)
        return "break"                                  # the click removes; it does not also select

    def remove(self, f, iid):
        if self.job:
            return
        rows = list(f.tree.get_children())
        was_shown = f.pv_shown == iid
        f.tree.delete(iid)
        f.paths.pop(iid, None)
        f.outputs.pop(iid, None)
        self._layout(f)
        if was_shown:                                   # show the next row (or the one before the last), or nothing
            rest = [r for r in rows if r != iid]
            nxt = rows[rows.index(iid) + 1] if rows.index(iid) + 1 < len(rows) else (rest[-1] if rest else None)
            if nxt:
                f.tree.selection_set(nxt)
            else:
                self.show_preview(f, None)

    def clear(self, f):
        if self.job:
            return
        f.tree.delete(*f.tree.get_children())
        f.paths.clear()
        f.outputs.clear()
        self._layout(f)
        self.show_preview(f, None)

    # Save to: "Next to each photo" · the chosen folder (once there is one) · "Choose a folder…" (opens the dialog)
    def _show_where(self, f):
        f.cb_where.configure(values=[f.beside] + ([f.out_dir] if f.out_dir else []) + [CHOOSE])
        f.cb_where.set(f.out_dir if f.where.get() == "chosen" and f.out_dir else f.beside)

    def _where_picked(self, f):
        f.cb_where.selection_clear()
        pick = f.cb_where.get()
        if pick == CHOOSE:
            self.pick_out(f)
        else:
            f.where.set("beside" if pick == f.beside else "chosen")

    def pick_out(self, f):
        d = filedialog.askdirectory(initialdir=f.out_dir or None)
        if d:
            f.out_dir = os.path.normpath(d)
            f.where.set("chosen")
        self._show_where(f)                             # cancelled: back to what was shown before

    # -- models
    def models_ok(self, model=None):
        try:
            for m in ([model] if model else ["depth", "see-through"]):
                models.locate(self.models_dir, m)
            return True
        except FileNotFoundError:
            return False

    def check_models(self):
        self.check_third_party()
        have = [m for m in ("depth", "see-through") if self.models_ok(m)]
        if len(have) == 2:
            self.banner.pack_forget()
            return
        need = download.missing(self.models_dir, ["depth", "see-through"])
        gb = sum(e["size"] for e in need) / 1e9
        what = "Models" if not have else ("See-through model" if "depth" in have else "Depth model")
        self.banner_text.set(f"{what} not found in {self._models_place()} (Total required space {gb:.1f} GB)")
        self.btn_dl.pack(side="right")
        self.btn_have.pack(side="right", padx=4)
        self.banner.pack(fill="x", before=self.nb)

    # -- PyTorch + OpenCV
    def tp_missing(self):
        return third_party.missing()

    def check_third_party(self):
        """The PyTorch + OpenCV row: hidden when both are installed (or importable: the development environment)."""
        if self.tp_job:
            return
        miss = self.tp_missing()
        if not miss:
            self.tp_banner.pack_forget()
            return
        if self.gpu_note is None:                       # nvidia-smi, up to 15 s: off the Tk thread, once
            self.gpu_note = ""
            threading.Thread(target=lambda: self.ui.put(lambda g=self._gpu_sentence(): self._gpu_checked(g)),
                             daemon=True).start()
        self.tp_text.set(self._tp_sentence(miss))
        self.btn_tp.configure(text="Download", command=self.start_third_party, state="normal")
        self.tp_banner.pack(fill="x", before=self.nb)

    def _tp_sentence(self, miss):
        gb = sum(e["size"] for e in miss) / 1e9
        short = lambda v: ".".join(v.split("+")[0].split(".")[:2])          # 2.10.0+cu128 -> 2.10
        names = " and ".join(f"{e['name']} {short(e['version'])} (from {e['publisher']})" for e in miss)
        s = (f"The app also needs {names}, {gb:.1f} GB to download. They are not included in Vathography App and "
             f"are used under their own licences.")
        if self.gpu_note:
            s += " " + self.gpu_note
        return s

    def _gpu_sentence(self):
        """'' = this computer has a card the app can use; else why not (before 2.9 GB are downloaded for nothing)."""
        g = gpu_info()
        if g is None:
            return ("This computer has no NVIDIA graphics card the app can use, or its driver is missing, so the app "
                    "cannot run here.")
        if g["cc"] < MIN_CC:
            return f"{g['name']} is too old for the app, which needs an NVIDIA RTX 30-series card or newer."
        return ""

    def _gpu_checked(self, note):
        self.gpu_note = note
        miss = self.tp_missing()
        if miss and not self.tp_job:
            self.tp_text.set(self._tp_sentence(miss))

    def _tp_writable(self):
        """None when third_party\\ can be written, else a sentence (a read-only place, e.g. Program Files)."""
        try:
            os.makedirs(third_party.HOME, exist_ok=True)
            probe = os.path.join(third_party.HOME, f".write_test_{os.getpid()}")
            with open(probe, "w", encoding="ascii") as fh:
                fh.write("x")
            os.remove(probe)
            return None
        except OSError:
            return (f"The app cannot write into {third_party.HOME}. Move the Vathography App folder somewhere you can "
                    f"write to, such as your Documents folder, and start it from there.")

    def _tp_space(self, miss):
        """None when the drive of third_party\\ has room for the wheels + their unpacked files (+ the models still to
        download, when they go to the same drive), else a sentence."""
        unpacked = sum(e["unpacked_size"] for e in miss)
        same = (os.path.splitdrive(os.path.abspath(self.models_dir))[0].lower()
                == os.path.splitdrive(os.path.abspath(third_party.HOME))[0].lower())
        models_left = download.bytes_needed(self.models_dir, download.missing(self.models_dir, ["depth", "see-through"])) if same else 0
        return download.check_space(third_party.HOME, miss, extra=unpacked + models_left,
                                    what="PyTorch, OpenCV and the models" if models_left else "PyTorch and OpenCV")

    def start_third_party(self):
        miss = self.tp_missing()
        if not miss or self.tp_job:
            return
        msg = self._tp_writable() or self._tp_space(miss)
        if msg:
            self.status.set(msg)
            return
        self.status.set("")
        cancel = threading.Event()
        self.tp_job = dict(cancel=cancel)
        self.btn_tp.configure(text="Stop", command=cancel.set)
        self.tp_bar.pack(side="right", padx=6)
        self.tp_bar.configure(value=0)

        def say(text, frac):
            self.ui.put(lambda: (self.tp_text.set(text), self.tp_bar.configure(value=100 * frac)))

        def run():
            err = None
            try:
                for e in miss:
                    label = f"{e['name']} {e['version'].split('+')[0]}"
                    t0 = [0.0]

                    def got(have, size, label=label):
                        if time.time() - t0[0] >= 0.2 or have == size:
                            t0[0] = time.time()
                            say(f"Downloading {label} · {have / 1e9:.2f} of {size / 1e9:.2f} GB", have / size)
                    wheel = download.fetch(e, third_party.HOME, got, cancel)   # ours: deleted once installed

                    def step(phase, done, total, label=label):
                        if time.time() - t0[0] >= 0.2 or done == total:
                            t0[0] = time.time()
                            word = "Checking" if phase == "check" else "Unpacking"
                            say(f"{word} {label} · {done / 1e9:.2f} of {total / 1e9:.2f} GB", done / max(total, 1))
                    third_party.install(e, wheel, step, cancel, delete_wheel=True)
            except (download.Cancelled, third_party.Cancelled):
                err = "The download was stopped; Download resumes it."
            except third_party.Busy as ex:
                err = str(ex)
            except Exception as ex:                     # noqa: BLE001 -- shown as one sentence
                err = f"PyTorch and OpenCV could not be installed: {ex}"
            self.ui.put(lambda: self._third_party_end(err))
        threading.Thread(target=run, daemon=True).start()

    def _third_party_end(self, err):
        self.tp_job = None
        self.tp_bar.pack_forget()
        self.status.set(err or "")
        self.check_third_party()

    def _models_place(self):
        """'the default folder' (beside the app), else the folder chosen with "I already have them…"."""
        same = os.path.normcase(os.path.abspath(self.models_dir)) == os.path.normcase(os.path.abspath(DEFAULT_MODELS))
        return "the default folder" if same else self.models_dir

    # -- the gear menu
    def open_menu(self):
        g = self.gear
        self.menu.tk_popup(g.winfo_rootx() + g.winfo_width(), g.winfo_rooty() + g.winfo_height())

    def change_models(self):
        """Any time (the banner's "I already have them…" exists only while models are missing)."""
        if self.job:
            return
        self.pick_models()

    def _dialog(self, title, w, h):
        top = tk.Toplevel(self.root, bg=BG)
        top.title(title)
        top.transient(self.root)
        top.geometry(f"{w}x{h}+{self.root.winfo_rootx() + 40}+{self.root.winfo_rooty() + 40}")
        dark_title_bar(top)
        return top

    def _text_box(self, parent, text):
        box = tk.Frame(parent, bg=BG)
        box.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        t = tk.Text(box, bg=SURFACE, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, wrap="none",
                    font=("Consolas", 9), padx=8, pady=6, selectbackground=ACCENT_SOFT, selectforeground=TEXT)
        sy = ttk.Scrollbar(box, orient="vertical", command=t.yview)
        sx = ttk.Scrollbar(box, orient="horizontal", command=t.xview)
        t.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        t.pack(side="left", fill="both", expand=True)
        t.insert("1.0", text)
        t.configure(state="disabled")                    # read-only; selecting + Ctrl+C still works
        return t

    def troubleshooting_text(self):
        """What we need to help: the app, the machine, the settings, then the last run's log, exactly as written."""
        g = gpu_info()
        try:
            r = subprocess.run([shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe",
                                "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                               timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            driver = r.stdout.strip().splitlines()[0]
        except Exception:
            driver = "?"
        try:
            ram = f"{ram_gb():.0f} GB"
        except Exception:
            ram = "?"
        lines = [f"{APP_NAME} {__version__}",
                 f"Windows: {platform.platform()}",
                 (f"Graphics card: {g['name']} | {g['total_gib']:.1f} GB, {g['free_gib']:.1f} GB free | "
                  f"compute {g['cc']} | driver {driver}") if g else "Graphics card: no NVIDIA card found",
                 f"Memory: {ram}",
                 f"Models folder: {self.models_dir} (found: "
                 f"{', '.join(m for m in ('depth', 'see-through') if self.models_ok(m)) or 'none'})",
                 f"Settings: {json.dumps({k: v for k, v in self.s.items() if k != 'models_dir'})}",
                 f"Message shown: {self.status.get() or '-'}",
                 "", f"--- last run ({RUN_LOG}) ---"]
        try:
            with open(RUN_LOG, encoding="utf-8", errors="replace") as fh:
                lines.append(fh.read().rstrip() or "(empty)")
        except OSError:
            lines.append("(no run yet)")
        return "\n".join(lines) + "\n"

    def show_log(self):
        top = self._dialog("Troubleshooting log", 860, 540)
        text = self.troubleshooting_text()
        self._text_box(top, text)
        # issues are public and the log holds the user's paths: the line says so before they paste it
        link = tk.Label(top, text=f"To report a problem, paste this log in a new issue at {ISSUES_URL.removeprefix('https://')}. "
                                  "Issues are public and the log shows your folder and file names.",
                        bg=BG, fg=TEXT_SOFT, font=(SANS, 9), cursor="hand2", anchor="w", justify="left", wraplength=836)
        link.pack(fill="x", padx=12, pady=(0, 6))
        link.bind("<Button-1>", lambda e: webbrowser.open(ISSUES_URL))
        top.issue_link = link
        row = tk.Frame(top, bg=BG)
        row.pack(fill="x", padx=12, pady=(0, 12))
        note = tk.StringVar()
        ttk.Label(row, textvariable=note, style="Status.TLabel").pack(side="left")

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()                           # hand it to Windows now: it stays after the app closes
            note.set("Copied. Paste it into your message.")

        def save():
            p = filedialog.asksaveasfilename(parent=top, defaultextension=".txt", initialfile="vathography_app_log.txt",
                                             filetypes=[("Text", "*.txt")])
            if p:
                with open(p + ".tmp", "w", encoding="utf-8") as fh:
                    fh.write(text)
                os.replace(p + ".tmp", p)
                note.set(f"Saved: {p}")

        ttk.Button(row, text="Close", command=top.destroy).pack(side="right")
        ttk.Button(row, text="Save as…", command=save).pack(side="right", padx=4)
        ttk.Button(row, text="Copy all", command=copy, style="Accent.TButton").pack(side="right")
        top.copy, top.save, top.note = copy, save, note  # (the window check calls them)
        return top

    def show_about(self):
        top = self._dialog("About Vathography App", 720, 520)
        head = tk.Frame(top, bg=BG)
        head.pack(fill="x", pady=(16, 4))
        tk.Label(head, text=APP_NAME, bg=BG, fg=TEXT, font=(SERIF, 20)).pack()
        tk.Label(head, text=f"Version {__version__}", bg=BG, fg=TEXT_SOFT, font=(SANS, 10)).pack()
        link = tk.Label(head, text="vathography.com", bg=BG, fg=ACCENT, font=(SANS, 10, "underline"), cursor="hand2")
        link.pack(pady=(2, 0))
        link.bind("<Button-1>", lambda e: webbrowser.open("https://vathography.com"))
        try:
            with open(NOTICE, encoding="utf-8") as fh:
                notice = fh.read()
        except OSError:
            notice = "(NOTICE.txt not found)"
        self._text_box(top, notice).configure(wrap="word")
        ttk.Button(top, text="Close", command=top.destroy).pack(side="right", padx=12, pady=(0, 12))
        return top

    def pick_models(self):
        if self.check:                                  # a check is already running
            return
        d = filedialog.askdirectory(title="The folder that holds the model files", initialdir=self.models_dir or None)
        if not d:
            return
        # the check is shown IN the banner, in place of its text, with a clock (also when the banner was hidden: the
        # gear menu's "Models folder…" with the models present); the line under the list keeps errors only
        self.status.set("")
        total = json.load(open(models.MANIFEST, encoding="utf-8"))["total_download_bytes"]   # what a full check reads
        self.check = dict(t0=time.time(), done=0, total=total, last=0.0)
        self.btn_have.configure(state="disabled")
        self.btn_dl.configure(state="disabled")
        self.banner.pack(fill="x", before=self.nb)
        self._render_check()

        def run():
            found, cache = [], {}                       # one checksum per file: both models share the backbone

            def got(n):
                c = self.check
                c["done"] += n
                if time.time() - c["last"] >= 0.2:      # at most 5 updates a second
                    c["last"] = time.time()
                    self.ui.put(self._render_check)
            for m in ("depth", "see-through"):
                try:
                    models.locate(d, m, verify=True, progress=got, cache=cache)
                    found.append(m)
                except FileNotFoundError:
                    pass
            self.ui.put(lambda: self._picked(d, found))
        threading.Thread(target=run, daemon=True).start()

    def _render_check(self):
        """The banner during the check: 'Checking the model files · 6.2 of 15.5 GB · 0:12' (the clock every second)."""
        c = self.check
        if not c:
            return
        sec = int(time.time() - c["t0"])
        read = c["done"] / 1e9
        amount = f"{read:.1f} of {c['total'] / 1e9:.1f} GB" if c["done"] <= c["total"] else f"{read:.1f} GB read"
        self.banner_text.set(f"Checking the model files · {amount} · {sec // 60}:{sec % 60:02d}")

    def _picked(self, d, found):
        self.check = None
        self.btn_have.configure(state="normal")
        self.btn_dl.configure(state="normal")
        if not found:
            self.check_models()                         # the banner as it was (or hidden again)
            self.status.set(f"The model files were not found in {d} (or their checksums differ).")
            return
        self.models_dir = d
        self.s["models_dir"] = d
        save_settings(self.s)
        self.status.set("")
        self.check_models()

    def start_download(self):
        need = download.missing(self.models_dir, ["depth", "see-through"])
        offline = [e for e in need if not download.hosted(e)]
        if offline:
            self.status.set("Some model files are not online yet: " +
                            ", ".join(os.path.basename(e["path"]) for e in offline) + ".")
            need = [e for e in need if download.hosted(e)]
            if not need:
                return
        tp = self.tp_missing()                          # PyTorch + OpenCV still to come onto the same drive
        same = (os.path.splitdrive(os.path.abspath(self.models_dir))[0].lower()
                == os.path.splitdrive(os.path.abspath(third_party.HOME))[0].lower())
        tp_left = (download.bytes_needed(third_party.HOME, tp) + sum(e["unpacked_size"] for e in tp)) if (tp and same) else 0
        msg = download.check_space(self.models_dir, need, extra=tp_left,
                                   what="the models, PyTorch and OpenCV" if tp_left else "the models")
        if msg:
            self.status.set(msg)
            return
        self.dl_cancel = threading.Event()
        self.btn_dl.configure(text="Stop", command=self.dl_cancel.set)
        self.btn_have.pack_forget()
        self.dl_bar.pack(side="right", padx=6)
        total = sum(e["size"] for e in need)
        done_before = [0]

        def prog(have, size):
            self.ui.put(lambda: self.dl_bar.configure(value=100 * (done_before[0] + have) / total))

        def run():
            err = None
            try:
                for e in need:
                    self.ui.put(lambda e=e: self.banner_text.set(f"Downloading {os.path.basename(e['path'])} "
                                                                        f"({e['size'] / 1e9:.2f} GB)…"))
                    download.fetch(e, self.models_dir, prog, self.dl_cancel)
                    done_before[0] += e["size"]
            except download.Cancelled:
                err = "The download was stopped; Download resumes it."
            except Exception as ex:
                err = f"The download failed: {ex}"
            self.ui.put(lambda: self._download_end(err))
        threading.Thread(target=run, daemon=True).start()

    def _download_end(self, err):
        self.dl_bar.pack_forget()
        self.btn_dl.configure(text="Download", command=self.start_download)
        self.status.set(err or "")
        self.check_models()

    # -- jobs
    def start(self, f):
        if self.job:
            return
        items = list(f.tree.get_children())
        if not items:
            return
        if self.tp_missing():                           # both tabs: the work needs PyTorch + OpenCV
            self.status.set("PyTorch and OpenCV are not installed yet. Click Download at the top first.")
            return
        if f.kind == "depth" and not self.models_ok(self.model.get()):
            self.status.set(f"Models not found in {self._models_place()}.")
            return
        # the memory check (nvidia-smi, up to 15 s) runs off the Tk thread; it is shown on the first photo's row, whose
        # clock runs from the click (the first photo waits for the check and the loading)
        now = time.time()
        self.job = dict(tab=f, items=items, out=None, current=None, t0=now, row=1, row_t0=now,
                        head="checking the graphics card", own="", before={i: f.tree.set(i, "status") for i in items})
        self.status.set("")
        for i in items:                                 # a re-run of finished rows: they wait again (restored if refused)
            f.tree.set(i, "status", "waiting")
        f.btn_start.configure(state="disabled")
        f.btn_cancel.configure(state="normal")
        self._render()

        def check():
            r = preflight()
            self.ui.put(lambda: self._start_worker(f, items, *r))
        threading.Thread(target=check, daemon=True).start()

    def _start_worker(self, f, items, budget, notes):
        j = self.job
        if j is None or j.get("cancelled") or budget is None:
            if budget is None:
                self.status.set(notes[0])
            if j:                                       # refused before any work: the rows as they were
                for i, st in j["before"].items():
                    if f.tree.exists(i):
                        f.tree.set(i, "status", st)
            self._end_job(f)
            return
        files = [f.paths[i] for i in items]
        out = self._out_for(f)
        if f.kind == "depth":
            args = ["depth"] + files + ["--format", self.fmt.get(), "--models", self.models_dir, "--budget", str(budget)]
            if self.model.get() == "see-through":
                args.append("--see-through")
            self.s.update(model=self.model.get(), format=self.fmt.get())
        else:
            args = ["relief"] + files + ["--strength", str(self.level.get()), "--budget", str(budget)]
            self.s.update(level=self.level.get())
        if out:
            args += ["--out", out]
            self.s[f.out_key] = out
        save_settings(self.s)
        for i in items:
            f.tree.set(i, "status", "waiting")
            f.tree.set(i, "remove", "")
        j.update(out=out, head="starting the depth engine", own="")
        self._render()
        self._layout(f)
        self.status.set(" ".join(notes))
        try:
            self.worker = Worker(args)
        except OSError as ex:                           # last_run.log not writable, or the worker cannot start: idle again
            for i, st in j["before"].items():
                if f.tree.exists(i):
                    f.tree.set(i, "status", st)
                    f.tree.set(i, "remove", REMOVE_MARK)
            self.status.set(f"The work could not start: {ex}")
            self._end_job(f)

    def _end_job(self, f):
        """Back to idle: the buttons; the clock stops (it only writes while there is a job)."""
        f.btn_start.configure(state="normal")
        f.btn_cancel.configure(state="disabled")
        self.job = None

    def _clock(self):
        """Every second: the current photo's row is written again, so its clock moves even when the work sends
        nothing (the clock is the sign of life)."""
        if self.job:
            self._render()
        if self.check:
            self._render_check()
        self.root.after(1000, self._clock)

    def _render(self):
        """ONE place for the progress: the current photo's row = '<stage> · <%> · about N min left · m:ss'.
        % = the stage's own count, or the photo's % in the depth passes; the time left only in the passes; m:ss =
        this photo's clock (the first photo's includes the memory check and the loading)."""
        j = self.job
        if not j or not j.get("row") or j.get("row") > len(j["items"]):
            return
        f = j["tab"]
        iid = j["items"][j["row"] - 1]
        if not f.tree.exists(iid):
            return
        st = f.tree.set(iid, "status")
        if st.startswith("done") or st in ("failed", "cancelled", "not done"):
            return
        parts = [j.get("head") or "working…"]
        if j.get("passes") and j.get("pct"):
            parts.append(j["pct"])
            if j.get("eta"):
                parts.append(j["eta"])
        elif j.get("own"):
            parts.append(j["own"])
        sec = int(time.time() - j.get("row_t0", j["t0"]))
        h, m = divmod(sec, 3600)
        parts.append(f"{h}:{m // 60:02d}:{m % 60:02d}" if h else f"{m // 60}:{m % 60:02d}")
        f.tree.set(iid, "status", " · ".join(parts))

    def _stage(self, j, f, ev):
        """A stage event -> the head of the current row (photo 0 = loading: the first photo's row). The fast detail
        ('block 37/60 · tile 9/23', 'sample 7 of 16') is not shown: the head and its % only."""
        name = ev["name"]
        j["head"] = _stage_row(name)
        j["passes"] = name.startswith(_PASS_STAGES)
        j["own"] = f"{100 * ev['done'] // ev['total']} %" if ev.get("total") else ""
        self._render()

    def cancel(self):
        if not self.job:
            return
        self.job["cancelled"] = True
        if self.worker:                                 # during the memory check there is no worker yet
            self.worker.cancel()

    def _remove_tmp(self, path):
        """A worker ended mid-write leaves <output>.tmp (photo_io.write_depth: temp + os.replace)."""
        stem = os.path.splitext(os.path.basename(path))[0]
        d = self.job["out"] or os.path.dirname(os.path.abspath(path))
        try:
            for fn in os.listdir(d):
                if fn.startswith(stem + "_") and fn.endswith(".tmp"):
                    os.remove(os.path.join(d, fn))
        except OSError:
            pass

    def poll(self):
        while True:                                     # threads never call tkinter themselves (not thread-safe)
            try:
                self.ui.get_nowait()()
            except queue.Empty:
                break
        try:
            while self.worker:
                ev = self.worker.q.get_nowait()
                self.handle(ev)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def handle(self, ev):
        j = self.job
        f = j["tab"]
        k = ev["ev"]
        if k == "model":
            self.status.set("")
        elif k == "stage":
            self._stage(j, f, ev)
        elif k == "photo":
            j["current"] = ev["i"]
            j["n"] = ev["n"]
            j.pop("s0", None)
            j.update(pct="", eta="", passes=False, own="", head="starting…" if f.kind == "depth" else "working…")
            if ev["i"] != j.get("row"):                 # another photo: its own clock (the first photo keeps the clock from Start)
                j["row"], j["row_t0"] = ev["i"], time.time()
            self._render()
            f.tree.see(j["items"][ev["i"] - 1])
        elif k == "step":
            # time left for THIS photo from its own measured step rate, once a third of its steps are done (photos
            # differ in tile count, so one photo's time says nothing about the next)
            d, t = ev["done"], ev["total"]
            if "s0" not in j:
                j["s0"] = time.time()
            eta = ""
            if 3 * d >= t and d < t:
                sec = (time.time() - j["s0"]) / d * (t - d)
                eta = f"about {max(1, round(sec / 60))} min left" if sec >= 60 else f"about {max(5, round(sec, -1)):.0f} s left"
            j.update(pct=f"{100 * d // t} %", eta=eta)
            self._render()
        elif k == "written":
            j.pop("s0", None)
            iid = j["items"][ev["i"] - 1]
            f.tree.set(iid, "status", f"done ({ev['s']:.0f} s)")
            f.outputs[iid] = list(ev.get("files") or [])
            if f.pv_shown == iid:                       # the row being looked at is done: show its result
                self.show_preview(f, iid, to_output=True)
        elif k == "error":
            f.tree.set(j["items"][ev["i"] - 1], "status", "failed")
            self.status.set(f"{os.path.basename(ev['path'])}: {ev['msg']}  (details in {RUN_LOG})")
        elif k == "fatal":
            self.status.set(f"{ev['msg']}  (details in {RUN_LOG})")
        elif k == "warning":
            self.status.set(f"{ev['msg']}  (details in {RUN_LOG})")
        elif k == "exit":
            if j.get("cancelled") and j["current"]:
                self._remove_tmp(f.paths[j["items"][j["current"] - 1]])
            for i in j["items"]:
                f.tree.set(i, "remove", REMOVE_MARK)
                st = f.tree.set(i, "status")            # every row not finished, whatever stage it showed
                if not (st.startswith("done") or st == "failed"):
                    f.tree.set(i, "status", "cancelled" if j.get("cancelled") else "not done")
            if not j.get("cancelled") and ev["code"] != 0 and not self.status.get():
                self.status.set(f"The work stopped unexpectedly (details in {RUN_LOG}).")
            self.worker.log.close()
            self.worker = None
            self._end_job(f)
        self._layout(f)


# the row's part of a stage: up to the first " · " (the fast detail is not shown); in the depth passes the row adds the
# photo's % and time left from the step events
_PASS_STAGES = ("encoding tile", "depth pass", "decoding tile")


def _stage_row(name):
    return name.split(" · ")[0]


def apply_theme(root):
    """The site's dark theme on every ttk widget. 'clam' = the built-in theme that honours colours ('vista' ignores
    them on buttons, tabs and lists)."""
    st = ttk.Style(root)
    st.theme_use("clam")
    body = (SANS, 10)
    root.configure(bg=BG)
    root.option_add("*Font", body)
    st.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE, bordercolor=LINE, lightcolor=BG,
                 darkcolor=BG, troughcolor=BG_ALT, focuscolor=ACCENT, selectbackground=ACCENT_SOFT,
                 selectforeground=TEXT, insertcolor=TEXT, arrowcolor=TEXT_SOFT, font=body)
    st.map(".", foreground=[("disabled", TEXT_FAINT)])
    st.configure("Status.TLabel", foreground=TEXT_SOFT)
    st.configure("Notice.TFrame", background=ACCENT_SOFT)
    st.configure("Notice.TLabel", background=ACCENT_SOFT, foreground=TEXT)
    st.configure("TButton", background=SURFACE, foreground=TEXT, bordercolor=LINE, lightcolor=SURFACE,
                 darkcolor=SURFACE, padding=(12, 5))
    st.map("TButton", background=[("disabled", BG), ("pressed", ACCENT_SOFT), ("active", BG_ALT)],
           bordercolor=[("active", ACCENT)], foreground=[("disabled", TEXT_FAINT)])
    # = the site's .btn--solid: gold fill, dark text
    st.configure("Accent.TButton", background=ACCENT, foreground=ON_ACCENT, bordercolor=ACCENT, lightcolor=ACCENT,
                 darkcolor=ACCENT)
    st.map("Accent.TButton", background=[("disabled", BG_ALT), ("pressed", "#c49a3e"), ("active", "#e2bb66")],
           foreground=[("disabled", TEXT_FAINT)], bordercolor=[("disabled", LINE)])
    for w in ("TRadiobutton", "TCheckbutton"):
        st.configure(w, background=BG, foreground=TEXT, indicatorbackground=SURFACE, indicatorforeground=ACCENT,
                     upperbordercolor=LINE, lowerbordercolor=LINE)
        st.map(w, background=[("active", BG)], indicatorbackground=[("selected", SURFACE), ("active", BG_ALT)],
               foreground=[("disabled", TEXT_FAINT)])
    # tabs (user 2026-10-31: the selected tab must come FORWARD, outlined in gold together with its page): the
    # unselected tabs sit back (darker than the window, faint text, grey edge); the selected tab = the page's own colour,
    # grows 3 px, gold text, and the page + the selected tab are framed in gold
    st.configure("TNotebook", background=BG, bordercolor=ACCENT, lightcolor=BG, darkcolor=BG, tabmargins=(2, 4, 2, 0))
    st.configure("TNotebook.Tab", background=TAB_BACK, foreground=TEXT_FAINT, bordercolor=LINE, lightcolor=TAB_BACK,
                 darkcolor=TAB_BACK, padding=(16, 6), font=(SERIF, 12))
    st.map("TNotebook.Tab", background=[("selected", BG), ("active", BG_ALT)],
           foreground=[("selected", ACCENT), ("active", TEXT)],
           bordercolor=[("selected", ACCENT)], lightcolor=[("selected", BG), ("active", BG_ALT)],
           darkcolor=[("selected", BG), ("active", BG_ALT)], expand=[("selected", (2, 3, 2, 0))])
    # the preview's own two tabs: the body font, so they read as a sub-level of the serif page tabs
    st.configure("Preview.TNotebook.Tab", font=body, padding=(12, 4))
    st.map("Preview.TNotebook.Tab", foreground=[("disabled", TAB_DISABLED), ("selected", ACCENT), ("active", TEXT)])
    st.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT, bordercolor=LINE,
                 lightcolor=SURFACE, darkcolor=SURFACE,
                 rowheight=tkfont.Font(root, font=body).metrics("linespace") + 10)   # pixels: follows the scaling
    st.map("Treeview", background=[("selected", ACCENT_SOFT)], foreground=[("selected", TEXT)])
    st.configure("Sash", sashthickness=6, gripcount=0, background=BG, bordercolor=BG, lightcolor=BG, darkcolor=BG)
    st.configure("Treeview.Heading", background=BG_ALT, foreground=TEXT_SOFT, bordercolor=LINE, lightcolor=BG_ALT,
                 darkcolor=BG_ALT, relief="flat", padding=(6, 4))
    st.map("Treeview.Heading", background=[("active", BG_ALT)])
    st.configure("TProgressbar", background=ACCENT, troughcolor=BG_ALT, bordercolor=LINE, lightcolor=ACCENT,
                 darkcolor=ACCENT)
    st.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE, foreground=TEXT, arrowcolor=TEXT_SOFT,
                 bordercolor=LINE, lightcolor=SURFACE, darkcolor=SURFACE, padding=(6, 4))
    st.map("TCombobox", fieldbackground=[("readonly", SURFACE)], foreground=[("readonly", TEXT)],
           background=[("active", BG_ALT), ("readonly", SURFACE)], bordercolor=[("focus", ACCENT), ("active", ACCENT)],
           selectbackground=[("readonly", SURFACE)], selectforeground=[("readonly", TEXT)],
           arrowcolor=[("active", ACCENT)])
    # the open list of a dropdown is a classic Tk listbox: styled through the option database
    root.option_add("*TCombobox*Listbox.background", SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_SOFT)
    root.option_add("*TCombobox*Listbox.selectForeground", ACCENT)
    root.option_add("*TCombobox*Listbox.font", body)
    root.option_add("*TCombobox*Listbox.borderWidth", 0)
    st.configure("Vertical.TScrollbar", background=LINE, troughcolor=SURFACE, arrowcolor=TEXT_SOFT, bordercolor=LINE,
                 lightcolor=LINE, darkcolor=LINE, gripcount=0)
    st.map("Vertical.TScrollbar", background=[("pressed", ACCENT), ("active", TEXT_FAINT)],
           arrowcolor=[("active", ACCENT)])


def dark_title_bar(root):
    """Windows 10 20H1+ / 11: the title bar follows the window (DWMWA_USE_IMMERSIVE_DARK_MODE = 20; 19 before 20H1).
    Older Windows keeps its grey title bar; nothing else depends on it."""
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        on = ctypes.c_int(1)
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
    except Exception:
        pass


def main(argv=None):
    try:                       # sharp type on a scaled display (Windows would otherwise stretch a blurry bitmap)
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:                       # the taskbar groups the window as its own app (with its icon), not as pythonw.exe
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Vathography.App")
    except Exception:
        pass
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    try:                       # the masthead's "V" (packaging\make_icon.py) instead of Tk's feather; default= also
        root.iconbitmap(default=ICON)   # covers the folder dialogs
    except tk.TclError:
        pass
    try:
        apply_theme(root)
    except tk.TclError:
        pass
    dark_title_bar(root)
    App(root, list(argv or []))
    root.mainloop()
    return 0
