# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""backend.py — the app's model backend:
Marigold V2 one-step inference on diffusers' QwenImageTransformer2DModel with the node's numerics, no ComfyUI, no
bitsandbytes. Layer-major (bit-identical to a full load), VAE in 1024-px tiles (stride 768; judged the same as
untiled; fits an 8 GB card), inputs in GROUPS sized to the GPU (`run_grouped`).

Numerics (each line = where the node does the same thing):
  input      uint8 BGR -> RGB float32 / 255                    native_backend.to_image
             * 2 - 1, cast to the VAE dtype (bf16)            comfy/sd.py process_input
  encode     posterior MEAN (latent_dist.mode())               comfy WanVAE.encode returns mu
  latent     (z - mean) / std, float32                         latent_formats.Wan21.process_in (scale 1.0)
  model      bf16 activations; nf4 weights decoded to bf16; bf16 non-4-bit params; LoRA scale 1, merged into the streamed blocks' nf4 weights once per pass (fp32 add, bf16 result), fp32 unmerged elsewhere
                                                               nf4_loader.NF4Linear.forward (same code below)
             timestep 0.5 (sigma 0.5 x multiplier 1.0)         ModelSamplingAuraFlow multiplier=1.0 + ManualSigmas 0.5, 0
             no text mask (the conditioning file has none)     ConditioningLoader -> no attention mask
  step       x0 = x - v  (euler 0.5 -> 0, IMG_TO_IMG_VELOCITY: denoised = x - v; one euler step lands on it)
  decode     z * std + mean, bf16, decode, (x + 1) / 2 clamped to [0, 1]   comfy/sd.py:514 process_output
  raw        2 * img - 1, channel mean, float32                native_backend.py:173
Model code = diffusers' class; it differs from ComfyUI's qwen_image code in small ways (measured: maps similar, not
bit-identical).

Execution modes:
  full        everything on the GPU
  per_call    NF4Linear tensors stay in pinned RAM, moved to the GPU on every call; the rest on the GPU
  layer_major one block on the GPU at a time, every input through it before the next block; hidden states of
              the waiting inputs on the GPU (states="gpu") or in pinned RAM (states="cpu")
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

FORMAT = "vathography-nf4-v1"
SIGMA = 0.5
# attention kernels, best first. Flash is missing from the Windows torch build, so cuDNN runs (2.6x faster per call
# than PyTorch's default "efficient" kernel; a 33 MP photo 141 -> 115 s, map rms 0.0029 vs it, judged similar).
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    KERNELS_BEST_FIRST = [getattr(SDPBackend, k) for k in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION",
                                                           "MATH")]
except ImportError:                                  # the pinned torch 2.10 has it; never reached in the package
    sdpa_kernel = None
LORA_PREFIX = "diffusion_model."
# img_in lora_B |sum| (float32) -> recipe; = native_backend.LORA_FINGERPRINTS (node 1.3.0)
LORA_FINGERPRINTS = {"depth": 3005.3411, "see-through": 3126.4729}
HERE = os.path.dirname(os.path.abspath(__file__))
VAE_KEYMAP = os.path.join(HERE, "vae_keymap.json")


VAE_TILE, VAE_STRIDE = 1024, 768          # the smallest VAE tile judged the same as untiled
# Measured on a whole photo under the 6.5 GiB cap (layer-major, VAE 1024), RESERVED memory (what the cap counts): 31
# inputs -> 6.12 GiB reserved with every hidden state on the GPU; two inputs -> 3.4-4.3 GiB. Per input ~ 0.08 GiB, base
# ~ 3.9 GiB => 32 inputs per weight pass at 6.5 GiB (a 66 MP photo = 64 inputs = 2 passes).
STATE_GIB_PER_INPUT = 0.08
BASE_GIB = 3.9
# kept free for allocator fragmentation: sized to the cap with 0.04 GiB spare, the 2nd landscape of a batch ran out of
# memory (3.86 GiB in use + 2.11 reserved-unused, cap 6.5); with this margin (group 23 at 6.5) both passed, maps
# unchanged. expandable_segments, PyTorch's remedy, is not supported on Windows.
MARGIN_GIB = 0.7


def log(*a):
    print("[backend]", *a, flush=True)


def group_size(budget_gib):
    """How many inputs go through the 60 blocks together: their hidden states stay on the GPU (S0 run 2: moving states
    to RAM per block cost 236 s of 265 s on a 64-input photo). One weight pass (~13 GB host->GPU) per group."""
    return max(1, int((budget_gib - BASE_GIB - MARGIN_GIB) / STATE_GIB_PER_INPUT))


# ----------------------------------------------------------------------------- nf4 layer (= the node's, ComfyUI-free)

def nf4_decode(packed, absmax, quant_map, shape, blocksize):
    p = packed.view(-1)
    idx = torch.stack(((p >> 4) & 0xF, p & 0xF), dim=1).view(-1).long()
    return (quant_map[idx].view(-1, blocksize) * absmax.view(-1, 1)).view(shape).to(torch.bfloat16)


class NF4Linear(torch.nn.Module):
    """Same forward as nf4_loader.NF4Linear (node 1.3.0): input to bf16, weight decoded to bf16, linear, + LoRA in
    fp32, output back to the input dtype. Tensors left in RAM are moved to the input's device on each call (counted
    in `h2d_bytes` for the per_call arm)."""
    h2d_bytes = 0

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        for name in ("packed", "absmax", "quant_map", "weight", "bias", "lora_A", "lora_B"):
            self.register_parameter(name, None)
        self.shape, self.blocksize = (out_features, in_features), None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        p = lambda t: torch.nn.Parameter(t, requires_grad=False)
        if prefix + "weight.absmax" in state_dict:
            qs = json.loads(bytes(state_dict.pop(prefix + "weight.quant_state.bitsandbytes__nf4").tolist()).decode("utf-8"))
            assert qs["quant_type"] == "nf4" and qs["dtype"] == "bfloat16", (prefix, qs)
            self.shape, self.blocksize = tuple(qs["shape"]), int(qs["blocksize"])
            self.packed = p(state_dict.pop(prefix + "weight"))
            self.absmax = p(state_dict.pop(prefix + "weight.absmax"))
            self.quant_map = p(state_dict.pop(prefix + "weight.quant_map"))
        elif prefix + "weight" in state_dict:
            self.weight = p(state_dict.pop(prefix + "weight"))
        else:
            missing_keys.append(prefix + "weight")
        if prefix + "bias" in state_dict:
            self.bias = p(state_dict.pop(prefix + "bias"))

    def _dev(self, t, dev):
        if t.device != dev:
            NF4Linear.h2d_bytes += t.numel() * t.element_size()
            return t.to(dev, non_blocking=True)
        return t

    def forward(self, x):
        dev, xb = x.device, x.to(torch.bfloat16)
        w = getattr(self, "_wcache", None)                     # set by Streamer.upload: unpacked once per weight pass
        if w is None and self.packed is not None:
            w = nf4_decode(self._dev(self.packed, dev), self._dev(self.absmax, dev), self._dev(self.quant_map, dev),
                           self.shape, self.blocksize)
        elif w is None:
            w = self._dev(self.weight, dev).to(torch.bfloat16)
        out = F.linear(xb, w, None if self.bias is None else self._dev(self.bias, dev).to(torch.bfloat16))
        if self.lora_A is not None and not getattr(self, "_merged", False):   # merged layers carry the LoRA in w
            out = out + ((xb.float() @ self._dev(self.lora_A, dev).float().t()) @ self._dev(self.lora_B, dev).float().t()).to(out.dtype)
        return out.to(x.dtype)


# ----------------------------------------------------------------------------- loading

_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8, "I8": torch.int8,
              "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool}


def load_lowmem(path, progress=None):
    """safetensors file -> {name: tensor}, each tensor read straight from the file into ITS OWN allocation (S0 run 1:
    safetensors 0.8's load_file / get_tensor keep the whole file as one allocation, freed only when every tensor is
    gone => the weights sat in RAM twice once placed / streamed). Same bytes as load_file (selftest T6)."""
    import struct
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        base = 8 + n
        for k, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = v["data_offsets"]
            buf = torch.empty(b - a, dtype=torch.uint8)
            f.seek(base + a)
            got = f.readinto(memoryview(buf.numpy()))
            if got != b - a:
                raise IOError(f"{path}: short read on {k}")
            if progress:
                progress(got)
            out[k] = buf.view(_ST_DTYPES[v["dtype"]]).reshape(v["shape"]) if b > a else \
                torch.empty(v["shape"], dtype=_ST_DTYPES[v["dtype"]])
    return out


def load_transformer(nf4_path, lora_path, stage=None, label="depth"):
    """diffusers' class, every Linear swapped for NF4Linear, weights from the ComfyUI-layout nf4 file (names = diffusers'
    1:1), LoRA attached unmerged. Returns (transformer on the CPU, recipe, info)."""
    stage = stage or (lambda *a, **k: None)
    stage(f"preparing the {label} model")
    from accelerate import init_empty_weights
    from diffusers import QwenImageTransformer2DModel

    t0 = time.time()
    with safe_open(nf4_path, "pt") as f:
        meta = f.metadata() or {}
    if meta.get("format") != FORMAT or meta.get("double_quant") != "false" or "unet_config" not in meta:
        raise ValueError(f"{nf4_path} is not the {FORMAT} backbone")
    uc = json.loads(meta["unet_config"])
    with init_empty_weights():
        tr = QwenImageTransformer2DModel()                     # defaults = the 2509 transformer config
    assert tr.config.num_layers == uc["num_layers"] and tr.config.in_channels == uc["in_channels"], (tr.config, uc)
    n_swap = 0
    n_lin = sum(1 for m in tr.modules() if isinstance(m, torch.nn.Linear))
    for mod in list(tr.modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(mod, cname, NF4Linear(child.in_features, child.out_features, bias=child.bias is not None))
                n_swap += 1
                stage(f"preparing the {label} model", n_swap, n_lin)
    # one byte counter over the backbone AND the LoRA
    nbytes = os.path.getsize(nf4_path) + os.path.getsize(lora_path)
    read = [0]

    def got(n):
        read[0] += n
        stage(f"loading the {label} model", read[0], nbytes)
    sd = load_lowmem(nf4_path, got)
    missing, unexpected = tr.load_state_dict(sd, strict=False, assign=True)
    del sd
    if missing or unexpected:
        raise ValueError(f"backbone load: missing {missing[:5]} ({len(missing)}) | unexpected {unexpected[:5]} ({len(unexpected)})")
    meta_left = [n for n, p in tr.named_parameters() if p.is_meta] + [n for n, b in tr.named_buffers() if b.is_meta]
    if meta_left:
        raise ValueError(f"{len(meta_left)} tensors left unloaded, e.g. {meta_left[:3]}")
    lins = {n: m for n, m in tr.named_modules() if isinstance(m, NF4Linear)}
    bad = [n for n, m in lins.items() if m.packed is None and m.weight is None]
    if bad:
        raise ValueError(f"{len(bad)} layers without weights, e.g. {bad[0]}")
    lsd = load_lowmem(lora_path, got)
    n_l = 0
    for k in [k for k in lsd if k.endswith(".lora_A.weight")]:
        name = k[len(LORA_PREFIX):-len(".lora_A.weight")]
        m = lins.get(name)
        if m is None:
            raise ValueError(f"the LoRA targets '{name}', not a Linear layer of this backbone")
        m.lora_A = torch.nn.Parameter(lsd[k], requires_grad=False)
        m.lora_B = torch.nn.Parameter(lsd[k.replace(".lora_A.", ".lora_B.")], requires_grad=False)
        n_l += 1
    fp = float(lsd[LORA_PREFIX + "img_in.lora_B.weight"].float().abs().sum())
    recipe = next((r for r, ref in LORA_FINGERPRINTS.items() if abs(fp - ref) < 0.05), None)
    if recipe is None:
        raise ValueError(f"unknown LoRA (img_in fingerprint {fp:.4f})")
    tr.eval().requires_grad_(False)
    n_q = sum(1 for m in lins.values() if m.packed is not None)
    info = dict(linear_swapped=n_swap, nf4_layers=n_q, bf16_linear=len(lins) - n_q, lora_layers=n_l, recipe=recipe,
                lora_fingerprint=round(fp, 4), load_s=round(time.time() - t0, 1))
    log(f"transformer: {info}")
    return tr, recipe, info


def load_vae(vae_path, config_dir):
    """diffusers' AutoencoderKLQwenImage, ComfyUI-layout weights renamed by vae_keymap.json (194 names), bf16."""
    from diffusers import AutoencoderKLQwenImage
    km = json.load(open(VAE_KEYMAP, encoding="utf-8"))
    sc = load_file(vae_path)
    if set(km.values()) != set(sc):
        raise ValueError(f"{vae_path}: names do not match the key map")
    vae = AutoencoderKLQwenImage.from_config(AutoencoderKLQwenImage.load_config(config_dir))
    vae.load_state_dict({d: sc[c] for d, c in km.items()}, strict=True)
    vae = vae.to(torch.bfloat16).eval().requires_grad_(False)
    vae.enable_tiling(tile_sample_min_height=VAE_TILE, tile_sample_min_width=VAE_TILE,
                      tile_sample_stride_height=VAE_STRIDE, tile_sample_stride_width=VAE_STRIDE)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1)
    return vae, mean, std


def load_conditioning(path):
    c = load_file(path)
    assert list(c) == ["conditioning"], list(c)
    return c["conditioning"]                                   # [1, L, 3584] bf16, no mask


# ----------------------------------------------------------------------------- VAE

def set_vae_tiling(vae, tiled, tile_px=None):
    if tiled:
        kw = {} if tile_px is None else dict(tile_sample_min_height=tile_px, tile_sample_min_width=tile_px)
        vae.enable_tiling(**kw)
    else:
        vae.disable_tiling()


@torch.inference_mode()
def encode(vae, mean, std, bgr_u8, dev):
    assert bgr_u8.dtype == np.uint8 and bgr_u8.ndim == 3 and bgr_u8.shape[0] % 16 == 0 and bgr_u8.shape[1] % 16 == 0
    rgb = bgr_u8[:, :, ::-1].astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    x = (x * 2.0 - 1.0).to(dev, torch.bfloat16)[:, :, None]
    mu = vae.encode(x).latent_dist.mode()[:, :, 0].float()
    return (mu - mean.to(dev)) / std.to(dev)                   # [1, 16, H/8, W/8] float32


@torch.inference_mode()
def decode_raw(vae, mean, std, lat, dev):
    z = (lat.to(dev) * std.to(dev) + mean.to(dev)).to(torch.bfloat16)[:, :, None]
    img = vae.decode(z).sample[:, :, 0].float()
    img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
    return (2.0 * img - 1.0).mean(dim=1)[0].cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------- transformer step

def pack(lat):
    b, c, h, w = lat.shape
    return lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(tok, h, w, c=16):
    b = tok.shape[0]
    return tok.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def _timestep(dev):
    return torch.full((1,), SIGMA, device=dev, dtype=torch.bfloat16)


@torch.inference_mode()
def step_full(tr, cond, lat, dev):
    """One euler step through diffusers' own forward. lat: [1, 16, h, w] float32 (normalised). Returns x - v."""
    h, w = lat.shape[-2:]
    v = tr(hidden_states=pack(lat).to(dev, torch.bfloat16), encoder_hidden_states=cond.to(dev, torch.bfloat16),
           encoder_hidden_states_mask=None, timestep=_timestep(dev), img_shapes=[[(1, h // 2, w // 2)]],
           return_dict=False)[0]
    return lat.to(dev) - unpack(v.float(), h, w)


NON_BLOCK = ("img_in", "txt_norm", "txt_in", "time_text_embed", "pos_embed", "norm_out", "proj_out")


class Streamer:
    """Keeps every block's tensors in (pinned) RAM and swaps one block's tensors onto the GPU at a time."""

    def __init__(self, tr, pin=False):                         # S0 run 1: pinning doubled RAM (rounded allocations)
        self.blocks, self.pinned = [], pin and torch.cuda.is_available()
        self.nf4 = [[m for m in b.modules() if isinstance(m, NF4Linear) and m.packed is not None]
                    for b in tr.transformer_blocks]
        for b in tr.transformer_blocks:
            ents = []
            for m in b.modules():
                for n, p in list(m._parameters.items()):
                    if p is None:
                        continue
                    cpu = p.data.cpu()
                    if self.pinned:
                        cpu = cpu.pin_memory()
                    p.data = cpu
                    ents.append((p, cpu))
            self.blocks.append(ents)
        self.h2d_bytes, self.upload_s = 0, 0.0

    def upload(self, i, dev):
        t0 = time.time()
        for p, cpu in self.blocks[i]:
            p.data = cpu.to(dev, non_blocking=True)
            if torch.device(dev).type != "cpu":
                self.h2d_bytes += cpu.numel() * cpu.element_size()
        # each nf4 layer unpacked ONCE for the whole group (was: once per input per layer). Same nf4_decode on the same
        # tensors => the same bf16 weight => map identical (V8 P1b: max |d| 0.0, must-fire 0.43; -9.3 s on DSCF0813)
        for m in self.nf4[i]:
            m._wcache = nf4_decode(m.packed, m.absmax, m.quant_map, m.shape, m.blocksize)
            # LoRA merged into that weight once per pass (ComfyUI's way): bf16(W + B·A) in fp32, then one bf16 linear per
            # call instead of + x·A·B in fp32 (a 33 MP photo 117 -> 103 s, merge 0.6 s; map changed slightly, judged the same)
            if m.lora_A is not None:
                m._wcache = (m._wcache.float() + m.lora_B.float() @ m.lora_A.float()).to(torch.bfloat16)
                m._merged = True
        if torch.device(dev).type == "cuda":
            torch.cuda.synchronize(dev)
        self.upload_s += time.time() - t0

    def release(self, i):
        for m in self.nf4[i]:
            m._wcache, m._merged = None, False
        for p, cpu in self.blocks[i]:
            p.data = cpu


def place(tr, mode, dev):
    """Puts the transformer in the arm's placement; returns a Streamer for layer_major."""
    if mode == "full":
        tr.to(dev)
        return None
    if mode == "per_call":
        for m in tr.modules():
            if isinstance(m, NF4Linear):
                for n, p in list(m._parameters.items()):
                    if p is not None:
                        p.data = p.data.cpu()                  # pageable (S0 run 1: pinning doubled RAM)
            else:
                for n, p in list(m._parameters.items()):
                    if p is not None:
                        p.data = p.data.to(dev)
        return None
    if mode == "layer_major":
        for name in NON_BLOCK:
            getattr(tr, name).to(dev)
        return Streamer(tr)
    raise ValueError(mode)


@torch.inference_mode()
def step_layer_major(tr, streamer, cond, lats, dev, states="gpu", tick=None, sub=None):
    """All inputs through block 1, then block 2, ... = the same module calls as diffusers' forward, in the same order
    per input (lines transformer_qwenimage.py:910-988 with mask None, no zero_cond_t, no controlnet)."""
    st = []
    for lat in lats:
        h, w = lat.shape[-2:]
        hs = tr.img_in(pack(lat).to(dev, torch.bfloat16))
        t = _timestep(dev).to(hs.dtype)
        enc = tr.txt_in(tr.txt_norm(cond.to(dev, torch.bfloat16)))
        temb = tr.time_text_embed(t, hs)
        rope = tr.pos_embed([[(1, h // 2, w // 2)]], max_txt_seq_len=enc.shape[1], device=dev)
        if states == "cpu" and torch.device(dev).type == "cuda":
            hs, enc = hs.cpu().pin_memory(), enc.cpu().pin_memory()
        st.append(dict(hs=hs, enc=enc, temb=temb, rope=rope, h=h, w=w))
    t_move = 0.0
    for i, block in enumerate(tr.transformer_blocks):
        streamer.upload(i, dev)
        for k, s in enumerate(st, 1):
            if sub:
                sub(i + 1, k)
            hs, enc = s["hs"], s["enc"]
            off = states == "cpu" and torch.device(dev).type == "cuda"
            if off:
                t0 = time.time(); hs, enc = hs.to(dev, non_blocking=True), enc.to(dev, non_blocking=True); t_move += time.time() - t0
            enc, hs = block(hidden_states=hs, encoder_hidden_states=enc, encoder_hidden_states_mask=None, temb=s["temb"],
                            image_rotary_emb=s["rope"], joint_attention_kwargs={}, modulate_index=None)
            if off:
                t0 = time.time(); hs, enc = hs.cpu().pin_memory(), enc.cpu().pin_memory(); t_move += time.time() - t0
            s["hs"], s["enc"] = hs, enc
        streamer.release(i)
        if tick:
            tick(i + 1, len(tr.transformer_blocks))
    outs = []
    for s, lat in zip(st, lats):
        hs = s["hs"].to(dev)
        v = tr.proj_out(tr.norm_out(hs, s["temb"]))
        outs.append(lat.to(dev) - unpack(v.float(), s["h"], s["w"]))
    return outs, dict(state_moves_s=round(t_move, 2))


def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def run_inputs(tr, vae, mean, std, cond, inputs, dev, mode, streamer=None, states="gpu", log_fn=None):
    """inputs: list of uint8 BGR (/16). Encode all -> step (per mode) -> decode all. Returns (raw list, x0 list, timing)."""
    say = log_fn or (lambda m: None)
    tm = {}
    t0 = time.time()
    lats = [encode(vae, mean, std, x, dev).cpu() for x in inputs]
    _sync(dev); tm["encode_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    if mode == "layer_major":
        outs, extra = step_layer_major(tr, streamer, cond, lats, dev, states=states)
        tm.update(extra)
        x0 = [o.cpu() for o in outs]
    else:
        x0 = [step_full(tr, cond, lat, dev).cpu() for lat in lats]
    _sync(dev); tm["step_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    raws = [decode_raw(vae, mean, std, z, dev) for z in x0]
    _sync(dev); tm["decode_s"] = round(time.time() - t0, 2)
    say(f"timing {tm}")
    return raws, x0, tm


class Model:
    """The loaded model for one recipe ("depth" / "see-through"), placed layer-major on the GPU. `infer(inputs,
    progress)` = the contract of the node's run_pipeline (inputs: uint8 BGR /16, M1 first; returns float32 (H, W) raw
    predictions in the same order)."""

    def __init__(self, nf4_path, lora_path, vae_path, cond_path, vae_config_dir, device="cuda", budget_gib=6.5, log_fn=None,
                 stage=None, label="depth"):
        """stage(name, done=None, total=None): progress of each loading step (None = silent)."""
        self.say = log_fn or log
        self.stage = stage or (lambda *a, **k: None)
        self.dev = torch.device(device)
        self.tr, self.recipe, self.info = load_transformer(nf4_path, lora_path, stage=self.stage, label=label)
        self.stage("loading the image decoder")
        self.vae, self.mean, self.std = load_vae(vae_path, vae_config_dir)
        self.cond = load_conditioning(cond_path)
        self.stage("moving the model to the graphics card")
        self.streamer = place(self.tr, "layer_major", self.dev)
        self.vae.to(self.dev)
        self.group = group_size(budget_gib)
        self.say(f"model {self.recipe}: layer-major on {self.dev}, VAE tiles {VAE_TILE}/{VAE_STRIDE}, "
                 f"up to {self.group} inputs per weight pass (budget {budget_gib} GiB), attention "
                 f"{'cuDNN first' if sdpa_kernel else 'PyTorch default'}")

    def infer(self, inputs, progress=None, stage=None):
        if sdpa_kernel is None:
            return self._infer(inputs, progress, stage)
        with sdpa_kernel(KERNELS_BEST_FIRST, set_priority=True):
            return self._infer(inputs, progress, stage)

    @torch.inference_mode()
    def _infer(self, inputs, progress=None, stage=None):
        """stage(name, done, total): the sub-step names (per input, also inside a block)."""
        stage = stage or (lambda *a, **k: None)
        n = len(inputs)
        total = 2 * n + -(-n // self.group) * len(self.tr.transformer_blocks)
        done = [0]

        def tick(k=1):
            done[0] += k
            if progress:
                progress(done[0], total)
        lats = []
        for k, x in enumerate(inputs, 1):
            stage(f"encoding tile {k}/{n}", k - 1, n)
            lats.append(encode(self.vae, self.mean, self.std, x, self.dev).cpu())
            tick()
        stage(f"encoding tile {n}/{n}", n, n)
        x0 = []
        passes = -(-n // self.group)
        nb = len(self.tr.transformer_blocks)
        for pi, g in enumerate(range(0, n, self.group), 1):
            kk = len(lats[g:g + self.group])

            def sub(b, k, pi=pi, kk=kk):        # block b (1-based) is about to run input k (1-based) of this pass
                stage(f"depth pass {pi}/{passes} · block {b}/{nb} · tile {k}/{kk}", (b - 1) * kk + k - 1, nb * kk)
            outs, _ = step_layer_major(self.tr, self.streamer, self.cond, lats[g:g + self.group], self.dev,
                                       tick=lambda i, t: tick(), sub=sub)
            x0 += [o.cpu() for o in outs]
            del outs
        preds = []
        for k, z in enumerate(x0, 1):
            stage(f"decoding tile {k}/{n}", k - 1, n)
            preds.append(decode_raw(self.vae, self.mean, self.std, z, self.dev))
            tick()
        stage(f"decoding tile {n}/{n}", n, n)
        return preds
