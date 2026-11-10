# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""photo_io.py: reading photos and writing depth files.

Reads through PyAV (FFmpeg), because PIL differs by up to 54 levels on 4:2:0 JPEGs; mirrored EXIF orientations are not
applied, rotations are. Writes 16-bit PNG (clamped, truncated) and EXR (linear, float or half), + the app's depth tag:
PNG tEXt chunk / EXR string attribute `vathography` = `raw-depth`, so the Relief tab knows an app depth map from an
add-on export.
"""
import os
import struct
import zlib
from fractions import Fraction

import av
import numpy as np

TAG_KEY, TAG_VALUE = "vathography", "raw-depth"
EXR_MAGIC = b"\x76\x2f\x31\x01"
# Scanlines stored in one chunk, by the value of the EXR compression attribute (none, RLE, ZIPS, ZIP, PIZ, PXR24,
# B44, B44A, DWAA, DWAB, HTJ2K256, HTJ2K32).
EXR_CHUNK_LINES = {0: 1, 1: 1, 2: 1, 3: 16, 4: 32, 5: 16, 6: 32, 7: 32, 8: 32, 9: 256, 10: 256, 11: 32}


# ----------------------------------------------------------------------------- read

# Written from a behavioural specification only (a clean-room implementation, 2026-11-09).
# Decoded pixel formats that are read as 8-bit RGB and divided by 255.
_RGB8_FORMATS = frozenset(
    {"rgb24", "rgba", "pal8", "yuvj420p", "yuvj422p", "yuvj444p"}
)

_ALIGN = 32
_F255 = np.float32(255)


def _round_up(n, step):
    return -(-n // step) * step


def _float_target(frame):
    """Planar float target: keep an alpha plane when the source has one,
    so float RGBA passes through unconverted; alpha is dropped later."""
    has_alpha = any(c.is_alpha for c in frame.format.components)
    return "gbrapf32le" if has_alpha else "gbrpf32le"


def _make_pad_graph(frame, width0, height0):
    """Filter graph that pads the picture up to multiples of 32 on both axes
    (image kept at the top-left) and fills the new border by repeating the
    edge pixels. Configuration fails for borders the filter refuses."""
    padded_w = _round_up(width0, _ALIGN)
    padded_h = _round_up(height0, _ALIGN)
    graph = av.filter.Graph()
    src = graph.add_buffer(
        width=frame.width,
        height=frame.height,
        format=frame.format.name,
        time_base=Fraction(1, 1),
    )
    pad = graph.add("pad", f"{padded_w}:{padded_h}:0:0")
    fill = graph.add(
        "fillborders",
        f"right={padded_w - width0}:bottom={padded_h - height0}:mode=smear",
    )
    sink = graph.add("buffersink")
    src.link_to(pad)
    pad.link_to(fill)
    fill.link_to(sink)
    graph.configure()
    return graph


def _decode_frames(path):
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        width0 = height0 = None
        aligned = None
        pad_graph = None

        for frame in container.decode(stream):
            if width0 is None:
                # State fixed by the first frame for the whole call.
                width0, height0 = frame.width, frame.height
                aligned = width0 % _ALIGN == 0

            fmt = frame.format.name
            if fmt in _RGB8_FORMATS:
                rgb = frame.to_ndarray(format="rgb24")
                arr = rgb.astype(np.float32) / _F255
            else:
                target = _float_target(frame)
                if aligned:
                    arr = frame.reformat(format=target).to_ndarray()
                else:
                    if pad_graph is None:
                        pad_graph = _make_pad_graph(frame, width0, height0)
                    pad_graph.push(frame)
                    padded = pad_graph.pull()
                    arr = padded.reformat(format=target).to_ndarray()

            # Top-left crop to the first frame's size, RGB channels only.
            arr = arr[:height0, :width0, :3]

            # Display-matrix rotation, in counter-clockwise quarter turns.
            quarter_turns = frame.rotation // 90
            if quarter_turns:
                arr = np.rot90(arr, quarter_turns)

            # Own, C-ordered float32 buffer per frame.
            frames.append(np.array(arr, dtype=np.float32, order="C", copy=True))
    return frames


def read_photo_bgr_u8(path):
    """A photo file -> uint8 BGR [H, W, 3], the array the node's engine receives."""
    frames = _decode_frames(path)
    if len(frames) != 1:
        raise ValueError(f"{os.path.basename(path)}: {len(frames)} frames; one photo at a time")
    x = frames[0][:, :, :3]
    u8 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)        # = nodes.image_to_bgr_u8
    return np.ascontiguousarray(u8[:, :, ::-1])


# ----------------------------------------------------------------------------- write

def _encode(img_np, codec_name, frame_fmt, stream_fmt, options=None):
    """One image -> the bytes of one file, straight from a codec context."""
    context = av.CodecContext.create(codec_name, "w")
    context.height, context.width = img_np.shape[:2]
    context.pix_fmt = stream_fmt
    context.time_base = Fraction(1, 1)
    if options:
        context.options = dict(options)
    frame = av.VideoFrame.from_ndarray(img_np, format=frame_fmt)
    if frame_fmt != stream_fmt:
        frame = frame.reformat(format=stream_fmt)
    frame.pts = 0
    frame.time_base = context.time_base
    packets = list(context.encode(frame)) + list(context.encode(None))
    return b"".join(bytes(packet) for packet in packets)


def _rgb(img):
    """[H, W] grey (written in three channels, as the node's IMAGE) or [H, W, 3] -> float32 [H, W, 3]."""
    return (np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img[..., :3]).astype(np.float32)


def encode_png16(img01):
    """float32 [H, W] or [H, W, 3], 0..1 -> 16-bit RGB PNG bytes."""
    q = np.clip(_rgb(img01) * 65535.0, 0, 65535).astype(np.int32).astype(np.uint16)  # truncation, as core
    return _encode(q, "png", "rgb48le", "rgb48be")


def encode_png8(img01):
    """float32 [H, W] or [H, W, 3], 0..1 -> 8-bit RGB PNG bytes (x * 255 clamped, truncated: core's 8-bit save)."""
    q = np.clip(_rgb(img01) * 255.0, 0, 255).astype(np.uint8)
    return _encode(np.ascontiguousarray(q), "png", "rgb24", "rgb24")


def encode_exr(img, half=False):
    """float32 [H, W] or [H, W, 3] -> EXR bytes, 32-bit float or half."""
    rgb = _rgb(img)
    if half:
        rgb = rgb.astype(np.float16).astype(np.float32)               # numpy rounds; FFmpeg's own cast truncates
    options = {"format": "half" if half else "float"}
    return _encode(np.ascontiguousarray(rgb), "exr", "gbrpf32le", "gbrpf32le", options)


def read_image_float(path):
    """Any picture -> float32 [H, W, 3] (8-bit / 0..255 scaled; 16-bit PNG and EXR as float). One frame only."""
    frames = _decode_frames(path)
    if len(frames) != 1:
        raise ValueError(f"{os.path.basename(path)}: {len(frames)} frames; one picture at a time")
    return frames[0][:, :, :3]


def tag_png(data, key=TAG_KEY, value=TAG_VALUE):
    """Insert one tEXt chunk right after IHDR."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.unpack(">I", data[8:12])[0]
    cut = 8 + 12 + ihdr_len
    body = key.encode("latin-1") + b"\x00" + value.encode("latin-1")
    chunk = struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF)
    return data[:cut] + chunk + data[cut:]


def _exr_attributes(buf):
    """Walk an EXR header: yields (name, type, value start, value end), ending at the header's closing null byte or
    where an attribute would run past the end of buf."""
    pos = 8
    while pos < len(buf) and buf[pos] != 0:
        name_end = buf.find(b"\x00", pos)
        type_end = buf.find(b"\x00", name_end + 1)
        if name_end < 0 or type_end < 0 or type_end + 5 > len(buf):
            return
        size = struct.unpack_from("<i", buf, type_end + 1)[0]
        start = type_end + 5
        if size < 0 or start + size > len(buf):
            return
        yield buf[pos:name_end], buf[name_end + 1:type_end], start, start + size
        pos = start + size


def tag_exr(data, key=TAG_KEY, value=TAG_VALUE):
    """Add one string attribute at the end of a single-part scanline EXR's header; the chunk offsets move with it."""
    assert data[:4] == EXR_MAGIC
    header_end = 8
    for name, _, start, stop in _exr_attributes(data):
        header_end = stop
        if name == b"compression":
            compression = data[start]
        elif name == b"dataWindow":
            _, y_min, _, y_max = struct.unpack_from("<4i", data, start)
    assert data[header_end] == 0

    lines = EXR_CHUNK_LINES[compression]
    chunks = (y_max - y_min + 1 + lines - 1) // lines
    text = value.encode("utf-8")
    attribute = key.encode("utf-8") + b"\x00string\x00" + struct.pack("<i", len(text)) + text

    table_start = header_end + 1
    table_end = table_start + 8 * chunks
    offsets = struct.unpack_from(f"<{chunks}Q", data, table_start)
    table = struct.pack(f"<{chunks}Q", *(offset + len(attribute) for offset in offsets))
    return data[:header_end] + attribute + data[header_end:table_start] + table + data[table_end:]


def read_tag(path):
    """'raw-depth' when the file carries the app's tag, else None (PNG tEXt / EXR string attribute)."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    needle = TAG_KEY.encode("latin-1")
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        pos = 8
        while pos + 8 <= len(head):
            ln = struct.unpack(">I", head[pos:pos + 4])[0]; typ = head[pos + 4:pos + 8]
            if typ == b"tEXt":
                body = head[pos + 8:pos + 8 + ln]
                k, _, v = body.partition(b"\x00")
                if k == needle:
                    return v.decode("latin-1")
            if typ in (b"IDAT", b"IEND"):
                return None
            pos += 12 + ln
        return None
    if head[:4] == EXR_MAGIC:
        for name, _, start, stop in _exr_attributes(head):
            if name == needle:
                return head[start:stop].decode("utf-8")
    return None


ENCODERS = {"png": encode_png16, "png8": encode_png8, "exr": encode_exr,
            "exr16": lambda x: encode_exr(x, half=True)}     # png = 16-bit PNG, exr = 32-bit float EXR


def encode_depth(depth01, kind, tagged=True):
    """kind 'png' (16-bit) | 'png8' | 'exr' (32-bit float) | 'exr16' (half) -> the file's bytes. tagged=False for a
    relief written from an add-on export (it is not an app depth map)."""
    data = ENCODERS[kind](depth01)
    if tagged:
        data = tag_png(data) if kind.startswith("png") else tag_exr(data)
    return data


def write_depth(path, depth01, kind, tagged=True):
    """One file at an exact name, never over an existing file (FileExistsError). Crash-safe: <path>.tmp, then
    os.rename (Windows refuses an existing target). The app's outputs go through write_versioned instead."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(encode_depth(depth01, kind, tagged))
    try:
        os.rename(tmp, path)
    except OSError:
        os.remove(tmp)
        raise
    return path
