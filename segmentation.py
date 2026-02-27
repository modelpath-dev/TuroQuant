"""Segmentation module — the ONLY file you need to edit to swap in your own logic.

Contract (do not change the return structure):
─────────────────────────────────────────────
  segment(img, resolution, ...)
      → dict with keys:
          "images"  : dict[str, PIL.Image]   — channel outputs (e.g. Seg, Marker, Overlay …)
          "scoring" : dict                   — cell counts (num_total, num_pos, num_neg, percent_pos …)
          "notes"   : list[str]              — any warnings/info for the caller to display

  segment_video_frame(img, resolution, ...)
      → dict with keys:
          "images"  : dict[str, PIL.Image]
          "scoring" : dict
          "cells"   : list[dict]             — per-cell data: [{"centroid": (y, x), "positive": bool}, …]
          "notes"   : list[str]

  check_server() → bool

  PostprocessingError — raised when server-side postprocessing fails

As long as you keep that return shape, app.py will not break.
"""

import base64
import io
import time

import numpy as np
from PIL import Image
import requests

try:
    from deepliif.postprocessing import compute_final_results as _deepliif_compute
    from deepliif.postprocessing import compute_cell_results as _deepliif_cell_results
    from deepliif.postprocessing import cells_to_final_results as _deepliif_cells_to_final
    _DEEPLIIF_PP = True
except ImportError:
    _DEEPLIIF_PP = False


# ─── Exceptions ──────────────────────────────────────────────────────────────

class PostprocessingError(RuntimeError):
    pass


# ─── Internal constants ──────────────────────────────────────────────────────

_API_URL = "https://deepliif.org/api/infer"
_MAX_DIM = 3000
_MIN_DIM = {"40x": 512, "20x": 256, "10x": 128}


# ─── Internal helpers (change freely) ────────────────────────────────────────

def _prepare(img: Image.Image, resolution: str) -> tuple[Image.Image, list[str]]:
    """Resize/pad to satisfy API size constraints."""
    img = img.convert("RGB")
    notes: list[str] = []
    min_dim = _MIN_DIM.get(resolution, 512)

    if img.width > _MAX_DIM or img.height > _MAX_DIM:
        ratio = min(_MAX_DIM / img.width, _MAX_DIM / img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        notes.append(f"scaled down to {img.width}\u00d7{img.height}")

    if img.width < min_dim or img.height < min_dim:
        new_w = max(img.width, min_dim)
        new_h = max(img.height, min_dim)
        padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        padded.paste(img, (0, 0))
        img = padded
        notes.append(f"padded to {new_w}\u00d7{new_h} (min tile={min_dim})")

    return img, notes


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _call_api(png_bytes: bytes, fname: str, params: dict, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                _API_URL,
                files={"img": (fname, png_bytes, "image/png")},
                params=params,
                timeout=180,
            )
            if not resp.ok:
                if resp.status_code == 500 and "nopost" not in params:
                    raise PostprocessingError(
                        "Server postprocessing failed (HTTP 500). "
                        "Try again with nopost=True."
                    )
                try:
                    detail = resp.json()
                except Exception:
                    raw = resp.text[:300].strip()
                    detail = "server returned an error page" if raw.startswith("<") else raw
                if resp.status_code == 500:
                    raise RuntimeError(
                        "Server error (HTTP 500) \u2014 the server may be overloaded. Try again later."
                    )
                if resp.status_code in (502, 503, 504):
                    raise IOError(f"HTTP {resp.status_code}: {detail}")
                raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
            return resp.json()
        except (RuntimeError, PostprocessingError):
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")


def _decode_images(raw: dict) -> dict[str, Image.Image]:
    out = {}
    for key, b64 in raw.get("images", {}).items():
        out[key] = Image.open(io.BytesIO(base64.b64decode(b64)))
    return out


def _extract_scoring(raw: dict) -> dict:
    for key in ("scoring", "scores", "cell_scoring", "score", "results"):
        if key in raw:
            sc = raw[key]
            return sc if isinstance(sc, dict) else {}
    return {}


def _build_params(resolution, prob_thresh, slim, nopost, use_pil) -> dict:
    params: dict = {"resolution": resolution}
    if slim:
        params["slim"] = "true"
    if nopost:
        params["nopost"] = "true"
    else:
        params["prob_thresh"] = int(prob_thresh * 254)
    if use_pil:
        params["pil"] = "true"
    return params


def _to_pil(x):
    if isinstance(x, Image.Image):
        return x
    arr = np.array(x)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _local_scoring(prepared_img, decoded, resolution):
    """Run local postprocessing. Returns (scoring, extra_images)."""
    if not _DEEPLIIF_PP:
        return {}, {}
    seg = decoded.get("Seg") or decoded.get("seg")
    if seg is None:
        return {}, {}
    marker = decoded.get("Marker") or decoded.get("marker")
    try:
        overlay, refined, scoring = _deepliif_compute(
            orig=prepared_img.convert("RGB"),
            seg=seg.convert("RGB"),
            marker=marker.convert("RGB") if marker is not None else None,
            resolution=resolution,
        )
        extras = {"Overlay": _to_pil(overlay), "Refined": _to_pil(refined)}
        return scoring, extras
    except Exception:
        return {}, {}


def _local_cell_scoring(prepared_img, decoded, resolution):
    """Run local postprocessing with per-cell data. Returns (scoring, cells)."""
    if not _DEEPLIIF_PP:
        return {}, []
    seg = decoded.get("Seg") or decoded.get("seg")
    if seg is None:
        return {}, []
    marker = decoded.get("Marker") or decoded.get("marker")
    try:
        seg_rgb = seg.convert("RGB")
        marker_rgb = marker.convert("RGB") if marker is not None else None
        cell_data = _deepliif_cell_results(seg_rgb, marker_rgb, resolution)
        cells = [
            {"centroid": c["centroid"], "positive": bool(c["positive"])}
            for c in cell_data.get("cells", [])
        ]
        _, _, scoring = _deepliif_cells_to_final(
            data=cell_data, orig=prepared_img.convert("RGB"),
        )
        return scoring, cells
    except Exception:
        return {}, []


# ─── Public API (keep the return shape stable) ───────────────────────────────

def segment(
    img: Image.Image,
    resolution: str = "40x",
    prob_thresh: float = 0.5,
    slim: bool = False,
    nopost: bool = False,
    use_pil: bool = False,
) -> dict:
    """Segment a single image. Returns {images, scoring, notes}."""
    prepared, notes = _prepare(img, resolution)
    params = _build_params(resolution, prob_thresh, slim, nopost, use_pil)
    raw = _call_api(_to_png(prepared), "image.png", params)

    scoring = _extract_scoring(raw)
    images = _decode_images(raw)

    local_sc, extras = _local_scoring(prepared, images, resolution)
    if local_sc:
        scoring = local_sc
        images.update(extras)

    return {"images": images, "scoring": scoring, "notes": notes}


def segment_video_frame(
    img: Image.Image,
    resolution: str = "40x",
    prob_thresh: float = 0.5,
    slim: bool = False,
    nopost: bool = False,
    use_pil: bool = False,
) -> dict:
    """Segment a video frame. Returns {images, scoring, cells, notes}."""
    prepared, notes = _prepare(img, resolution)
    params = _build_params(resolution, prob_thresh, slim, nopost, use_pil)
    raw = _call_api(_to_png(prepared), "frame.png", params)

    scoring = _extract_scoring(raw)
    images = _decode_images(raw)

    local_sc, cells = _local_cell_scoring(prepared, images, resolution)
    if local_sc:
        scoring = local_sc

    return {"images": images, "scoring": scoring, "cells": cells, "notes": notes}


def check_server() -> bool:
    """Return True if the TuroQuant server is reachable and healthy."""
    try:
        ping = requests.get("https://deepliif.org", timeout=10)
        return ping.status_code < 500
    except Exception:
        return False
