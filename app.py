import streamlit as st
import requests
import base64
import json
import zipfile
import io
import tempfile
import os
import time

import cv2
import numpy as np
import tifffile
from PIL import Image

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


# ─── Constants ───────────────────────────────────────────────────────────────

DEEPLIIF_API = "https://deepliif.org/api/infer"

IMAGE_FORMATS = [
    "png", "jpg", "jpeg",
    "tif", "tiff",          # priority
    "bmp", "gif",
    "svs", "ndpi", "scn",   # whole-slide
    "czi", "lif", "mrxs",
    "vms", "vmu", "qptiff",
]

VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]

ALL_FORMATS = IMAGE_FORMATS + VIDEO_FORMATS

# Human-readable labels for the format dropdown shown to users
FORMAT_LABELS = {
    "png":    "PNG – Portable Network Graphics",
    "jpg":    "JPG/JPEG – Joint Photographic Experts Group",
    "jpeg":   "JPG/JPEG – Joint Photographic Experts Group",
    "tif":    "TIF/TIFF – Tagged Image File Format  ★",
    "tiff":   "TIF/TIFF – Tagged Image File Format  ★",
    "bmp":    "BMP – Bitmap",
    "gif":    "GIF – Graphics Interchange Format",
    "svs":    "SVS – Aperio Whole Slide Image",
    "ndpi":   "NDPI – Hamamatsu Whole Slide Image",
    "scn":    "SCN – Leica Whole Slide Image",
    "czi":    "CZI – Zeiss Confocal",
    "lif":    "LIF – Leica Image File",
    "mrxs":   "MRXS – 3DHistech Whole Slide",
    "vms":    "VMS – Hamamatsu Virtual Microscope Slide",
    "vmu":    "VMU – Hamamatsu Uncompressed Virtual Slide",
    "qptiff": "QPTIFF – PerkinElmer Whole Slide",
    "mp4":    "MP4 – Video  ★",
    "avi":    "AVI – Video",
    "mov":    "MOV – Apple QuickTime Video",
    "mkv":    "MKV – Matroska Video",
    "webm":   "WebM – Web Video",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_video_frames(video_bytes: bytes, ext: str, every_n_sec: float) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(video_fps * every_n_sec))
        frames, idx = [], 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
    finally:
        os.unlink(tmp_path)
    return frames


def tif_pages(tif_bytes: bytes) -> list[Image.Image]:
    pages = []
    try:
        with tifffile.TiffFile(io.BytesIO(tif_bytes)) as tf:
            for page in tf.pages:
                arr = page.asarray()
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                pages.append(Image.fromarray(arr.astype(np.uint8)))
    except Exception:
        img = Image.open(io.BytesIO(tif_bytes))
        for i in range(getattr(img, "n_frames", 1)):
            img.seek(i)
            pages.append(img.convert("RGB").copy())
    return pages


MAX_DIM = 3000  # DeepLIIF API upper limit

# Minimum tile size per resolution — smaller images cause a server 500
MIN_DIM = {"40x": 512, "20x": 256, "10x": 128}


def prepare_image(img: Image.Image, resolution: str) -> tuple[Image.Image, list[str]]:
    """Resize/pad image to satisfy DeepLIIF size constraints. Returns (img, warnings)."""
    img = img.convert("RGB")
    notes = []
    min_dim = MIN_DIM.get(resolution, 512)

    # Scale down if too large
    if img.width > MAX_DIM or img.height > MAX_DIM:
        ratio = min(MAX_DIM / img.width, MAX_DIM / img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        notes.append(f"scaled down to {img.width}×{img.height}")

    # Pad up if too small
    if img.width < min_dim or img.height < min_dim:
        new_w = max(img.width, min_dim)
        new_h = max(img.height, min_dim)
        padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        padded.paste(img, (0, 0))
        img = padded
        notes.append(f"padded to {new_w}×{new_h} (min tile={min_dim})")

    return img, notes


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def call_api(png_bytes: bytes, fname: str, params: dict, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                DEEPLIIF_API,
                files={"img": (fname, png_bytes, "image/png")},
                params=params,
                timeout=180,
            )
            if not resp.ok:
                if resp.status_code == 500 and "nopost" not in params:
                    raise PostprocessingError(
                        "Server postprocessing failed (HTTP 500). "
                        "Enable **Skip postprocessing** in the sidebar and click Run again."
                    )
                try:
                    detail = resp.json()
                except Exception:
                    raw = resp.text[:300].strip()
                    detail = "server returned an error page" if raw.startswith("<") else raw
                if resp.status_code == 500:
                    raise RuntimeError(
                        "Server error (HTTP 500) — the server may be overloaded. Try again later."
                    )
                raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
            return resp.json()
        except (RuntimeError, PostprocessingError):
            raise  # don't retry on 4xx/5xx with a clear message
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")


def decode_result_images(raw: dict) -> dict[str, Image.Image]:
    out = {}
    for key, b64 in raw.get("images", {}).items():
        out[key] = Image.open(io.BytesIO(base64.b64decode(b64)))
    return out


def extract_scoring(raw: dict) -> tuple[dict, list[str], bool]:
    """Return (scoring_dict, all_non_image_keys_from_raw, key_found).
    Tries several key names the DeepLIIF API might use."""
    non_image_keys = [k for k in raw if k != "images"]
    for key in ("scoring", "scores", "cell_scoring", "score", "results"):
        if key in raw:
            sc = raw[key]
            return (sc if isinstance(sc, dict) else {}), non_image_keys, True
    return {}, non_image_keys, False


def run_local_scoring(
    orig_img: Image.Image,
    decoded: dict,
    resolution: str,
) -> tuple[dict, dict]:
    """Run DeepLIIF postprocessing locally to count positive/negative cells.

    Uses the Seg + Marker images returned by the API together with the
    original (prepared) image.  Returns:
        scoring  – dict with num_total, num_pos, num_neg, percent_pos, …
        extras   – dict of additional PIL images (Overlay, Refined)
    """
    if not _DEEPLIIF_PP:
        return {}, {}

    # Accept both capitalisation styles ("Seg" / "seg")
    seg = decoded.get("Seg") or decoded.get("seg")
    if seg is None:
        return {}, {}

    marker = decoded.get("Marker") or decoded.get("marker")

    try:
        overlay, refined, scoring = _deepliif_compute(
            orig=orig_img.convert("RGB"),
            seg=seg.convert("RGB"),
            marker=marker.convert("RGB") if marker is not None else None,
            resolution=resolution,
        )
        # compute_final_results returns numpy arrays — convert to PIL
        def _to_pil(x):
            if isinstance(x, Image.Image):
                return x
            arr = np.array(x)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        extras = {"Overlay": _to_pil(overlay), "Refined": _to_pil(refined)}
        return scoring, extras
    except Exception:
        return {}, {}


def run_video_frame_scoring(
    frame_img: Image.Image,
    decoded: dict,
    resolution: str,
) -> tuple[dict, list]:
    """Like run_local_scoring but also returns per-cell centroid data for
    cross-frame deduplication.  Returns (scoring_dict, cells_list) where
    each cell is {centroid: (y, x), positive: bool}."""
    if not _DEEPLIIF_PP:
        return {}, []

    seg = decoded.get("Seg") or decoded.get("seg")
    if seg is None:
        return {}, []

    marker = decoded.get("Marker") or decoded.get("marker")

    try:
        seg_rgb    = seg.convert("RGB")
        marker_rgb = marker.convert("RGB") if marker is not None else None

        # Detect cells once; reuse data for both per-cell list and scoring
        cell_data = _deepliif_cell_results(seg_rgb, marker_rgb, resolution)
        cells = [
            {"centroid": c["centroid"], "positive": bool(c["positive"])}
            for c in cell_data.get("cells", [])
        ]

        # Derive aggregate scoring without re-running detection
        _, _, scoring = _deepliif_cells_to_final(
            data=cell_data,
            orig=frame_img.convert("RGB"),
        )
        return scoring, cells
    except Exception:
        return {}, []


def deduplicate_video_cells(all_results: dict, dedup_radius: int = 20) -> dict:
    """Deduplicate cells across overlapping video frames (e.g. WSI scans).

    Algorithm:
      1. For each frame, estimate the shift from the previous frame using
         phase correlation on the Seg image.
      2. Convert every cell centroid to a global coordinate system using
         the cumulative shift.
      3. Insert into a spatial grid; skip any cell whose global position
         is within `dedup_radius` pixels of an already-inserted cell.

    Returns a scoring dict with num_total, num_pos, num_neg, percent_pos.
    """
    grid: dict = {}   # (grid_x, grid_y) -> [(gx, gy, is_positive), ...]
    cumulative = np.array([0.0, 0.0])   # (dx, dy) in global coords
    prev_gray  = None
    raw_total  = 0

    for res in all_results.values():
        cells = res.get("_cells", [])
        if not cells:
            continue

        # ── Estimate shift from previous frame ────────────────────────────
        seg = res["images"].get("Seg") or res["images"].get("seg")
        if seg is not None:
            curr_gray = np.array(seg.convert("L"), dtype=np.float32)
            if prev_gray is not None and prev_gray.shape == curr_gray.shape:
                try:
                    shift, _ = cv2.phaseCorrelate(prev_gray, curr_gray)
                    # phaseCorrelate(prev, curr) returns (dx, dy) such that
                    # prev shifted by (dx, dy) ≈ curr.
                    # Global coord = frame_coord - cumulative_shift
                    cumulative += np.array(shift)
                except Exception:
                    pass
            prev_gray = curr_gray

        # ── Insert cells into global grid ─────────────────────────────────
        for cell in cells:
            raw_total += 1
            cy, cx = cell["centroid"]          # (row, col) = (y, x)
            gx = cx - cumulative[0]
            gy = cy - cumulative[1]
            gix = int(gx // dedup_radius)
            giy = int(gy // dedup_radius)

            duplicate = False
            for dix in range(-1, 2):
                for diy in range(-1, 2):
                    for ex, ey, _ in grid.get((gix + dix, giy + diy), []):
                        if np.hypot(gx - ex, gy - ey) < dedup_radius:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if duplicate:
                    break

            if not duplicate:
                key = (gix, giy)
                grid.setdefault(key, []).append((gx, gy, cell["positive"]))

    unique = [c for bucket in grid.values() for c in bucket]
    n_total = len(unique)
    n_pos   = sum(1 for c in unique if c[2])
    n_neg   = n_total - n_pos
    pct     = round(n_pos / n_total * 100, 2) if n_total > 0 else 0.0

    return {
        "num_total":   n_total,
        "num_pos":     n_pos,
        "num_neg":     n_neg,
        "percent_pos": pct,
        "pos_neg_ratio": f"{n_pos} : {n_neg}",
        "raw_detections": raw_total,
        "note": (
            f"Deduplicated from {raw_total} raw detections across "
            f"{len(all_results)} frames using phase-correlation shift "
            f"estimation (dedup radius = {dedup_radius} px)."
        ),
    }


def interpolate_frames(frames: list, steps: int) -> list:
    """Insert `steps` linearly blended frames between every consecutive pair."""
    if steps == 0 or len(frames) < 2:
        return frames
    out = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        a = np.array(frames[i].convert("RGB"), dtype=float)
        b = np.array(frames[i + 1].convert("RGB"), dtype=float)
        for s in range(1, steps + 1):
            t = s / (steps + 1)
            out.append(Image.fromarray(((1 - t) * a + t * b).astype(np.uint8)))
    out.append(frames[-1])
    return out


def frames_to_video(pil_frames: list, fps: float) -> bytes:
    """Stitch a list of PIL images into an MP4 and return the bytes."""
    if not pil_frames:
        return b""
    w, h = pil_frames[0].size
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        for img in pil_frames:
            writer.write(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))
        writer.release()
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


def build_zip(all_results: dict) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, res in all_results.items():
            for ch, img in res["images"].items():
                ib = io.BytesIO()
                img.save(ib, format="PNG")
                zf.writestr(f"{name}/{ch}.png", ib.getvalue())
            zf.writestr(
                f"{name}/scoring.json",
                json.dumps(res["scoring"], indent=2),
            )
    buf.seek(0)
    return buf


# ─── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DeepLIIF Pipeline", layout="wide")
st.title("DeepLIIF Pipeline")
st.caption("IHC image quantification via DeepLIIF · supports video & multi-page TIF")

# --- Sidebar: options ---------------------------------------------------------
with st.sidebar:
    st.header("DeepLIIF Options")

    resolution = st.selectbox(
        "Scan Resolution",
        ["40x", "20x", "10x"],
        help="Resolution at which the slide was scanned",
    )
    prob_thresh = st.slider(
        "Probability Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Minimum probability for a pixel to count as positive cell",
    )
    slim = st.checkbox("Slim mode", help="Return only the segmentation image")
    # Apply deferred nopost flag set during PostprocessingError recovery
    if st.session_state.get("_request_nopost"):
        st.session_state["nopost"] = True
        del st.session_state["_request_nopost"]
    if "nopost" not in st.session_state:
        st.session_state["nopost"] = False
    nopost = st.checkbox("Skip postprocessing", key="nopost")
    use_pil = st.checkbox(
        "Pillow loader (faster, PNG/JPG only)",
        help="Uncheck to use Bio-Formats for TIF/WSI",
    )


# --- Main: upload + format picker --------------------------------------------
uploaded = st.file_uploader(
    "Upload file",
    type=ALL_FORMATS,
    help="Video or image file — TIF / video are the priority formats",
)

if uploaded:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()

    # Video-only options
    every_n_sec = 1.0
    out_fps = 5
    interp_steps = 1
    api_delay = 1.0
    if ext in VIDEO_FORMATS:
        every_n_sec = st.slider(
            "Extract one frame every N seconds", 0.5, 10.0, 1.0, 0.5,
            help="Lower = more frames extracted = smoother output (but more API calls)",
        )
        out_fps = st.slider(
            "Output video FPS", 1, 30, 5,
            help="Playback speed of the stitched output videos",
        )
        interp_steps = st.slider(
            "Smoothing interpolation steps", 0, 4, 1,
            help="Blended frames inserted between each processed frame — makes motion smoother without extra API calls",
        )
        api_delay = st.slider(
            "Delay between API calls (s)", 0.0, 5.0, 1.0, 0.5,
            help="Pause between frames to avoid overloading the DeepLIIF server",
        )

    # Build API param dict — match official API spec exactly
    # Only 'resolution' is always required; all others are opt-in flags
    api_params: dict = {"resolution": resolution}
    if slim:
        api_params["slim"] = "true"
    if nopost:
        api_params["nopost"] = "true"
    else:
        # prob_thresh only applies when postprocessing is active (API expects 0–254)
        api_params["prob_thresh"] = int(prob_thresh * 254)
    if use_pil:
        api_params["pil"] = "true"

    if st.button("Run DeepLIIF", type="primary"):
        # Quick server reachability check
        try:
            ping = requests.get("https://deepliif.org", timeout=10)
            if ping.status_code >= 500:
                st.error("deepliif.org is returning server errors right now. Try again later.")
                st.stop()
        except Exception:
            st.error("Cannot reach deepliif.org. Check your internet connection.")
            st.stop()

        raw_bytes = uploaded.read()
        all_results: dict = {}
        error_occurred = False

        # ── Video ──────────────────────────────────────────────────────────
        if ext in VIDEO_FORMATS:
            with st.spinner("Extracting frames…"):
                frames = extract_video_frames(raw_bytes, ext, every_n_sec)
            st.info(f"{len(frames)} frame(s) extracted")
            bar = st.progress(0, text="Processing frames…")

            for i, frame_arr in enumerate(frames):
                if i > 0 and api_delay > 0:
                    time.sleep(api_delay)
                fname = f"frame_{i:04d}"
                try:
                    frame_img, notes = prepare_image(Image.fromarray(frame_arr), resolution)
                    if notes:
                        st.caption(f"{fname}: {', '.join(notes)}")
                    png = image_to_png_bytes(frame_img)
                    raw = call_api(png, f"{fname}.png", api_params)
                    sc, raw_keys, sc_key_found = extract_scoring(raw)
                    decoded_imgs = decode_result_images(raw)
                    local_sc, frame_cells = run_video_frame_scoring(frame_img, decoded_imgs, resolution)
                    if local_sc:
                        sc = local_sc
                        sc_key_found = True
                    all_results[fname] = {
                        "images": decoded_imgs,
                        "scoring": sc,
                        "_cells": frame_cells,
                        "_raw_keys": raw_keys,
                        "_sc_key_found": sc_key_found,
                    }
                except PostprocessingError as e:
                    st.warning(str(e))
                    st.session_state["_request_nopost"] = True
                    st.rerun()
                except Exception as e:
                    st.warning(f"{fname}: {e}")
                    error_occurred = True
                bar.progress((i + 1) / len(frames), text=f"Frame {i+1}/{len(frames)}")

        # ── TIF (multi-page) ───────────────────────────────────────────────
        elif ext in ("tif", "tiff"):
            with st.spinner("Reading TIF pages…"):
                pages = tif_pages(raw_bytes)
            st.info(f"{len(pages)} page(s) found")
            bar = st.progress(0, text="Processing pages…")

            for i, page_img in enumerate(pages):
                fname = f"page_{i:04d}"
                try:
                    page_img, notes = prepare_image(page_img, resolution)
                    if notes:
                        st.caption(f"{fname}: {', '.join(notes)}")
                    png = image_to_png_bytes(page_img)
                    raw = call_api(png, f"{fname}.png", api_params)
                    sc, raw_keys, sc_key_found = extract_scoring(raw)
                    decoded_imgs = decode_result_images(raw)
                    local_sc, extras = run_local_scoring(page_img, decoded_imgs, resolution)
                    if local_sc:
                        sc = local_sc
                        sc_key_found = True
                        decoded_imgs.update(extras)
                    all_results[fname] = {
                        "images": decoded_imgs,
                        "scoring": sc,
                        "_raw_keys": raw_keys,
                        "_sc_key_found": sc_key_found,
                    }
                except PostprocessingError as e:
                    st.warning(str(e))
                    st.session_state["_request_nopost"] = True
                    st.rerun()
                except Exception as e:
                    st.warning(f"{fname}: {e}")
                    error_occurred = True
                bar.progress((i + 1) / len(pages), text=f"Page {i+1}/{len(pages)}")

        # ── Standard image ─────────────────────────────────────────────────
        else:
            with st.spinner("Processing…"):
                try:
                    src_img = Image.open(io.BytesIO(raw_bytes))
                    src_img, notes = prepare_image(src_img, resolution)
                    if notes:
                        st.caption(f"image: {', '.join(notes)}")
                    raw_bytes = image_to_png_bytes(src_img)
                    raw = call_api(raw_bytes, uploaded.name, api_params)
                    sc, raw_keys, sc_key_found = extract_scoring(raw)
                    decoded_imgs = decode_result_images(raw)
                    local_sc, extras = run_local_scoring(src_img, decoded_imgs, resolution)
                    if local_sc:
                        sc = local_sc
                        sc_key_found = True
                        decoded_imgs.update(extras)
                    all_results["result"] = {
                        "images": decoded_imgs,
                        "scoring": sc,
                        "_raw_keys": raw_keys,
                        "_sc_key_found": sc_key_found,
                    }
                except PostprocessingError as e:
                    st.warning(str(e))
                    st.session_state["_request_nopost"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"API error: {e}")
                    error_occurred = True

        # ── Pre-stitch all channel videos and store everything in session state ──
        if all_results:
            video_bytes_map: dict = {}
            if ext in VIDEO_FORMATS:
                channels = list(next(iter(all_results.values()))["images"].keys())
                stitch_bar = st.progress(0, text="Stitching output videos…")
                for i, ch in enumerate(channels):
                    ch_frames = [
                        res["images"][ch]
                        for res in all_results.values()
                        if ch in res["images"]
                    ]
                    if ch_frames:
                        with st.spinner(f"Stitching {ch}…"):
                            smooth_frames = interpolate_frames(ch_frames, interp_steps)
                            vid_bytes = frames_to_video(smooth_frames, out_fps)
                        video_bytes_map[ch] = {
                            "bytes": vid_bytes,
                            "n_processed": len(ch_frames),
                            "n_smooth": len(smooth_frames),
                        }
                    stitch_bar.progress((i + 1) / len(channels), text=f"Stitched {i+1}/{len(channels)} channels")

            # ── Deduplicate cells across frames (WSI overlap-aware) ────────
            deduped_scoring: dict = {}
            if ext in VIDEO_FORMATS and all_results:
                with st.spinner("Deduplicating cells across frames…"):
                    deduped_scoring = deduplicate_video_cells(all_results)

            st.session_state["_results"] = {
                "all_results": all_results,
                "ext": ext,
                "error_occurred": error_occurred,
                "video_bytes_map": video_bytes_map,
                "interp_steps": interp_steps if ext in VIDEO_FORMATS else 0,
                "deduped_scoring": deduped_scoring,
            }

# ── Results display — lives outside the button block so it survives
#    download-button reruns without losing data ─────────────────────────────
if "_results" in st.session_state:
    saved = st.session_state["_results"]
    _all     = saved["all_results"]
    _ext     = saved["ext"]
    _err     = saved["error_occurred"]
    _vmap    = saved["video_bytes_map"]
    _isteps  = saved["interp_steps"]
    _deduped = saved.get("deduped_scoring", {})

    st.success(
        f"Done — {len(_all)} item(s) processed" + (" (some errors)" if _err else "")
    )

    if st.button("Process New Sample", type="secondary"):
        del st.session_state["_results"]
        st.rerun()

    if _ext in VIDEO_FORMATS:
        # ── All-channels zip — MP4s + deduped scoring JSON ──────────────────
        all_zip = io.BytesIO()
        with zipfile.ZipFile(all_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for ch, vdata in _vmap.items():
                zf.writestr(f"{ch}.mp4", vdata["bytes"])
            if _deduped:
                zf.writestr("scoring_summary.json", json.dumps(_deduped, indent=2))
        all_zip.seek(0)
        st.download_button(
            "Download all channels (.zip)",
            all_zip,
            file_name="deepliif_all_channels.zip",
            mime="application/zip",
            key="dl_all_zip",
        )

        # ── Deduplicated cell-count summary ─────────────────────────────────
        if _deduped and _deduped.get("num_total", 0) > 0:
            st.subheader("Cell counts (deduplicated across frames)")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Cells",      _deduped["num_total"])
            m2.metric("Positive Cells",   _deduped["num_pos"])
            m3.metric("Negative Cells",   _deduped["num_neg"])
            m4.metric("% Positive",       f"{_deduped['percent_pos']}%")
            m5.metric("Pos : Neg",        _deduped["pos_neg_ratio"])
            st.caption(
                f"Raw detections before deduplication: "
                f"{_deduped.get('raw_detections', '—')}. "
                + _deduped.get("note", "")
            )
        else:
            # Fallback: simple sum if dedup produced nothing
            frame_scores = [r["scoring"] for r in _all.values() if r.get("scoring")]
            if frame_scores and "num_total" in (frame_scores[0] or {}):
                agg_total = sum(s.get("num_total", 0) for s in frame_scores)
                agg_pos   = sum(s.get("num_pos",   0) for s in frame_scores)
                agg_neg   = sum(s.get("num_neg",   0) for s in frame_scores)
                agg_pct   = round(agg_pos / agg_total * 100, 2) if agg_total > 0 else 0.0
                st.subheader("Cell counts (sum across all frames)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Cells",    agg_total)
                m2.metric("Positive Cells", agg_pos)
                m3.metric("Negative Cells", agg_neg)
                m4.metric("% Positive",     f"{agg_pct}%")

        st.subheader("Output channels")
        for ch, vdata in _vmap.items():
            col1, col2 = st.columns([4, 1])
            col1.write(
                f"**{ch}** — {vdata['n_processed']} processed frames"
                + (f" → {vdata['n_smooth']} after interpolation" if _isteps else "")
            )
            col2.download_button(
                f"Download {ch}.mp4",
                vdata["bytes"],
                file_name=f"{ch}.mp4",
                mime="video/mp4",
                key=f"dl_vid_{ch}",
            )

        zip_buf = build_zip(_all)
        st.download_button(
            label="Download all frames (.zip)",
            data=zip_buf,
            file_name="deepliif_frames.zip",
            mime="application/zip",
            key="dl_frames",
        )

    else:
        # ── Image / TIF expander view ───────────────────────────────────────
        for name, res in _all.items():
            with st.expander(name, expanded=(len(_all) == 1)):
                sc = res["scoring"]
                if sc and "num_total" in sc:
                    # Local deepliif scoring (full detail)
                    pct = sc.get("percent_pos", 0)
                    pct_str = f"{round(pct, 2)}%" if isinstance(pct, (int, float)) else str(pct)
                    ratio = (
                        f"{sc.get('num_pos', 0)} : {sc.get('num_neg', 0)}"
                        if sc.get("num_total", 0) > 0 else "—"
                    )
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Cells",    sc.get("num_total", "—"))
                    m2.metric("Positive Cells", sc.get("num_pos",   "—"))
                    m3.metric("Negative Cells", sc.get("num_neg",   "—"))
                    m4.metric("% Positive",     pct_str)
                    m5.metric("Pos : Neg",       ratio)
                elif sc:
                    # Fallback: API-level scoring (older key names)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Nuclei",   sc.get("total_nuclei",    sc.get("num_total", "—")))
                    c2.metric("Positive Cells", sc.get("positive_cells",  sc.get("num_pos",   "—")))
                    c3.metric("% Positive",     sc.get("percent_positive",sc.get("percent_pos","—")))
                else:
                    if res.get("_sc_key_found"):
                        st.warning(
                            "The API returned a `scoring` key but it was empty — "
                            "the server may not have produced cell counts for this image "
                            "(check that postprocessing is enabled and the image meets size requirements)."
                        )
                    else:
                        raw_keys = res.get("_raw_keys", [])
                        st.warning(
                            f"No scoring key found in the API response. "
                            f"Top-level keys returned: `{raw_keys}`."
                        )

                imgs = res["images"]
                if imgs:
                    cols = st.columns(len(imgs))
                    for col, (ch, img) in zip(cols, imgs.items()):
                        col.image(img, caption=ch, use_container_width=True)

        zip_buf = build_zip(_all)
        st.download_button(
            label="Download all results (.zip)",
            data=zip_buf,
            file_name="deepliif_results.zip",
            mime="application/zip",
        )
