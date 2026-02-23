import streamlit as st
import io
import json
import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import tifffile
from PIL import Image

# ─── Segmentation API — the only import from segmentation.py ─────────────────
from segmentation import (
    segment,
    segment_video_frame,
    check_server,
    PostprocessingError,
)


# ─── Constants ───────────────────────────────────────────────────────────────

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


def deduplicate_video_cells(all_results: dict, dedup_radius: int = 20) -> dict:
    """Deduplicate cells across overlapping video frames (e.g. WSI scans)."""
    grid: dict = {}
    cumulative = np.array([0.0, 0.0])
    prev_gray  = None
    raw_total  = 0

    for res in all_results.values():
        cells = res.get("_cells", [])
        if not cells:
            continue

        seg = res["images"].get("Seg") or res["images"].get("seg")
        if seg is not None:
            curr_gray = np.array(seg.convert("L"), dtype=np.float32)
            if prev_gray is not None and prev_gray.shape == curr_gray.shape:
                try:
                    shift, _ = cv2.phaseCorrelate(prev_gray, curr_gray)
                    cumulative += np.array(shift)
                except Exception:
                    pass
            prev_gray = curr_gray

        for cell in cells:
            raw_total += 1
            cy, cx = cell["centroid"]
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


# ─── Segmentation config shared across all code paths ────────────────────────

def _seg_kwargs() -> dict:
    """Collect the current sidebar settings into kwargs for segment()."""
    return dict(
        resolution=resolution,
        prob_thresh=prob_thresh,
        slim=slim,
        nopost=nopost,
        use_pil=use_pil,
    )


# ─── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="TuroQuant Pipeline", layout="wide")
st.title("TuroQuant Pipeline")
st.caption("IHC image quantification via TuroQuant · supports video & multi-page TIF")

# --- Sidebar: options ---------------------------------------------------------
with st.sidebar:
    st.header("TuroQuant Options")

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
    max_workers = 4
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
        max_workers = st.slider(
            "Parallel workers", 1, 8, 4,
            help="Number of frames sent to the API at the same time. Higher = faster but may overload the server.",
        )

    if st.button("Run TuroQuant", type="primary"):
        # Quick server reachability check
        if not check_server():
            st.error("Cannot reach TuroQuant server or it is returning errors. Try again later.")
            st.stop()

        raw_bytes = uploaded.read()
        all_results: dict = {}
        error_occurred = False
        kwargs = _seg_kwargs()

        # ── Video (parallel) ───────────────────────────────────────────────
        if ext in VIDEO_FORMATS:
            with st.spinner("Extracting frames…"):
                frames = extract_video_frames(raw_bytes, ext, every_n_sec)
            st.info(f"{len(frames)} frame(s) extracted — processing with {max_workers} parallel workers")
            bar = st.progress(0, text="Processing frames…")
            done_count = 0

            def _process_frame(i, frame_arr):
                fname = f"frame_{i:04d}"
                result = segment_video_frame(Image.fromarray(frame_arr), **kwargs)
                return i, fname, result

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_process_frame, i, arr): i
                    for i, arr in enumerate(frames)
                }
                for future in as_completed(futures):
                    done_count += 1
                    try:
                        i, fname, result = future.result()
                        if result["notes"]:
                            st.caption(f"{fname}: {', '.join(result['notes'])}")
                        all_results[fname] = {
                            "images": result["images"],
                            "scoring": result["scoring"],
                            "_cells": result["cells"],
                        }
                    except PostprocessingError as e:
                        st.warning(str(e))
                        st.session_state["_request_nopost"] = True
                        st.rerun()
                    except Exception as e:
                        st.warning(f"frame_{futures[future]:04d}: {e}")
                        error_occurred = True
                    bar.progress(done_count / len(frames), text=f"Frame {done_count}/{len(frames)}")

            # Sort results by frame order (as_completed returns in finish order)
            all_results = dict(sorted(all_results.items()))

        # ── TIF (multi-page) ───────────────────────────────────────────────
        elif ext in ("tif", "tiff"):
            with st.spinner("Reading TIF pages…"):
                pages = tif_pages(raw_bytes)
            st.info(f"{len(pages)} page(s) found")
            bar = st.progress(0, text="Processing pages…")

            for i, page_img in enumerate(pages):
                fname = f"page_{i:04d}"
                try:
                    result = segment(page_img, **kwargs)
                    if result["notes"]:
                        st.caption(f"{fname}: {', '.join(result['notes'])}")
                    all_results[fname] = {
                        "images": result["images"],
                        "scoring": result["scoring"],
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
                    result = segment(src_img, **kwargs)
                    if result["notes"]:
                        st.caption(f"image: {', '.join(result['notes'])}")
                    all_results["result"] = {
                        "images": result["images"],
                        "scoring": result["scoring"],
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
            file_name="turoquant_all_channels.zip",
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
            file_name="turoquant_frames.zip",
            mime="application/zip",
            key="dl_frames",
        )

    else:
        # ── Image / TIF expander view ───────────────────────────────────────
        for name, res in _all.items():
            with st.expander(name, expanded=(len(_all) == 1)):
                sc = res["scoring"]
                if sc and "num_total" in sc:
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
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Nuclei",   sc.get("total_nuclei",    sc.get("num_total", "—")))
                    c2.metric("Positive Cells", sc.get("positive_cells",  sc.get("num_pos",   "—")))
                    c3.metric("% Positive",     sc.get("percent_positive",sc.get("percent_pos","—")))
                else:
                    st.warning(
                        "No scoring data returned — the server may not have produced "
                        "cell counts for this image (check that postprocessing is enabled "
                        "and the image meets size requirements)."
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
            file_name="turoquant_results.zip",
            mime="application/zip",
        )
