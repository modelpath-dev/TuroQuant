"""
TuroQuant  —  IHC Quantification
==================================
Two source modes:
  • File Upload  — full pipeline: image / multi-page TIF / video with all channels
  • Camera       — browser-based live camera (st.camera_input) + ROI + batch capture
"""

import io
import json
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import tifffile
import streamlit as st
from PIL import Image

from scoring import score_image, compute_global_erpr_score
from segmentation import (
    PostprocessingError,
    check_server,
    segment,
    segment_video_frame,
)


# ─── Constants ───────────────────────────────────────────────────────────────

IMAGE_FORMATS = [
    "png", "jpg", "jpeg",
    "tif", "tiff",
    "bmp", "gif",
    "svs", "ndpi", "scn",
    "czi", "lif", "mrxs",
    "vms", "vmu", "qptiff",
]
VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]
ALL_FORMATS   = IMAGE_FORMATS + VIDEO_FORMATS
_STAINS       = ["KI67", "ER", "PR"]


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TuroQuant Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Session state ───────────────────────────────────────────────────────────
for k, v in {
    "cam_results":   None,
    "batch_frames":  [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — file-upload pipeline (original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_video_frames(video_bytes: bytes, ext: str, every_n_sec: float) -> list:
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(fps * every_n_sec))
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


def tif_pages(tif_bytes: bytes) -> list:
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
    if not pil_frames:
        return b""
    w, h = pil_frames[0].size
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        writer = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h),
        )
        for img in pil_frames:
            writer.write(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))
        writer.release()
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


def deduplicate_video_cells(all_results: dict, dedup_radius: int = 20) -> dict:
    grid: dict = {}
    cumulative = np.array([0.0, 0.0])
    prev_gray  = None
    raw_total  = 0
    for res in all_results.values():
        cells = res.get("_cells", [])
        seg   = res["images"].get("Seg") or res["images"].get("seg")
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
            gix, giy = int(gx // dedup_radius), int(gy // dedup_radius)
            dup = False
            for dix in range(-1, 2):
                for diy in range(-1, 2):
                    for ex, ey, _ in grid.get((gix + dix, giy + diy), []):
                        if np.hypot(gx - ex, gy - ey) < dedup_radius:
                            dup = True; break
                    if dup: break
                if dup: break
            if not dup:
                grid.setdefault((gix, giy), []).append((gx, gy, cell["positive"]))
    unique  = [c for bucket in grid.values() for c in bucket]
    n_total = len(unique)
    n_pos   = sum(1 for c in unique if c[2])
    n_neg   = n_total - n_pos
    pct     = round(n_pos / n_total * 100, 2) if n_total > 0 else 0.0
    return {
        "num_total":      n_total,
        "num_pos":        n_pos,
        "num_neg":        n_neg,
        "percent_pos":    pct,
        "pos_neg_ratio":  f"{n_pos} : {n_neg}",
        "raw_detections": raw_total,
        "note": (
            f"Deduplicated from {raw_total} raw detections across "
            f"{len(all_results)} frames (dedup radius = {dedup_radius} px)."
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
            zf.writestr(f"{name}/scoring.json", json.dumps(res["scoring"], indent=2))
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — camera / analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _to_png_bytes(img) -> bytes:
    """PIL Image → PNG bytes (for st.image inline embedding)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _build_overlay(orig: Image.Image, seg: Image.Image) -> Image.Image:
    orig_arr = np.array(orig.convert("RGB"), dtype=np.float32)
    seg_arr  = np.array(seg.convert("RGB"))
    r, g, b  = seg_arr[:, :, 0], seg_arr[:, :, 1], seg_arr[:, :, 2]
    pos_mask = (r > 150) & (g < 100) & (b < 100)
    neg_mask = (b > 150) & (r < 100) & (g < 100)
    out = orig_arr * 0.45
    out[pos_mask] = [235, 60,  55]
    out[neg_mask] = [50,  110, 230]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _run_analysis(pil_img: Image.Image, stain: str, resolution: str,
                  prob_thresh: float, nopost: bool):
    """Run DeepLIIF + scoring. Returns result dict or None on error."""
    try:
        result = segment(pil_img, resolution=resolution, prob_thresh=prob_thresh,
                         slim=False, nopost=nopost, use_pil=False)
    except PostprocessingError:
        st.session_state["nopost"] = True
        try:
            result = segment(pil_img, resolution=resolution, prob_thresh=prob_thresh,
                             slim=False, nopost=True, use_pil=False)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return None
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

    seg_img    = result["images"].get("Seg") or result["images"].get("seg")
    marker_img = result["images"].get("Marker") or result["images"].get("marker")
    if seg_img is None:
        st.error("DeepLIIF did not return a Seg image.")
        return None

    return {
        "scoring": score_image(seg_img, marker_img, stain),
        "overlay": _build_overlay(pil_img, seg_img),
        "notes":   result["notes"],
    }


def _cam_deduplicate(frame_results: list, dedup_radius: int = 20) -> dict:
    grid: dict = {}
    raw_total  = 0
    for fr in frame_results:
        for cell in fr.get("cells", []):
            raw_total += 1
            cy, cx = cell["centroid"]
            gix, giy = int(cx // dedup_radius), int(cy // dedup_radius)
            dup = False
            for dix in range(-1, 2):
                for diy in range(-1, 2):
                    for ex, ey, _ in grid.get((gix + dix, giy + diy), []):
                        if np.hypot(cx - ex, cy - ey) < dedup_radius:
                            dup = True; break
                    if dup: break
                if dup: break
            if not dup:
                grid.setdefault((gix, giy), []).append((cx, cy, cell["positive"]))
    unique  = [c for bucket in grid.values() for c in bucket]
    n_total = len(unique)
    n_pos   = sum(1 for c in unique if c[2])
    n_neg   = n_total - n_pos
    pct     = round(n_pos / n_total * 100, 1) if n_total > 0 else 0.0
    return {"num_total": n_total, "num_pos": n_pos, "num_neg": n_neg,
            "percent_pos": pct, "raw_detections": raw_total}


def _show_scoring(scoring: dict, stain: str, n_frames: int = 1, is_batch: bool = False):
    """Render metric columns for any stain type."""
    n   = scoring.get("num_total",  "—")
    pos = scoring.get("num_pos",    "—")
    neg = scoring.get("num_neg",    "—")
    pct = scoring.get("percent_pos", 0)

    if is_batch:
        st.info(
            f"**Final index: frame {n_frames} of {n_frames}** — "
            f"deduplicated from {scoring.get('raw_detections','—')} raw detections"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cells",   n)
    m2.metric("Positive",      pos)
    m3.metric("Negative",      neg)
    m4.metric("% Positive",    f"{pct:.1f}%" if isinstance(pct, float) else f"{pct}%")

    if stain in ("ER", "PR") and "h_score" in scoring:
        h    = scoring["h_score"]
        al   = scoring["allred_score"]
        ps   = scoring["allred_proportion"]
        is_  = scoring["allred_intensity"]
        hl   = scoring.get("h_score_label", "")
        al_l = scoring.get("allred_label", "")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"H-Score ({hl})", h)
        e2.metric(f"Allred ({al_l})", f"{al}/8")
        e3.metric("Proportion", f"{ps}/5")
        e4.metric("Intensity",  f"{is_}/3")
        dist = scoring.get("intensity_distribution", {})
        if any(dist.values()):
            st.caption(
                f"Intensity distribution — "
                f"Neg: {dist.get(0,0)}  |  "
                f"1+: {dist.get(1,0)}  |  "
                f"2+: {dist.get(2,0)}  |  "
                f"3+: {dist.get(3,0)}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("TuroQuant Options")

    source_mode = st.radio("Source", ["File Upload", "Camera"], horizontal=True)

    st.markdown("**Stain**")
    stain = st.radio("Stain type", _STAINS, horizontal=True,
                     label_visibility="collapsed")
    dedup_radius = 20
    if source_mode == "Camera":
        dedup_radius = st.slider("Dedup radius (px)", 5, 50, 20)

    st.markdown("---")
    st.markdown("**Acquisition**")
    resolution  = st.selectbox("Scan Resolution", ["40x", "20x", "10x"])
    prob_thresh = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.05)
    slim        = st.checkbox("Slim mode")

    if st.session_state.get("_request_nopost"):
        st.session_state["nopost"] = True
        del st.session_state["_request_nopost"]
    if "nopost" not in st.session_state:
        st.session_state["nopost"] = True   # default ON — server postprocessing often returns HTTP 500
    nopost  = st.checkbox("Skip postprocessing", key="nopost")
    use_pil = st.checkbox("Pillow loader (faster, PNG/JPG only)")

    st.markdown("---")
    if st.button("Check server", use_container_width=True):
        with st.spinner("Checking…"):
            ok = check_server()
        st.success("Server reachable") if ok else st.error("Server unreachable")


def _seg_kwargs() -> dict:
    return dict(resolution=resolution, prob_thresh=prob_thresh,
                slim=slim, nopost=nopost, use_pil=use_pil)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD MODE  —  original pipeline
# ═══════════════════════════════════════════════════════════════════════════════

if source_mode == "File Upload":
    st.title("TuroQuant Pipeline")
    st.caption("IHC quantification via TuroQuant · supports video, multi-page TIF, and standard images")

    uploaded = st.file_uploader("Upload file", type=ALL_FORMATS)

    if uploaded:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()

        every_n_sec  = 1.0
        out_fps      = 5
        interp_steps = 1
        max_workers  = 4

        if ext in VIDEO_FORMATS:
            every_n_sec  = st.slider("Extract one frame every N seconds", 0.5, 10.0, 1.0, 0.5)
            out_fps      = st.slider("Output video FPS", 1, 30, 5)
            interp_steps = st.slider("Smoothing interpolation steps", 0, 4, 1)
            max_workers  = st.slider("Parallel workers", 1, 8, 4)

        if st.button("Run TuroQuant", type="primary"):
            if not check_server():
                st.error("Cannot reach TuroQuant server. Try again later.")
                st.stop()

            raw_bytes      = uploaded.read()
            all_results    = {}
            error_occurred = False
            kwargs         = _seg_kwargs()
            final_index    = 0   # track last frame index

            # ── Video ────────────────────────────────────────────────────────
            if ext in VIDEO_FORMATS:
                with st.spinner("Extracting frames…"):
                    frames = extract_video_frames(raw_bytes, ext, every_n_sec)
                final_index = len(frames)
                st.info(f"{len(frames)} frame(s) extracted — processing with {max_workers} workers  |  **Final index: {final_index}**")
                bar        = st.progress(0, text="Processing frames…")
                done_count = 0

                def _process_frame(i, frame_arr):
                    fname  = f"frame_{i:04d}"
                    result = segment_video_frame(Image.fromarray(frame_arr), **kwargs)
                    _seg_i = result["images"].get("Seg") or result["images"].get("seg")
                    _mrk_i = result["images"].get("Marker") or result["images"].get("marker")
                    if _seg_i and not result.get("cells"):
                        _sc = score_image(_seg_i, _mrk_i, stain)
                        result["scoring"] = _sc
                        result["cells"] = _sc.get("cells", [])
                    return i, fname, result

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_process_frame, i, arr): i
                               for i, arr in enumerate(frames)}
                    for future in as_completed(futures):
                        done_count += 1
                        try:
                            i, fname, result = future.result()
                            if result["notes"]:
                                st.caption(f"{fname}: {', '.join(result['notes'])}")
                            all_results[fname] = {
                                "images":  result["images"],
                                "scoring": result["scoring"],
                                "_cells":  result["cells"],
                            }
                        except PostprocessingError as e:
                            st.warning(str(e))
                            st.session_state["_request_nopost"] = True
                            st.rerun()
                        except Exception as e:
                            st.warning(f"frame_{futures[future]:04d}: {e}")
                            error_occurred = True
                        bar.progress(done_count / len(frames),
                                     text=f"Frame {done_count} of {len(frames)}  (final index: {final_index})")

                all_results = dict(sorted(all_results.items()))

            # ── TIF ──────────────────────────────────────────────────────────
            elif ext in ("tif", "tiff"):
                with st.spinner("Reading TIF pages…"):
                    pages = tif_pages(raw_bytes)
                final_index = len(pages)
                st.info(f"{len(pages)} page(s) found  |  **Final index: {final_index}**")
                bar = st.progress(0, text="Processing pages…")
                for i, page_img in enumerate(pages):
                    fname = f"page_{i:04d}"
                    try:
                        result = segment(page_img, **kwargs)
                        if result["notes"]:
                            st.caption(f"{fname}: {', '.join(result['notes'])}")
                        _seg_i = result["images"].get("Seg") or result["images"].get("seg")
                        _mrk_i = result["images"].get("Marker") or result["images"].get("marker")
                        _sc = score_image(_seg_i, _mrk_i, stain) if _seg_i else result["scoring"]
                        all_results[fname] = {"images": result["images"], "scoring": _sc}
                    except PostprocessingError as e:
                        st.warning(str(e)); st.session_state["_request_nopost"] = True; st.rerun()
                    except Exception as e:
                        st.warning(f"{fname}: {e}"); error_occurred = True
                    bar.progress((i + 1) / len(pages),
                                 text=f"Page {i+1} of {len(pages)}  (final index: {final_index})")

            # ── Image ────────────────────────────────────────────────────────
            else:
                final_index = 1
                with st.spinner("Processing…"):
                    try:
                        src_img = Image.open(io.BytesIO(raw_bytes))
                        result  = segment(src_img, **kwargs)
                        if result["notes"]:
                            st.caption(f"image: {', '.join(result['notes'])}")
                        _seg_i = result["images"].get("Seg") or result["images"].get("seg")
                        _mrk_i = result["images"].get("Marker") or result["images"].get("marker")
                        _sc = score_image(_seg_i, _mrk_i, stain) if _seg_i else result["scoring"]
                        all_results["result"] = {"images": result["images"], "scoring": _sc}
                    except PostprocessingError as e:
                        st.warning(str(e)); st.session_state["_request_nopost"] = True; st.rerun()
                    except Exception as e:
                        st.error(f"API error: {e}"); error_occurred = True

            # ── Stitch + dedup + store ────────────────────────────────────
            if all_results:
                video_bytes_map: dict = {}
                if ext in VIDEO_FORMATS:
                    channels   = list(next(iter(all_results.values()))["images"].keys())
                    stitch_bar = st.progress(0, text="Stitching output videos…")
                    for i, ch in enumerate(channels):
                        ch_frames = [res["images"][ch] for res in all_results.values()
                                     if ch in res["images"]]
                        if ch_frames:
                            smooth = interpolate_frames(ch_frames, interp_steps)
                            video_bytes_map[ch] = {
                                "bytes":       frames_to_video(smooth, out_fps),
                                "n_processed": len(ch_frames),
                                "n_smooth":    len(smooth),
                            }
                        stitch_bar.progress((i + 1) / len(channels))

                deduped_scoring: dict = {}
                if ext in VIDEO_FORMATS:
                    with st.spinner("Deduplicating cells…"):
                        deduped_scoring = deduplicate_video_cells(all_results)
                        if stain in ("ER", "PR") and all_results:
                            total_dist = {0: 0, 1: 0, 2: 0, 3: 0}
                            for res in all_results.values():
                                dist = res.get("scoring", {}).get("intensity_distribution", {})
                                for k, v in dist.items():
                                    total_dist[int(k)] += v
                            global_scores = compute_global_erpr_score(
                                total_dist,
                                deduped_scoring.get("num_pos", 0),
                                deduped_scoring.get("num_neg", 0)
                            )
                            deduped_scoring.update(global_scores)

                st.session_state["_results"] = {
                    "all_results":    all_results,
                    "ext":            ext,
                    "error_occurred": error_occurred,
                    "video_bytes_map": video_bytes_map,
                    "interp_steps":   interp_steps if ext in VIDEO_FORMATS else 0,
                    "deduped_scoring": deduped_scoring,
                    "final_index":    final_index,
                }

    # ── Results display ───────────────────────────────────────────────────────
    if "_results" in st.session_state:
        saved    = st.session_state["_results"]
        _all     = saved["all_results"]
        _ext     = saved["ext"]
        _err     = saved["error_occurred"]
        _vmap    = saved["video_bytes_map"]
        _isteps  = saved["interp_steps"]
        _deduped = saved.get("deduped_scoring", {})
        _fidx    = saved.get("final_index", len(_all))

        st.success(
            f"Done — {len(_all)} item(s) processed  |  **Final index: {_fidx}**"
            + ("  (some errors)" if _err else "")
        )

        if st.button("Process New Sample", type="secondary"):
            del st.session_state["_results"]
            st.rerun()

        if _ext in VIDEO_FORMATS:
            all_zip = io.BytesIO()
            with zipfile.ZipFile(all_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for ch, vdata in _vmap.items():
                    zf.writestr(f"{ch}.mp4", vdata["bytes"])
                if _deduped:
                    zf.writestr("scoring_summary.json", json.dumps(_deduped, indent=2))
            all_zip.seek(0)
            st.download_button("Download all channels (.zip)", all_zip,
                               file_name="turoquant_all_channels.zip",
                               mime="application/zip", key="dl_all_zip")

            if _deduped and _deduped.get("num_total", 0) > 0:
                st.subheader(f"Deduplicated cell counts  (final index: frame {_fidx})")
                _show_scoring(_deduped, stain, n_frames=_fidx, is_batch=True)
                st.caption(_deduped.get("note", ""))

            st.subheader("Output channels")
            for ch, vdata in _vmap.items():
                col1, col2 = st.columns([4, 1])
                col1.write(
                    f"**{ch}** — {vdata['n_processed']} frames"
                    + (f" → {vdata['n_smooth']} after interpolation" if _isteps else "")
                )
                col2.download_button(f"Download {ch}.mp4", vdata["bytes"],
                                     file_name=f"{ch}.mp4", mime="video/mp4",
                                     key=f"dl_vid_{ch}")

            zip_buf = build_zip(_all)
            st.download_button("Download all frames (.zip)", zip_buf,
                               file_name="turoquant_frames.zip",
                               mime="application/zip", key="dl_frames")

        else:
            for name, res in _all.items():
                with st.expander(name, expanded=(len(_all) == 1)):
                    sc = res["scoring"]
                    if sc and "num_total" in sc:
                        _show_scoring(sc, stain)
                    elif sc:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Nuclei",   sc.get("total_nuclei",   sc.get("num_total","—")))
                        c2.metric("Positive Cells", sc.get("positive_cells", sc.get("num_pos",  "—")))
                        c3.metric("% Positive",
                                  sc.get("percent_positive", sc.get("percent_pos", "—")))
                    else:
                        st.warning("No scoring data returned — check postprocessing is enabled.")

                    imgs = res["images"]
                    if imgs:
                        cols = st.columns(len(imgs))
                        for col, (ch, img) in zip(cols, imgs.items()):
                            col.image(img, caption=ch, use_container_width=True)

            zip_buf = build_zip(_all)
            st.download_button("Download all results (.zip)", zip_buf,
                               file_name="turoquant_results.zip", mime="application/zip")


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA MODE  —  browser-based via st.camera_input (no cv2 camera required)
# ═══════════════════════════════════════════════════════════════════════════════

elif source_mode == "Camera":
    st.title(f"TuroQuant — Camera  ({stain})")
    st.caption(
        "The camera preview below uses your browser's built-in camera access. "
        "Click the **snapshot** button (camera icon) to capture a frame."
    )

    # ── Live capture ──────────────────────────────────────────────────────────
    captured = st.camera_input("Point at sample — click 📷 to capture")

    if captured is not None:
        pil_img = Image.open(captured).convert("RGB")
        img_arr = np.array(pil_img)
        orig_h, orig_w = img_arr.shape[:2]

        st.markdown("---")

        # ── ROI selection ─────────────────────────────────────────────────────
        use_roi = st.checkbox("Select ROI (region of interest)")
        if use_roi:
            sc1, sc2 = st.columns(2)
            with sc1:
                x1 = st.slider("Left",   0, orig_w - 2, orig_w // 4, key="roi_x1")
                y1 = st.slider("Top",    0, orig_h - 2, orig_h // 4, key="roi_y1")
            with sc2:
                x2 = st.slider("Right",  x1 + 1, orig_w,
                                max(x1 + 1, orig_w * 3 // 4), key="roi_x2")
                y2 = st.slider("Bottom", y1 + 1, orig_h,
                                max(y1 + 1, orig_h * 3 // 4), key="roi_y2")
            preview = img_arr.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (99, 102, 241), 3)
            st.image(preview, use_container_width=True,
                     caption=f"ROI: ({x1},{y1}) → ({x2},{y2})")
            analyse_img = pil_img.crop((x1, y1, x2, y2))
        else:
            analyse_img = pil_img

        st.markdown("---")

        # ── Action buttons ────────────────────────────────────────────────────
        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("▶ Analyse This Frame", type="primary",
                         use_container_width=True):
                with st.spinner("Sending to DeepLIIF…"):
                    res = _run_analysis(analyse_img, stain, resolution,
                                        prob_thresh, nopost)
                if res:
                    st.session_state.cam_results = {
                        "scoring":   res["scoring"],
                        "overlay":   res["overlay"],
                        "notes":     res["notes"],
                        "n_frames":  1,
                        "is_batch":  False,
                    }
                    st.rerun()

        with b2:
            if st.button("➕ Add to Batch", use_container_width=True):
                st.session_state.batch_frames.append(analyse_img.copy())
                st.success(f"Added — batch has {len(st.session_state.batch_frames)} frame(s)")

        with b3:
            batch = st.session_state.batch_frames
            label = f"Process Batch ({len(batch)})" if batch else "Batch empty"
            if batch and st.button(label, use_container_width=True):
                final_idx  = len(batch)
                prog       = st.progress(0, text=f"Processing batch…  (final index: {final_idx})")
                frame_results = []
                for i, fr in enumerate(batch):
                    r = _run_analysis(fr, stain, resolution, prob_thresh, nopost)
                    if r:
                        frame_results.append({
                            "scoring": r["scoring"],
                            "overlay": r["overlay"],
                            "cells":   r["scoring"].get("cells", []),
                        })
                    prog.progress((i + 1) / final_idx,
                                  text=f"Frame {i+1} of {final_idx}  (final index: {final_idx})")
                if frame_results:
                    deduped = _cam_deduplicate(frame_results, dedup_radius)
                    if stain in ("ER", "PR"):
                        total_dist = {0: 0, 1: 0, 2: 0, 3: 0}
                        for fr in frame_results:
                            dist = fr.get("scoring", {}).get("intensity_distribution", {})
                            for k, v in dist.items():
                                total_dist[int(k)] += v
                        global_scores = compute_global_erpr_score(
                            total_dist,
                            deduped.get("num_pos", 0),
                            deduped.get("num_neg", 0)
                        )
                        deduped.update(global_scores)
                    st.session_state.cam_results = {
                        "scoring":  deduped,
                        "overlay":  frame_results[-1]["overlay"],
                        "notes":    [],
                        "n_frames": final_idx,
                        "is_batch": True,
                    }
                    st.session_state.batch_frames = []
                    st.rerun()

    # Clear batch
    if st.session_state.batch_frames:
        if st.button(f"🗑 Clear batch  ({len(st.session_state.batch_frames)} frame(s))"):
            st.session_state.batch_frames = []
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    res = st.session_state.cam_results
    if res is not None:
        st.markdown("---")
        scoring  = res["scoring"]
        overlay  = res["overlay"]
        n_frames = res["n_frames"]
        is_batch = res["is_batch"]

        col_ov, col_mt = st.columns([55, 45])
        with col_ov:
            title = (f"Batch Result — final index: frame {n_frames}"
                     if is_batch else "Analysis Result")
            st.subheader(title)
            if overlay is not None:
                st.image(_to_png_bytes(overlay), use_container_width=True)
            st.caption("🔴 Positive cells   🔵 Negative cells")

        with col_mt:
            st.subheader(f"{stain} Scoring")
            _show_scoring(scoring, stain, n_frames=n_frames, is_batch=is_batch)

            if overlay is not None:
                # Build scoring JSON (exclude raw cell list to keep it readable)
                json_bytes = json.dumps(
                    {k: v for k, v in scoring.items() if k != "cells"}, indent=2
                ).encode()

                # Combined ZIP: overlay PNG + scoring JSON
                result_zip = io.BytesIO()
                with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    png_buf = io.BytesIO()
                    overlay.save(png_buf, format="PNG")
                    zf.writestr(f"turoquant_{stain.lower()}_overlay.png", png_buf.getvalue())
                    zf.writestr(f"turoquant_{stain.lower()}_scoring.json", json_bytes)
                result_zip.seek(0)

                d1, d2, d3 = st.columns(3)
                with d1:
                    png_buf2 = io.BytesIO()
                    overlay.save(png_buf2, format="PNG"); png_buf2.seek(0)
                    st.download_button(
                        "⬇ Overlay PNG", png_buf2,
                        file_name=f"turoquant_{stain.lower()}_overlay.png",
                        mime="image/png", use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "⬇ Scoring JSON", json_bytes,
                        file_name=f"turoquant_{stain.lower()}_scoring.json",
                        mime="application/json", use_container_width=True,
                    )
                with d3:
                    st.download_button(
                        "⬇ Results ZIP", result_zip,
                        file_name=f"turoquant_{stain.lower()}_results.zip",
                        mime="application/zip", use_container_width=True,
                    )

        notes = res.get("notes", [])
        if notes:
            for n in notes:
                st.caption(f"ℹ️ {n}")

        if st.button("✕ Clear results"):
            st.session_state.cam_results = None
            st.rerun()
