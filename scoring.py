"""IHC Scoring Module

Computes cell counts and clinical IHC scores from DeepLIIF output images.

Inputs expected:
    seg_img   – PIL Image: DeepLIIF "Seg" channel
                  red pixels  (R>150, G<100, B<100) → positive nuclei
                  blue pixels (B>150, R<100, G<100) → negative nuclei
    marker_img – PIL Image: DeepLIIF "Marker" channel (separated DAB signal)
                  bright pixels → high DAB staining (strongly positive)

The per-nucleus intensity sampling is adapted from Morpho's
NucleiFeatureExtractor._extract_intensity_features, using
skimage.measure.regionprops with an intensity_image argument.

Scores returned:
    KI67  → num_total, num_pos, num_neg, percent_pos, cells
    ER/PR → + h_score (0-300), allred_score (0-8),
              allred_proportion (0-5), allred_intensity (0-3),
              intensity_distribution {0,1,2,3}
"""

import numpy as np
from PIL import Image
from skimage import measure

# ─── Minimum nucleus area in pixels (noise filter) ───────────────────────────
_MIN_CELL_AREA = 15


# ─── Cell detection from Seg image ───────────────────────────────────────────

def _find_cells(seg_arr: np.ndarray):
    """
    Parse DeepLIIF Seg image into positive and negative cell masks.
    Returns: (pos_mask, neg_mask) as boolean numpy arrays.
    """
    r, g, b = seg_arr[:, :, 0], seg_arr[:, :, 1], seg_arr[:, :, 2]
    pos_mask = (r > 150) & (g < 100) & (b < 100)
    neg_mask = (b > 150) & (r < 100) & (g < 100)
    return pos_mask, neg_mask


def _label_cells(mask: np.ndarray, intensity_img: np.ndarray | None = None):
    """
    Label connected components in mask, filter by minimum area.
    Adapted from Morpho's regionprops-based intensity sampling.
    Returns: list of regionprops objects.
    """
    labeled = measure.label(mask)
    if intensity_img is not None:
        props = measure.regionprops(labeled, intensity_image=intensity_img)
    else:
        props = measure.regionprops(labeled)
    return [p for p in props if p.area >= _MIN_CELL_AREA]


def _max_intensity(prop) -> float:
    """Return per-region max intensity to preserve DAB probability peaks."""
    if hasattr(prop, "intensity_max"):
        return float(prop.intensity_max)
    return float(prop.max_intensity)


# ─── Intensity classification for H-score / Allred ───────────────────────────

def _intensity_grade(max_intensity: float) -> int:
    """
    Classify max DAB intensity (0-255, bright = high staining) into:
        1  → weak  (1+)
        2  → moderate (2+)
        3  → strong (3+)
    Note: We only grade cells that are already classified as Positive.
    So the minimum grade is 1+.
    """
    if max_intensity < 50:
        return 1
    elif max_intensity < 100:
        return 2
    else:
        return 3


# ─── Allred scoring helpers ───────────────────────────────────────────────────

def _allred_proportion_score(pct_pos: float) -> int:
    """
    Convert % positive cells to Allred proportion score (0-5).
    Standard Allred thresholds used in clinical practice.
    """
    if pct_pos == 0:
        return 0
    elif pct_pos <= 1:
        return 1
    elif pct_pos <= 10:
        return 2
    elif pct_pos <= 33:
        return 3
    elif pct_pos <= 66:
        return 4
    else:
        return 5


def _allred_intensity_score(mean_grade: float) -> int:
    """Convert mean intensity grade (0-3) to Allred intensity score (0-3)."""
    if mean_grade < 0.5:
        return 0
    elif mean_grade < 1.5:
        return 1
    elif mean_grade < 2.5:
        return 2
    else:
        return 3


# ─── Clinical interpretation helpers ─────────────────────────────────────────

def allred_interpretation(score: int) -> str:
    if score <= 2:
        return "Negative"
    elif score <= 4:
        return "Weakly Positive"
    elif score <= 6:
        return "Moderately Positive"
    else:
        return "Strongly Positive"


def h_score_interpretation(h: int) -> str:
    if h <= 100:
        return "Low"
    elif h <= 200:
        return "Intermediate"
    else:
        return "High"


# ─── Public scoring functions ─────────────────────────────────────────────────

def compute_ki67_score(seg_img: Image.Image, marker_img: Image.Image | None = None) -> dict:
    """
    KI67 scoring: positive / negative cell counts and proliferation index.

    Returns dict with keys:
        num_total, num_pos, num_neg, percent_pos, cells
    """
    seg_arr = np.array(seg_img.convert("RGB"))
    pos_mask, neg_mask = _find_cells(seg_arr)

    pos_props = _label_cells(pos_mask)
    neg_props = _label_cells(neg_mask)

    n_pos = len(pos_props)
    n_neg = len(neg_props)
    n_total = n_pos + n_neg
    pct = round(n_pos / n_total * 100, 1) if n_total > 0 else 0.0

    cells = (
        [{"centroid": (float(p.centroid[0]), float(p.centroid[1])), "positive": True}  for p in pos_props] +
        [{"centroid": (float(p.centroid[0]), float(p.centroid[1])), "positive": False} for p in neg_props]
    )

    return {
        "num_total":   n_total,
        "num_pos":     n_pos,
        "num_neg":     n_neg,
        "percent_pos": pct,
        "cells":       cells,
    }


def compute_erpr_score(seg_img: Image.Image, marker_img: Image.Image) -> dict:
    """
    ER / PR scoring: positive/negative counts + H-score + Allred score.

    H-score  = 1×(%1+) + 2×(%2+) + 3×(%3+)  → 0-300
    Allred   = proportion_score + intensity_score → 0-8

    The per-nucleus mean intensity is sampled from the Marker (DAB) channel
    using skimage.measure.regionprops, following the same approach as Morpho's
    NucleiFeatureExtractor._extract_intensity_features.

    Returns dict with all keys from compute_ki67_score plus:
        h_score, allred_score, allred_proportion, allred_intensity,
        intensity_distribution, h_score_label, allred_label
    """
    seg_arr    = np.array(seg_img.convert("RGB"))
    marker_rgb = np.array(marker_img.convert("RGB"))
    marker_arr = np.max(marker_rgb, axis=2)   # max over channels to preserve DAB probability

    pos_mask, neg_mask = _find_cells(seg_arr)

    # ── Per-nucleus intensity sampling (Morpho pattern) ───────────────────────
    pos_props = _label_cells(pos_mask, marker_arr)   # has .mean_intensity
    neg_props = _label_cells(neg_mask)

    n_pos   = len(pos_props)
    n_neg   = len(neg_props)
    n_total = n_pos + n_neg

    if n_total == 0:
        return {
            "num_total": 0, "num_pos": 0, "num_neg": 0,
            "percent_pos": 0.0,
            "h_score": 0, "h_score_label": "Low",
            "allred_score": 0, "allred_label": "Negative",
            "allred_proportion": 0, "allred_intensity": 0,
            "intensity_distribution": {0: 0, 1: 0, 2: 0, 3: 0},
            "cells": [],
        }

    pct_pos = round(n_pos / n_total * 100, 1)

    # ── Intensity grading per positive nucleus ────────────────────────────────
    grades = [_intensity_grade(_max_intensity(p)) for p in pos_props]
    grade_counts = {0: n_neg, 1: 0, 2: 0, 3: 0}
    for g in grades:
        grade_counts[g] += 1

    # ── H-score: weighted sum relative to ALL cells ───────────────────────────
    h_score = round(
        (1 * grade_counts[1] + 2 * grade_counts[2] + 3 * grade_counts[3])
        / n_total * 100
    )

    # ── Allred ────────────────────────────────────────────────────────────────
    ps        = _allred_proportion_score(pct_pos)
    if n_pos > 0:
        is_ = max([1, 2, 3], key=lambda k: grade_counts[k])
    else:
        is_ = 0
    allred    = ps + is_

    cells = (
        [{"centroid": (float(p.centroid[0]), float(p.centroid[1])), "positive": True}  for p in pos_props] +
        [{"centroid": (float(p.centroid[0]), float(p.centroid[1])), "positive": False} for p in neg_props]
    )

    return {
        "num_total":              n_total,
        "num_pos":                n_pos,
        "num_neg":                n_neg,
        "percent_pos":            pct_pos,
        "h_score":                h_score,
        "h_score_label":          h_score_interpretation(h_score),
        "allred_score":           allred,
        "allred_label":           allred_interpretation(allred),
        "allred_proportion":      ps,
        "allred_intensity":       is_,
        "intensity_distribution": grade_counts,
        "cells":                  cells,
    }


def score_image(seg_img: Image.Image, marker_img: Image.Image | None, stain: str) -> dict:
    """
    Dispatch to the correct scorer based on stain type.
    stain: "KI67" | "ER" | "PR"
    """
    if stain in ("ER", "PR") and marker_img is not None:
        return compute_erpr_score(seg_img, marker_img)
    return compute_ki67_score(seg_img, marker_img)

def compute_global_erpr_score(total_dist: dict, dedup_pos: int, dedup_neg: int) -> dict:
    """
    Computes global H-score and Allred score over deduplicated cells,
    given the aggregated intensity distribution from all frames.
    """
    pos_total_raw = total_dist.get(1, 0) + total_dist.get(2, 0) + total_dist.get(3, 0)
    
    if pos_total_raw > 0:
        prop_1 = total_dist.get(1, 0) / pos_total_raw
        prop_2 = total_dist.get(2, 0) / pos_total_raw
        prop_3 = total_dist.get(3, 0) / pos_total_raw
    else:
        prop_1 = prop_2 = prop_3 = 0.0

    dedup_dist = {
        0: dedup_neg,
        1: int(round(dedup_pos * prop_1)),
        2: int(round(dedup_pos * prop_2)),
        3: int(round(dedup_pos * prop_3)),
    }
    
    # fix rounding errors to ensure they sum to dedup_pos
    diff = dedup_pos - (dedup_dist[1] + dedup_dist[2] + dedup_dist[3])
    if diff != 0 and dedup_pos > 0:
        max_k = max([1, 2, 3], key=lambda k: dedup_dist[k])
        dedup_dist[max_k] += diff
        
    dedup_total = dedup_pos + dedup_neg
    if dedup_total > 0:
        h_score = int(round(
            (1 * dedup_dist[1] + 2 * dedup_dist[2] + 3 * dedup_dist[3])
            / dedup_total * 100
        ))
        pct_pos = round(dedup_pos / dedup_total * 100, 1)
    else:
        h_score = 0
        pct_pos = 0.0
        
    ps = _allred_proportion_score(pct_pos)
    if dedup_pos > 0:
        is_ = max([1, 2, 3], key=lambda k: dedup_dist[k])
    else:
        is_ = 0
    allred = ps + is_
    
    return {
        "h_score": h_score,
        "h_score_label": h_score_interpretation(h_score),
        "allred_score": allred,
        "allred_label": allred_interpretation(allred),
        "allred_proportion": ps,
        "allred_intensity": is_,
        "intensity_distribution": dedup_dist,
    }
