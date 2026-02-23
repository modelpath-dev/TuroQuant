# Video Cell Deduplication — How It Works

When you scan a whole-slide image by recording a video, consecutive frames overlap.
The same cell appears in multiple frames. Without deduplication, it gets counted 2-5 times.

This document explains how the pipeline handles that.

---

## The problem

Imagine a slide with 300 cells. You record a video panning across it.
Each frame sees ~50 cells, and frames overlap by ~60%.

```
Frame 1:  [---- 50 cells ----]
Frame 2:       [---- 50 cells ----]       ← 30 cells shared with frame 1
Frame 3:            [---- 50 cells ----]  ← 30 cells shared with frame 2
...
```

Naive counting: 20 frames x 50 cells = 1000 (wrong)
Actual unique cells: ~300 (correct)

---

## The solution (3 steps)

### Step 1 — Get cell positions from each frame

Each frame goes through `segment_video_frame()` in `segmentation.py`.
It returns a list of cells with their pixel positions:

```python
"cells": [
    {"centroid": (120, 340), "positive": True},
    {"centroid": (200, 410), "positive": False},
    ...
]
```

`centroid` is `(y, x)` — the row and column of the cell center in that frame's image.

### Step 2 — Figure out how much the camera moved

The problem: cell at `(120, 340)` in frame 1 and cell at `(120, 290)` in frame 2
might be the **same cell** — the camera just moved 50 pixels to the right.

We estimate the camera shift using **phase correlation** (`cv2.phaseCorrelate`).
This compares the Seg (segmentation) images of consecutive frames and returns
how many pixels the image shifted in x and y.

```
Frame 1 Seg image  →  phaseCorrelate  →  shift = (50, 0)
Frame 2 Seg image  ↗                     (camera moved 50px right)
```

We accumulate these shifts to build a running total:

```
Frame 1: cumulative shift = (0, 0)
Frame 2: cumulative shift = (50, 0)       ← moved 50px right
Frame 3: cumulative shift = (100, 0)      ← moved another 50px right
Frame 4: cumulative shift = (100, -30)    ← also moved 30px down
```

### Step 3 — Convert to global coordinates and deduplicate

Each cell's position gets adjusted by the cumulative shift to get **global coordinates**:

```
Frame 1: cell at (120, 340) → global (120, 340)       # shift (0, 0)
Frame 2: cell at (120, 290) → global (120, 340)       # shift (50, 0) → 290 + 50 = 340
                                                        # Same cell!
```

Now we insert cells into a **spatial grid**. Before adding a cell, we check
if any existing cell is within 20 pixels (the `dedup_radius`). If yes, it's a duplicate — skip it.

```
Cell A at global (120, 340) → added to grid
Cell B at global (120, 342) → distance = 2px < 20px → DUPLICATE, skipped
Cell C at global (300, 500) → no neighbor nearby → added to grid
```

---

## Where this happens in the code

| Step | What | Where |
|------|------|-------|
| Cell detection | Each frame → list of cell centroids | `segmentation.py` → `segment_video_frame()` |
| Parallel processing | Send frames to API simultaneously | `app.py` → `ThreadPoolExecutor` (line ~333) |
| Shift estimation | `cv2.phaseCorrelate` on Seg images | `app.py` → `deduplicate_video_cells()` (line ~168) |
| Global coordinates | Adjust centroids by cumulative shift | Same function |
| Grid deduplication | Skip cells within 20px of existing | Same function |

---

## The dedup_radius parameter

Default is **20 pixels**. This means:
- Two cells within 20px of each other (in global coords) are considered the same cell
- Only the first one seen is kept

**Too small** (e.g. 5px): might not catch duplicates if the shift estimation is slightly off
**Too large** (e.g. 50px): might merge two genuinely different cells that are close together

20px works well for most IHC slides at 40x resolution.

To change it, edit the call in `app.py`:
```python
deduped_scoring = deduplicate_video_cells(all_results, dedup_radius=20)
```

---

## Limitations

1. **Translation only.** Phase correlation assumes the camera moves linearly (left/right/up/down).
   If the video has rotation or zoom, the shift estimation will be inaccurate.

2. **Depends on Seg image quality.** If the segmentation is poor or very different between frames,
   `phaseCorrelate` may return wrong shifts.

3. **Fixed radius.** The 20px radius doesn't adapt to cell size. For very large or very small cells,
   you may want to adjust it.

4. **No tracking.** Cells are not tracked across frames — they're just deduplicated by position.
   If a cell moves (e.g. in a live imaging video), it would be counted as separate cells.

---

## What the output looks like

After deduplication, you get:

```python
{
    "num_total": 312,                  # unique cells
    "num_pos": 248,                    # unique positive cells
    "num_neg": 64,                     # unique negative cells
    "percent_pos": 79.49,
    "pos_neg_ratio": "248 : 64",
    "raw_detections": 1547,            # total before dedup (many duplicates)
    "note": "Deduplicated from 1547 raw detections across 20 frames ..."
}
```

In the UI this shows as "Cell counts (deduplicated across frames)" with both
the final count and the raw detection count for comparison.
