# Segmentation Module — How to Swap in Your Own Model

This guide explains how to replace the default DeepLIIF API with your own segmentation logic.
You only need to edit **one file**: `segmentation.py`. Nothing in `app.py` needs to change.

---

## How it works right now

```
Your image  -->  segmentation.py  -->  { images, scoring, notes }
                      |
                      |  (currently)
                      v
               DeepLIIF remote API   (neural network runs on their server)
                      |
                      v
               Local deepliif lib    (cell counting runs on your machine)
```

You are replacing everything inside `segmentation.py` with your own pipeline.

---

## What you need to keep (the contract)

`app.py` imports exactly **4 things** from `segmentation.py`. These must always exist:

### 1. `segment(img, resolution, ...)`

Called for **single images and TIF pages**.

**Input:**
| Parameter     | Type          | What it is                                  |
|---------------|---------------|---------------------------------------------|
| `img`         | `PIL.Image`   | The raw input image (any size, any mode)     |
| `resolution`  | `str`         | Scan resolution: `"40x"`, `"20x"`, or `"10x"` |
| `prob_thresh` | `float`       | 0.0 to 1.0 — sensitivity for positive cells |
| `slim`        | `bool`        | If True, only return segmentation mask       |
| `nopost`      | `bool`        | If True, skip server-side postprocessing     |
| `use_pil`     | `bool`        | Loader preference (you can ignore this)      |

**Output:** a dict with these keys:
```python
{
    "images": {
        "Seg": PIL.Image,        # segmentation mask (required)
        "Marker": PIL.Image,     # marker channel (optional)
        "Overlay": PIL.Image,    # overlay visualization (optional)
        # ... add any other channels you want to display
    },
    "scoring": {
        "num_total": 310,        # total cells detected
        "num_pos": 251,          # positive cells
        "num_neg": 59,           # negative cells
        "percent_pos": 81.0,     # percentage positive
        # ... you can add extra keys, app.py only reads the above four
    },
    "notes": [
        "scaled down to 2048x2048",  # warnings/info strings (can be empty list)
    ],
}
```

### 2. `segment_video_frame(img, resolution, ...)`

Called for **video frames**. Same input as `segment()`.

**Output:** same as `segment()` but with one extra key — `"cells"`:
```python
{
    "images": { ... },       # same as above
    "scoring": { ... },      # same as above
    "notes": [ ... ],        # same as above
    "cells": [
        {"centroid": (120, 340), "positive": True},
        {"centroid": (200, 410), "positive": False},
        # centroid is (y, x) — row, column in pixels
        # positive is True/False — is this cell stained positive?
    ],
}
```

The `cells` list is used by `app.py` to deduplicate cells across overlapping video frames.
If you don't need video support, just return `"cells": []`.

### 3. `check_server()`

Returns `True` if your backend is ready, `False` if not.
If you run a local model, you can just `return True`.

### 4. `PostprocessingError`

An exception class. Raise it when something fails in a recoverable way.
`app.py` catches this and shows a warning to the user.

```python
class PostprocessingError(RuntimeError):
    pass
```

Keep this class even if you never raise it — `app.py` imports it.

---

## Step-by-step: replacing with your own model

### Step 1 — Delete the internals

Open `segmentation.py`. Everything with a `_` prefix is internal. Delete all of it:
- `_API_URL`, `_MAX_DIM`, `_MIN_DIM`
- `_prepare()`, `_to_png()`, `_call_api()`, `_decode_images()`
- `_extract_scoring()`, `_build_params()`, `_to_pil()`
- `_local_scoring()`, `_local_cell_scoring()`
- The `deepliif` imports at the top

### Step 2 — Add your own imports

```python
import torch  # or tensorflow, or whatever your model uses
from PIL import Image

# Keep these — app.py needs them
class PostprocessingError(RuntimeError):
    pass
```

### Step 3 — Load your model

```python
# Load once when the module is imported
_model = torch.load("my_model.pth")
_model.eval()
```

### Step 4 — Write your segment() function

```python
def segment(
    img: Image.Image,
    resolution: str = "40x",
    prob_thresh: float = 0.5,
    slim: bool = False,
    nopost: bool = False,
    use_pil: bool = False,
) -> dict:
    # 1. Preprocess your image however you need
    tensor = your_preprocess(img)

    # 2. Run your model
    with torch.no_grad():
        output = _model(tensor)

    # 3. Convert output to PIL images
    seg_mask = to_pil(output["seg"])

    # 4. Count cells however you want
    total, pos, neg = your_cell_counter(output, prob_thresh)

    # 5. Return the dict — keep this shape!
    return {
        "images": {
            "Seg": seg_mask,
            # add more channels if you have them
        },
        "scoring": {
            "num_total": total,
            "num_pos": pos,
            "num_neg": neg,
            "percent_pos": round(pos / total * 100, 2) if total > 0 else 0.0,
        },
        "notes": [],
    }
```

### Step 5 — Write segment_video_frame() (if you need video)

Same as `segment()` but also return per-cell positions:

```python
def segment_video_frame(
    img: Image.Image,
    resolution: str = "40x",
    prob_thresh: float = 0.5,
    slim: bool = False,
    nopost: bool = False,
    use_pil: bool = False,
) -> dict:
    result = segment(img, resolution, prob_thresh, slim, nopost, use_pil)

    # Add cell positions for cross-frame deduplication
    cells = []
    for cell in your_cell_detector(img):
        cells.append({
            "centroid": (cell.y, cell.x),    # (row, col) in pixels
            "positive": cell.is_positive,     # True or False
        })

    result["cells"] = cells
    return result
```

If you don't care about video, just do:

```python
def segment_video_frame(img, **kwargs):
    result = segment(img, **kwargs)
    result["cells"] = []
    return result
```

### Step 6 — Update check_server()

```python
def check_server() -> bool:
    # Local model? Always ready:
    return True

    # Remote API? Check it:
    # try:
    #     resp = requests.get("https://your-api.com/health", timeout=5)
    #     return resp.ok
    # except:
    #     return False
```

---

## Quick reference — what app.py reads from your output

| Key                    | Where it's used                              | Required? |
|------------------------|----------------------------------------------|-----------|
| `images["Seg"]`       | Displayed as a channel, used in video dedup  | Yes       |
| `images[any_name]`    | Each key becomes a column in the UI          | Optional  |
| `scoring["num_total"]`| Displayed as "Total Cells" metric            | Yes       |
| `scoring["num_pos"]`  | Displayed as "Positive Cells" metric         | Yes       |
| `scoring["num_neg"]`  | Displayed as "Negative Cells" metric         | Yes       |
| `scoring["percent_pos"]` | Displayed as "% Positive" metric          | Yes       |
| `notes`               | Shown as captions under each processed item  | Yes (can be `[]`) |
| `cells[].centroid`    | Used for cross-frame deduplication (video)   | Only for video |
| `cells[].positive`    | Used for pos/neg counting after dedup        | Only for video |

---

## Common mistakes to avoid

1. **Don't rename the functions.** `app.py` imports `segment`, `segment_video_frame`, `check_server`, and `PostprocessingError` by name.

2. **Don't change the return keys.** `app.py` reads `result["images"]`, `result["scoring"]`, `result["notes"]`, and `result["cells"]`. Missing keys will crash the app.

3. **Don't forget `"notes"`.** Even if you have nothing to say, return `"notes": []`.

4. **Images must be PIL.Image objects.** Not numpy arrays, not file paths. Use `Image.fromarray(arr)` to convert.

5. **Scoring values should be numbers**, not strings. `"num_total": 310` not `"num_total": "310"`.

6. **Centroids are (y, x)** — row first, column second. This matches numpy array indexing.

---

## Testing your changes

After editing `segmentation.py`, run the test script:

```bash
python test.py
```

It should print a scoring dict. If it works, the Streamlit app will work too:

```bash
streamlit run app.py
```
