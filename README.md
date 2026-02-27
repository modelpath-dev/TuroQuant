# TuroQuant Pipeline

IHC (Immunohistochemistry) image quantification using the DeepLIIF API.
Supports **KI67**, **ER**, and **PR** stain types with automated cell counting,
H-score, and Allred scoring.

---

## Features

### File Upload mode
- Upload **images** (PNG, JPG, BMP), **multi-page TIF/TIFF**, or **video** (MP4, AVI, MOV, MKV, WebM)
- All DeepLIIF output channels displayed (DAPI, Hema, Lap2, Marker, Seg)
- Cell counting: Total / Positive / Negative / % Positive
- Video: parallel frame processing + phase-correlation deduplication across overlapping frames
- **Final index** shown at every step (frame N of N)
- Download: individual channel MP4s, all-frames ZIP, scoring JSON

### Camera mode
- **Browser-based** live camera preview (uses WebRTC built into your browser — no driver issues)
- Click the snapshot button to capture a frame
- **ROI selection**: define a crop region with sliders, run only that region through DeepLIIF
- **Batch capture**: add multiple frames, process as a deduplicated batch
- ER/PR stain: shows **H-score** (0-300) and **Allred score** (0-8) in addition to counts

---

## Stain Types

| Stain | Metrics |
|-------|---------|
| KI67  | Total cells, Positive, Negative, % Positive (proliferation index) |
| ER    | All KI67 metrics + H-score, Allred score, intensity distribution (0/1+/2+/3+) |
| PR    | Same as ER |

---

## Supported File Formats

| Type | Formats |
|------|---------|
| Image | PNG, JPG/JPEG, BMP, GIF |
| Microscopy | TIF/TIFF (multi-page), SVS, NDPI, SCN, CZI, LIF, MRXS, VMS, VMU, QPTIFF |
| Video | MP4, AVI, MOV, MKV, WebM |

---

## Setup

### Requirements
- Python 3.10+
- Internet connection (DeepLIIF API at `https://deepliif.org/api/infer`)

### Install

```bash
git clone https://github.com/modelpath-dev/Deep-Liif.git
cd TuroQuant

python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** Do NOT run `pip install deepliif` — it is incompatible with Python 3.10+ and is not needed. All local scoring is handled by `scoring.py`.

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Project Structure

```
TuroQuant/
  app.py           # Streamlit application (File Upload + Camera modes)
  segmentation.py  # DeepLIIF API client (swap backend here if needed)
  scoring.py       # H-score, Allred, KI67 scoring from Seg + Marker images
  requirements.txt # Python dependencies
  README.md        # This file
```

---

## Sidebar Options

| Option | Description |
|--------|-------------|
| Source | File Upload (full pipeline) or Camera (live browser capture) |
| Stain | KI67 / ER / PR - controls which scoring metrics are shown |
| Scan Resolution | 10x / 20x / 40x - must match your microscope setting |
| Probability Threshold | Minimum confidence for cell classification |
| Skip postprocessing | **Leave ON** - server postprocessing returns HTTP 500 |
| Pillow loader | Faster for PNG/JPG; disable for TIF/WSI formats |

---

## How Cell Counting Works

### Single Image / TIF
The app sends the image to the DeepLIIF API which returns channel-decomposed images. Local scoring uses the **Seg** (segmentation) and **Marker** images:
- **Red pixels** in Seg (R>150, G<100, B<100) = positive nuclei
- **Blue pixels** in Seg (B>150, R<100, G<100) = negative nuclei
- **Marker** channel brightness = DAB staining intensity (used for H-score / Allred)

### Video / Batch
Consecutive frames often overlap. The deduplication algorithm:
1. Extracts per-cell centroids from each frame
2. Estimates inter-frame shift using **phase correlation** (`cv2.phaseCorrelate`)
3. Maps centroids to a global coordinate system
4. Skips any cell within the dedup radius of an already-registered cell

---

## API Reference

Endpoint: `https://deepliif.org/api/infer`

To swap in a local model or different backend, edit only `segmentation.py` - `app.py` and `scoring.py` do not need to change.

## License

See [DeepLIIF License](https://github.com/nadeemlab/DeepLIIF/blob/main/LICENSE).
