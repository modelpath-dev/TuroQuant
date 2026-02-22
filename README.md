# TuroQuant Pipeline

A Streamlit web application for IHC (Immunohistochemistry) image quantification using the TuroQuant API and local postprocessing library.

Upload an IHC image, multi-page TIF, or video scan of a whole-slide image (WSI) and get:
- Multiplex immunofluorescence channel decomposition (DAPI, Hematoxylin, Lap2, Marker, Seg)
- Automated positive/negative cell counting with overlay visualisation
- For video input: phase-correlation-based cell deduplication across overlapping frames so cells are not double-counted

## Supported Formats

| Type | Formats |
|------|---------|
| Image | PNG, JPG/JPEG, BMP, GIF |
| Microscopy | TIF/TIFF (multi-page), SVS, NDPI, SCN, CZI, LIF, MRXS, VMS, VMU, QPTIFF |
| Video | MP4, AVI, MOV, MKV, WebM |

## Setup

### Prerequisites

- Python 3.10+
- Internet connection (for the TuroQuant inference API)

### Installation

```bash
git clone https://github.com/modelpath-dev/TuroQuant.git
cd TuroQuant

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install deepliif
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. **Upload** an image, TIF, or video file using the file uploader.
2. **Configure options** in the sidebar:
   - **Scan Resolution** (10x / 20x / 40x) — must match the resolution the slide was scanned at.
   - **Probability Threshold** — minimum probability for a pixel to be classified as a positive cell.
   - **Slim mode** — return only the segmentation image.
   - **Skip postprocessing** — bypass server-side postprocessing (useful if the server returns HTTP 500).
   - **Pillow loader** — use PIL instead of Bio-Formats for loading (faster for PNG/JPG).
3. **Click "Run TuroQuant"** to send the image to the API and run local cell counting.
4. **View results**:
   - Channel images (DAPI, Hema, Lap2, Marker, Seg, Overlay, Refined)
   - Cell count metrics: Total Cells, Positive, Negative, % Positive, Pos:Neg ratio
   - For video: deduplicated aggregate counts across all frames
5. **Download**:
   - Individual channel videos (for video input)
   - All channels + scoring JSON as a single ZIP
   - All frame PNGs + per-frame scoring as a ZIP
6. **Click "Process New Sample"** to clear results and start over.

## How Cell Counting Works

### Single Image / TIF

The app sends the image to the TuroQuant API which returns channel-decomposed images. It then runs local postprocessing using the **Seg** (segmentation) and **Marker** images to detect and classify individual cells:
- **Red cells** in the Seg image = positive (protein-expressing)
- **Blue cells** in the Seg image = negative

### Video (WSI Scan)

When processing a video scan of a whole-slide image, consecutive frames overlap significantly. Naive per-frame counting would double-count cells in overlapping regions.

The deduplication algorithm:
1. Extracts per-cell centroids from each frame using `compute_cell_results`
2. Estimates the inter-frame shift using **phase correlation** (`cv2.phaseCorrelate`) on the Seg images
3. Maps all cell centroids to a **global coordinate system** using the cumulative shift
4. Inserts cells into a **spatial grid** and skips any cell within a configurable dedup radius (default 20px) of an already-registered cell

The result is an accurate unique cell count across the entire slide scan.

## Project Structure

```
TuroQuant/
  app.py              # Streamlit application (all logic)
  requirements.txt    # Python dependencies
  sample_data/        # Example input files
    cell.png          # Sample IHC image
    sample2.mov       # Sample WSI video scan
```




