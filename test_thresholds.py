import numpy as np
from PIL import Image
from skimage import measure
from app import extract_video_frames
from segmentation import segment_video_frame

with open("sample_data/19_PGR_test.mov", "rb") as f:
    video_bytes = f.read()

frames = extract_video_frames(video_bytes, "mov", every_n_sec=1.0)
frame_arr = frames[0]

result = segment_video_frame(Image.fromarray(frame_arr), resolution="40x", nopost=True, use_pil=True)
seg = result["images"]["Seg"]
marker = result["images"]["Marker"]

seg_arr = np.array(seg.convert("RGB"))
marker_rgb = np.array(marker.convert("RGB"))
marker_max = np.max(marker_rgb, axis=2)

r, g, b = seg_arr[:, :, 0], seg_arr[:, :, 1], seg_arr[:, :, 2]
pos_mask = (r > 150) & (g < 100) & (b < 100)

labeled = measure.label(pos_mask)

# Max intensity
props = measure.regionprops(labeled, intensity_image=marker_max)
maxs = []

for p in props:
    if p.area >= 15:
        max_val = float(p.intensity_max) if hasattr(p, "intensity_max") else float(p.max_intensity)
        maxs.append(max_val)

def grade(val, t1, t2):
    if val < t1: return 1
    if val < t2: return 2
    return 3

print("Thresholds (40, 80):", {1: [grade(m, 40, 80) for m in maxs].count(1), 2: [grade(m, 40, 80) for m in maxs].count(2), 3: [grade(m, 40, 80) for m in maxs].count(3)})
print("Thresholds (50, 100):", {1: [grade(m, 50, 100) for m in maxs].count(1), 2: [grade(m, 50, 100) for m in maxs].count(2), 3: [grade(m, 50, 100) for m in maxs].count(3)})
