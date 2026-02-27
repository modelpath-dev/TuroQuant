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

seg.save("sample_data/debug_seg.png")
marker.save("sample_data/debug_marker.png")

seg_arr = np.array(seg.convert("RGB"))
marker_rgb = np.array(marker.convert("RGB"))
marker_arr = np.max(marker_rgb, axis=2)

r, g, b = seg_arr[:, :, 0], seg_arr[:, :, 1], seg_arr[:, :, 2]
pos_mask = (r > 150) & (g < 100) & (b < 100)

labeled = measure.label(pos_mask)

# Calculate both mean and max intensity per cell
props_mean = measure.regionprops(labeled, intensity_image=marker_arr)

means = []
maxs = []

for p in props_mean:
    if p.area >= 15:
        m_val = float(p.intensity_mean) if hasattr(p, "intensity_mean") else float(p.mean_intensity)
        max_val = float(p.intensity_max) if hasattr(p, "intensity_max") else float(p.max_intensity)
        means.append(m_val)
        maxs.append(max_val)

print(f"Total positive cells: {len(means)}")
print(f"MEAN intensities -> min: {np.min(means):.1f}, max: {np.max(means):.1f}, avg: {np.mean(means):.1f}")
print(f"MAX  intensities -> min: {np.min(maxs):.1f}, max: {np.max(maxs):.1f}, avg: {np.mean(maxs):.1f}")

def grade(val, t1=85, t2=170):
    if val < t1: return 1
    if val < t2: return 2
    return 3

grades_mean = [grade(m) for m in means]
grades_max = [grade(m) for m in maxs]

print("Grades using MEAN:", {1: grades_mean.count(1), 2: grades_mean.count(2), 3: grades_mean.count(3)})
print("Grades using MAX:", {1: grades_max.count(1), 2: grades_max.count(2), 3: grades_max.count(3)})
