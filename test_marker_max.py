from PIL import Image
import numpy as np
from segmentation import segment
from skimage import measure

img = Image.open("./sample_data/cell.png")
result = segment(img, resolution="40x", nopost=True)

seg = result["images"].get("Seg") or result["images"].get("seg")
marker = result["images"].get("Marker") or result["images"].get("marker")

marker_l = np.array(marker.convert("L"))
marker_max = np.max(np.array(marker.convert("RGB")), axis=2)

seg_arr = np.array(seg.convert("RGB"))
r, g, b = seg_arr[:, :, 0], seg_arr[:, :, 1], seg_arr[:, :, 2]
pos_mask = (r > 150) & (g < 100) & (b < 100)

labeled = measure.label(pos_mask)
props_l = measure.regionprops(labeled, intensity_image=marker_l)
props_max = measure.regionprops(labeled, intensity_image=marker_max)

def get_mean(p):
    return float(p.intensity_mean) if hasattr(p, "intensity_mean") else float(p.mean_intensity)

means_l = [get_mean(p) for p in props_l if p.area >= 15]
means_max = [get_mean(p) for p in props_max if p.area >= 15]

print(f"L max: {max(means_l):.1f}, mean: {np.mean(means_l):.1f}")
print(f"MAX max: {max(means_max):.1f}, mean: {np.mean(means_max):.1f}")

def grade(m, t1, t2, t3):
    if m < t1: return 0
    if m < t2: return 1
    if m < t3: return 2
    return 3

print("Grades with L (64, 128, 192):")
gl = [grade(m, 64, 128, 192) for m in means_l]
print({0: gl.count(0), 1: gl.count(1), 2: gl.count(2), 3: gl.count(3)})

print("Grades with MAX (64, 128, 192):")
gmax = [grade(m, 64, 128, 192) for m in means_max]
print({0: gmax.count(0), 1: gmax.count(1), 2: gmax.count(2), 3: gmax.count(3)})

print("Grades with MAX (30, 100, 170):")
gmax2 = [grade(m, 30, 100, 170) for m in means_max]
print({0: gmax2.count(0), 1: gmax2.count(1), 2: gmax2.count(2), 3: gmax2.count(3)})

