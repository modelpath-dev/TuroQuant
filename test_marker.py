from PIL import Image
import numpy as np
from segmentation import segment

img = Image.open("./sample_data/cell.png")
result = segment(img, resolution="40x", nopost=True)

seg = result["images"].get("Seg") or result["images"].get("seg")
marker = result["images"].get("Marker") or result["images"].get("marker")

if marker:
    marker_arr = np.array(marker.convert("L"))
    print("Marker min:", marker_arr.min(), "max:", marker_arr.max(), "mean:", marker_arr.mean())
    
    seg_arr = np.array(seg.convert("RGB"))
    r, g, b = seg_arr[:, :, 0], seg_arr[:, :, 1], seg_arr[:, :, 2]
    pos_mask = (r > 150) & (g < 100) & (b < 100)
    
    pos_marker_vals = marker_arr[pos_mask]
    if len(pos_marker_vals) > 0:
        print("Positive mask Marker min:", pos_marker_vals.min(), "max:", pos_marker_vals.max(), "mean:", pos_marker_vals.mean())
        print("Pos pixels < 64:", (pos_marker_vals < 64).sum() / len(pos_marker_vals))
        print("Pos pixels 64-128:", ((pos_marker_vals >= 64) & (pos_marker_vals < 128)).sum() / len(pos_marker_vals))
        print("Pos pixels 128-192:", ((pos_marker_vals >= 128) & (pos_marker_vals < 192)).sum() / len(pos_marker_vals))
        print("Pos pixels > 192:", (pos_marker_vals >= 192).sum() / len(pos_marker_vals))

