from PIL import Image
import numpy as np
from segmentation import segment

img = Image.open("./sample_data/cell.png")
result = segment(img, resolution="40x", nopost=True)
marker = result["images"].get("Marker") or result["images"].get("marker")

if marker:
    marker_arr = np.array(marker)
    print("Marker shape:", marker_arr.shape)
    print("Marker RGB min:", marker_arr.min(axis=(0,1)))
    print("Marker RGB max:", marker_arr.max(axis=(0,1)))
    print("Marker RGB mean:", marker_arr.mean(axis=(0,1)))
