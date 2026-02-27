import cv2
from PIL import Image
import numpy as np
from app import extract_video_frames
from segmentation import segment_video_frame
from scoring import compute_global_erpr_score

with open("sample_data/19_PGR_test.mov", "rb") as f:
    video_bytes = f.read()

frames = extract_video_frames(video_bytes, "mov", every_n_sec=1.0)
print(f"Extracted {len(frames)} frames")

if frames:
    frame_arr = frames[0]
    result = segment_video_frame(Image.fromarray(frame_arr), resolution="40x", nopost=True, use_pil=True)
    
    seg = result["images"].get("Seg") or result["images"].get("seg")
    marker = result["images"].get("Marker") or result["images"].get("marker")
    
    if seg and not result.get("cells"):
        from scoring import score_image
        _sc = score_image(seg, marker, "PR")
        result["scoring"] = _sc
        result["cells"] = _sc.get("cells", [])

    print("Scoring for first frame:", result.get("scoring"))
