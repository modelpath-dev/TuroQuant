from segmentation import segment
from PIL import Image

img = Image.open("./sample_data/cell.png")
result = segment(img, resolution="40x", nopost=True)
print(result["scoring"])
