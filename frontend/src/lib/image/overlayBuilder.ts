import { POS_COLOR, NEG_COLOR } from "../constants";

/**
 * Build overlay image: positive cells in red, negative in blue, rest darkened.
 * Port of _build_overlay from app.py.
 */
export function buildOverlay(
  origImageData: ImageData,
  segImageData: ImageData,
): ImageData {
  const { width, height } = origImageData;
  const out = new ImageData(width, height);
  const orig = origImageData.data;
  const seg = segImageData.data;

  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    const r = seg[o];
    const g = seg[o + 1];
    const b = seg[o + 2];

    const isPos = r > 150 && g < 100 && b < 100;
    const isNeg = b > 150 && r < 100 && g < 100;

    if (isPos) {
      out.data[o] = POS_COLOR.r;
      out.data[o + 1] = POS_COLOR.g;
      out.data[o + 2] = POS_COLOR.b;
    } else if (isNeg) {
      out.data[o] = NEG_COLOR.r;
      out.data[o + 1] = NEG_COLOR.g;
      out.data[o + 2] = NEG_COLOR.b;
    } else {
      out.data[o] = Math.round(orig[o] * 0.45);
      out.data[o + 1] = Math.round(orig[o + 1] * 0.45);
      out.data[o + 2] = Math.round(orig[o + 2] * 0.45);
    }
    out.data[o + 3] = 255;
  }

  return out;
}
