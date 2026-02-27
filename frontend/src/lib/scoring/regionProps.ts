import type { LabelResult, RegionProp } from "./types";
import { MIN_CELL_AREA } from "../constants";

/**
 * Compute region properties (area, centroid, max intensity) for each label.
 * Equivalent to skimage.measure.regionprops.
 */
export function regionProps(
  labelResult: LabelResult,
  intensityImage?: Uint8Array,
  minArea: number = MIN_CELL_AREA,
): RegionProp[] {
  const { labels, width, height } = labelResult;
  const stats = new Map<
    number,
    { area: number; sumY: number; sumX: number; maxIntensity: number }
  >();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const lbl = labels[idx];
      if (lbl === 0) continue;

      let s = stats.get(lbl);
      if (!s) {
        s = { area: 0, sumY: 0, sumX: 0, maxIntensity: 0 };
        stats.set(lbl, s);
      }
      s.area++;
      s.sumY += y;
      s.sumX += x;
      if (intensityImage) {
        s.maxIntensity = Math.max(s.maxIntensity, intensityImage[idx]);
      }
    }
  }

  const results: RegionProp[] = [];
  for (const [label, s] of stats) {
    if (s.area < minArea) continue;
    results.push({
      label,
      area: s.area,
      centroid: [s.sumY / s.area, s.sumX / s.area],
      maxIntensity: intensityImage ? s.maxIntensity : undefined,
    });
  }
  return results;
}
