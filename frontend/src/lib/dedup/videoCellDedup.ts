import type { Cell, DedupScoring, IntensityDistribution } from "@/types";
import { phaseCorrelate } from "./phaseCorrelation";
import { toGrayscale } from "../image/canvasUtils";
import { base64ToImageData } from "../image/canvasUtils";

interface FrameData {
  cells: Cell[];
  segBase64: string;
}

/**
 * Video cell deduplication with optical flow compensation.
 * Port of deduplicate_video_cells from app.py.
 */
export async function deduplicateVideoCells(
  frames: FrameData[],
  dedupRadius: number = 20,
): Promise<DedupScoring> {
  const grid = new Map<string, { gx: number; gy: number; positive: boolean }[]>();
  const cumulative = [0, 0];
  let prevGray: Float32Array | null = null;
  let prevWidth = 0;
  let prevHeight = 0;
  let rawTotal = 0;

  for (const frame of frames) {
    // Phase correlation for optical flow
    try {
      const segImageData = await base64ToImageData(frame.segBase64);
      const gray = toGrayscale(segImageData);
      const currGray = new Float32Array(gray);
      const w = segImageData.width;
      const h = segImageData.height;

      if (prevGray && prevWidth === w && prevHeight === h) {
        const [dx, dy] = phaseCorrelate(prevGray, currGray, w, h);
        cumulative[0] += dx;
        cumulative[1] += dy;
      }

      prevGray = currGray;
      prevWidth = w;
      prevHeight = h;
    } catch {
      // Skip shift estimation on error
    }

    // Deduplicate cells
    for (const cell of frame.cells) {
      rawTotal++;
      const [cy, cx] = cell.centroid;
      const gx = cx - cumulative[0];
      const gy = cy - cumulative[1];
      const gix = Math.floor(gx / dedupRadius);
      const giy = Math.floor(gy / dedupRadius);

      let dup = false;
      outer:
      for (let dix = -1; dix <= 1; dix++) {
        for (let diy = -1; diy <= 1; diy++) {
          const key = `${gix + dix},${giy + diy}`;
          const bucket = grid.get(key);
          if (!bucket) continue;
          for (const existing of bucket) {
            if (Math.hypot(gx - existing.gx, gy - existing.gy) < dedupRadius) {
              dup = true;
              break outer;
            }
          }
        }
      }

      if (!dup) {
        const key = `${gix},${giy}`;
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key)!.push({ gx, gy, positive: cell.positive });
      }
    }
  }

  const unique: { positive: boolean }[] = [];
  for (const bucket of grid.values()) {
    unique.push(...bucket);
  }

  const nTotal = unique.length;
  const nPos = unique.filter((c) => c.positive).length;
  const nNeg = nTotal - nPos;
  const pctPos = nTotal > 0 ? Math.round((nPos / nTotal) * 1000) / 10 : 0;

  return {
    numTotal: nTotal,
    numPos: nPos,
    numNeg: nNeg,
    percentPos: pctPos,
    rawDetections: rawTotal,
    note: `Deduplicated from ${rawTotal} raw detections across ${frames.length} frames (dedup radius = ${dedupRadius} px).`,
  };
}

/**
 * Aggregate intensity distributions across video frames for global ER/PR scoring.
 */
export function aggregateIntensityDistributions(
  distributions: IntensityDistribution[],
): IntensityDistribution {
  const total: IntensityDistribution = { 0: 0, 1: 0, 2: 0, 3: 0 };
  for (const dist of distributions) {
    total[0] += dist[0];
    total[1] += dist[1];
    total[2] += dist[2];
    total[3] += dist[3];
  }
  return total;
}
