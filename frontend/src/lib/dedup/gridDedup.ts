import type { Cell, DedupScoring } from "@/types";

interface GridCell {
  gx: number;
  gy: number;
  positive: boolean;
}

/**
 * Grid-based spatial deduplication of cells.
 * Port of deduplicate_video_cells from app.py (grid-only, no phase correlation).
 * Used for camera batch mode where frames have no motion.
 */
export function gridDedup(
  frameCells: Cell[][],
  dedupRadius: number = 20,
): DedupScoring {
  const grid = new Map<string, GridCell[]>();
  let rawTotal = 0;

  for (const cells of frameCells) {
    for (const cell of cells) {
      rawTotal++;
      const [cy, cx] = cell.centroid;
      const gix = Math.floor(cx / dedupRadius);
      const giy = Math.floor(cy / dedupRadius);

      let dup = false;
      outer:
      for (let dix = -1; dix <= 1; dix++) {
        for (let diy = -1; diy <= 1; diy++) {
          const key = `${gix + dix},${giy + diy}`;
          const bucket = grid.get(key);
          if (!bucket) continue;
          for (const existing of bucket) {
            const dist = Math.hypot(cx - existing.gx, cy - existing.gy);
            if (dist < dedupRadius) {
              dup = true;
              break outer;
            }
          }
        }
      }

      if (!dup) {
        const key = `${gix},${giy}`;
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key)!.push({ gx: cx, gy: cy, positive: cell.positive });
      }
    }
  }

  const unique: GridCell[] = [];
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
    note: `Deduplicated from ${rawTotal} raw detections across ${frameCells.length} frames (dedup radius = ${dedupRadius} px).`,
  };
}
